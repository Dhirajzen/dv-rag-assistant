"""Spec library: ingest any specification PDF and manage one vector collection per spec.

Each spec lives in its own Chroma collection (named ``spec-<slug>``). The collection's
metadata is the registry entry (display name, source file, page/chunk counts), so there
is no separate index file to keep in sync.
"""
import os
import re
import time
from collections import Counter
from functools import lru_cache

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_PREFIX = "spec-"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBED_BATCH = 64
MIN_PAGE_CHARS = 50


# ============================================================
# Shared resources
# ============================================================
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


# ============================================================
# Registry
# ============================================================
def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50].strip("-")
    return slug or "spec"


def spec_id_for(name: str) -> str:
    return COLLECTION_PREFIX + slugify(name)


def list_specs() -> list[dict]:
    """All ingested specs, newest first."""
    specs = []
    for col in get_client().list_collections():
        if not col.name.startswith(COLLECTION_PREFIX):
            continue
        meta = col.metadata or {}
        specs.append({
            "id": col.name,
            "name": meta.get("display_name", col.name[len(COLLECTION_PREFIX):]),
            "source_file": meta.get("source_file", ""),
            "pages": meta.get("pages", 0),
            "pages_indexed": meta.get("pages_indexed", 0),
            "chunks": col.count(),
            "created_at": meta.get("created_at", 0),
        })
    return sorted(specs, key=lambda s: s["created_at"], reverse=True)


def get_spec(spec_id: str) -> dict | None:
    return next((s for s in list_specs() if s["id"] == spec_id), None)


def resolve_spec(name_or_id: str | None = None) -> dict:
    """Find a spec by id, slug or display name. With no argument, return the only/newest one."""
    specs = list_specs()
    if not specs:
        raise LookupError("No specs ingested yet. Run: python ingest.py <spec.pdf> --name <name>")
    if not name_or_id:
        return specs[0]
    wanted = {name_or_id, spec_id_for(name_or_id)}
    for s in specs:
        if s["id"] in wanted or s["name"].lower() == name_or_id.lower():
            return s
    raise LookupError(f"Unknown spec '{name_or_id}'. Available: {', '.join(s['name'] for s in specs)}")


def delete_spec(spec_id: str) -> None:
    if not spec_id.startswith(COLLECTION_PREFIX):
        raise ValueError(f"Not a spec collection: {spec_id}")
    get_client().delete_collection(spec_id)


def get_vectorstore(spec_id: str) -> Chroma:
    return Chroma(client=get_client(), collection_name=spec_id, embedding_function=get_embeddings())


# ============================================================
# Spec-agnostic PDF cleaning
# ============================================================
_TOC_LINE = re.compile(r"(\.{5,}|(\. ){4,}).{0,12}$")


def _line_key(line: str) -> str:
    """Normalise a line so page-to-page variants of a header/footer compare equal."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _edge_lines(text: str, head: int = 5, tail: int = 4) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) <= head + tail:
        return lines
    return lines[:head] + lines[-tail:]


def find_boilerplate(pages: list[str]) -> set[str]:
    """Detect running headers/footers: short lines that recur at page edges across many pages."""
    counts = Counter()
    for text in pages:
        counts.update({_line_key(l) for l in _edge_lines(text) if len(l.strip()) < 160})
    threshold = max(3, min(8, int(len(pages) * 0.3)))
    return {key for key, n in counts.items() if n >= threshold and key}


def strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    lines = text.splitlines()
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    edge = set(nonempty[:5] + nonempty[-4:])
    kept = [l for i, l in enumerate(lines) if not (i in edge and _line_key(l) in boilerplate)]
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_toc_page(text: str) -> bool:
    """Table-of-contents / list-of-figures pages: mostly dot-leader lines ending in a page ref."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    hits = sum(1 for l in lines if _TOC_LINE.search(l.rstrip()))
    return hits / len(lines) >= 0.3


# ============================================================
# Ingestion
# ============================================================
def ingest_pdf(pdf_path: str, name: str, source_file: str | None = None,
               skip_first_pages: int = 0, progress=None) -> dict:
    """Ingest a spec PDF into its own collection, replacing any spec with the same name.

    progress: optional callable(fraction: float, message: str).
    Returns the registry entry for the new spec.
    """
    def report(frac, msg):
        if progress:
            progress(frac, msg)

    name = name.strip()
    if not name:
        raise ValueError("Spec name is required.")
    spec_id = spec_id_for(name)

    report(0.02, "Reading PDF...")
    documents = PyPDFLoader(pdf_path).load()
    total_pages = len(documents)
    if not any(len(d.page_content.strip()) > MIN_PAGE_CHARS for d in documents):
        raise ValueError("No extractable text found. Is this a scanned PDF? Run OCR on it first.")

    report(0.10, f"Cleaning {total_pages} pages...")
    boilerplate = find_boilerplate([d.page_content for d in documents])
    kept = []
    for doc in documents:
        page_idx = doc.metadata.get("page", 0)
        if page_idx < skip_first_pages or is_toc_page(doc.page_content):
            continue
        text = strip_boilerplate(doc.page_content, boilerplate)
        if len(text) < MIN_PAGE_CHARS:
            continue
        doc.page_content = text
        # 1-indexed PDF page: what a PDF viewer's "go to page" expects.
        doc.metadata = {"page": page_idx + 1, "spec": name}
        kept.append(doc)
    if not kept:
        raise ValueError("Nothing left to index after cleaning. Try a smaller 'skip first pages' value.")

    report(0.15, "Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(kept)

    report(0.18, "Loading embedding model...")
    embeddings = get_embeddings()

    client = get_client()
    if any(c.name == spec_id for c in client.list_collections()):
        client.delete_collection(spec_id)
    client.create_collection(spec_id, metadata={
        "display_name": name,
        "source_file": source_file or os.path.basename(pdf_path),
        "pages": total_pages,
        "pages_indexed": len(kept),
        "created_at": time.time(),
    })
    store = Chroma(client=client, collection_name=spec_id, embedding_function=embeddings)
    try:
        for start in range(0, len(chunks), EMBED_BATCH):
            store.add_documents(chunks[start:start + EMBED_BATCH])
            done = min(start + EMBED_BATCH, len(chunks))
            report(0.2 + 0.8 * done / len(chunks), f"Embedding chunks {done}/{len(chunks)}...")
    except Exception:
        client.delete_collection(spec_id)  # don't leave a half-built spec in the library
        raise

    return get_spec(spec_id)

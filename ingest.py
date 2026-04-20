import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Config ---
PDF_PATH = "data/axi4_spec.pdf"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def clean_page_text(text):
    """Strip recurring ARM copyright headers and page numbers."""
    # Remove ARM copyright lines
    text = re.sub(r"ARM IHI 0022H.*?Non-Confidential", "", text, flags=re.DOTALL)
    text = re.sub(r"Copyright © 2003-2020 Arm Limited.*?reserved\.", "", text)
    text = re.sub(r"ID\d{6}", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def main():
    # 1. Load the PDF
    print(f"Loading {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

 
    for doc in documents:
        doc.page_content = clean_page_text(doc.page_content)

    # Drop pages that became basically empty after cleaning
    documents = [d for d in documents if len(d.page_content) > 100]
    print(f"After cleaning: {len(documents)} pages with meaningful content")

    # 1b. Skip front matter (TOC, intro) — adjust based on your spec
    SKIP_FIRST_N_PAGES = 25  # ARM AXI spec front matter ends around page 28-30
    documents = [d for d in documents if d.metadata.get("page", 0) >= SKIP_FIRST_N_PAGES]
    print(f"After skipping front matter: {len(documents)} pages")

    # 2. Split into chunks
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # was 1000
        chunk_overlap=300,    # was 200
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings (runs locally, no API call)
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Build & persist the vector database
    print("Building vector database (this may take a few minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✓ Vector DB saved to {CHROMA_DIR}")
    print(f"✓ Indexed {vectorstore._collection.count()} chunks")

if __name__ == "__main__":
    main()

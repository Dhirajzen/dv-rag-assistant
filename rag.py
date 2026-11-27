"""Spec Q&A: retrieve from the selected spec's collection and answer with page citations."""
import os
import sys
from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from specs import get_vectorstore, resolve_spec

load_dotenv()

# --- Config ---
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
TOP_K = 10  # how many chunks to retrieve

PROMPT_TEMPLATE = """You are an expert in digital design and hardware verification,
answering questions about the specification "{spec_name}".

Use the context below to answer the question. The context contains excerpts from
the spec, each tagged with its PDF page number — synthesize across excerpts to give
a complete answer. Be technical and precise. Always cite the page number(s) you used,
like (p. 41).

If the context genuinely doesn't contain the information needed, say so. But if
the context contains partial or related information, use it and explain what's there.
Do not answer from general knowledge that the context does not support.

Context:
{context}

Question: {question}

Answer:"""


@lru_cache(maxsize=4)
def get_llm(max_tokens: int = 1500):
    return ChatAnthropic(model=LLM_MODEL, max_tokens=max_tokens)


def retrieve(spec_id: str, query: str, k: int = TOP_K, fetch_k: int = 25):
    """MMR retrieval: relevant but diversified chunks from one spec."""
    return get_vectorstore(spec_id).max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=0.5
    )


def format_context(docs) -> str:
    return "\n\n".join(f"[page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs)


def query_spec(question: str, spec: dict) -> dict:
    """Answer a question against one spec. `spec` is a registry entry from specs.list_specs()."""
    docs = retrieve(spec["id"], question)
    prompt = PROMPT_TEMPLATE.format(
        spec_name=spec["name"], context=format_context(docs), question=question
    )
    answer = get_llm().invoke(prompt).text
    return {
        "answer": answer,
        "sources": [
            {"page": d.metadata.get("page", "?"), "snippet": d.page_content[:300]}
            for d in docs
        ],
    }


def main():
    try:
        spec = resolve_spec(sys.argv[1] if len(sys.argv) > 1 else None)
    except LookupError as e:
        sys.exit(str(e))
    print(f"Spec Assistant — ask questions about: {spec['name']}")
    print("Type 'quit' to exit\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        result = query_spec(q, spec)
        print(f"\nA: {result['answer']}\n\n--- Sources used ---")
        for i, src in enumerate(result["sources"], 1):
            snippet = src["snippet"][:120].replace("\n", " ")
            print(f"  [{i}] page {src['page']}: {snippet}...")
        print()


if __name__ == "__main__":
    main()

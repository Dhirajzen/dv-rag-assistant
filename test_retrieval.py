"""Inspect retrieval quality for one spec.

    python test_retrieval.py "AXI4" "Explain the handshake mechanism"
"""
import sys

from specs import get_vectorstore, resolve_spec

spec = resolve_spec(sys.argv[1] if len(sys.argv) > 1 else None)
query = sys.argv[2] if len(sys.argv) > 2 else "Give an overview of the main interface signals"

print(f"\nSpec: {spec['name']}\nQuery: {query}\n" + "=" * 60)
for i, doc in enumerate(get_vectorstore(spec["id"]).similarity_search(query, k=3), 1):
    print(f"\n--- Result {i} (page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:400] + "...")

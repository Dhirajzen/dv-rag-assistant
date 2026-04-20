from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

query = "Explain the AXI handshake mechanism"
results = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: {query}\n")
print("=" * 60)
for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} (page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:400])
    print("...")

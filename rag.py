import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- Config ---
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-5"  # or "claude-haiku-4-5" for cheaper/faster
TOP_K = 10  # how many chunks to retrieve

PROMPT_TEMPLATE = """You are an expert in digital design and hardware verification,
answering questions about the AXI4 protocol specification.

Use the context below to answer the question. The context contains excerpts from
the spec — synthesize across chunks to give a complete answer. Be technical and
precise. Always cite the page number(s) you used.

If the context genuinely doesn't contain the information needed, say so. But if
the context contains partial or related information, use it and explain what's there.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain():
    # 1. Load the existing vector DB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # 2. Set up retriever (pulls top-k chunks by similarity)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 25, "lambda_mult": 0.5}
    )

    # 3. Set up Claude via LangChain
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0,   # deterministic answers for technical Q&A
        max_tokens=1024,
    )

    # 4. Custom prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # 5. Chain it all together
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # "stuff" = put all chunks into one prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def ask(qa_chain, question):
    print(f"\n{'='*70}")
    print(f"Q: {question}")
    print(f"{'='*70}")

    result = qa_chain.invoke({"query": question})

    print(f"\nA: {result['result']}")

    print(f"\n--- Sources used ---")
    for i, doc in enumerate(result["source_documents"], 1):
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] page {page}: {snippet}...")


def main():
    qa_chain = build_qa_chain()
    print("AXI4 Spec Assistant — ask questions about the AXI4 protocol")
    print("Type 'quit' to exit\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        ask(qa_chain, q)

if __name__ == "__main__":
    main()

def query_spec(question: str, qa_chain=None):
    """Single-call interface for use from other modules (e.g., Streamlit)."""
    if qa_chain is None:
        qa_chain = build_qa_chain()
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:300]
            }
            for doc in result["source_documents"]
        ]
    }
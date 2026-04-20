import streamlit as st
from rag import build_qa_chain, query_spec
from agent import run_coverage_agent

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="DV RAG Assistant",
    page_icon="🔧",
    layout="wide"
)

# Cache the RAG chain so we don't rebuild it every interaction
@st.cache_resource
def get_qa_chain():
    return build_qa_chain()

# ============================================================
# Header
# ============================================================
st.title("🔧 DV RAG Assistant")
st.markdown(
    "**LLM-powered verification assistant for the AXI4 protocol.** "
    "Ask questions about the spec or analyze coverage reports for gaps."
)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["💬 Spec Q&A", "📊 Coverage Analyzer", "ℹ️ About"])

# ============================================================
# Tab 1: Spec Q&A
# ============================================================
with tab1:
    st.header("Ask the AXI4 Specification")
    st.caption("RAG-powered Q&A grounded in the 500-page ARM AXI4 spec (IHI 0022H).")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Sources ({len(msg['sources'])} chunks)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] page {src['page']}**")
                        st.text(src["snippet"] + "...")

    # Chat input
    if question := st.chat_input("e.g., How does the AXI handshake work?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching spec and generating answer..."):
                qa_chain = get_qa_chain()
                result = query_spec(question, qa_chain)
            st.markdown(result["answer"])
            with st.expander(f"📄 Sources ({len(result['sources'])} chunks)"):
                for i, src in enumerate(result["sources"], 1):
                    st.markdown(f"**[{i}] page {src['page']}**")
                    st.text(src["snippet"] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

# ============================================================
# Tab 2: Coverage Analyzer
# ============================================================
with tab2:
    st.header("Coverage Gap Analyzer")
    st.caption(
        "Upload or paste a functional coverage report. "
        "An agentic LangGraph workflow will identify uncovered bins, "
        "retrieve relevant spec sections, and suggest UVM stimulus to close them."
    )

    # Two ways to provide the report
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload coverage report (.txt)", type=["txt"])
    with col2:
        st.markdown("**Or paste below:**")

    default_text = ""
    if uploaded:
        default_text = uploaded.read().decode("utf-8")

    coverage_text = st.text_area(
        "Coverage report contents",
        value=default_text,
        height=300,
        placeholder="Paste your coverage report here..."
    )

    if st.button("🚀 Analyze Coverage", type="primary", disabled=not coverage_text):
        with st.spinner("Running agent... this takes 1-2 minutes for ~6 gaps."):
            result = run_coverage_agent(coverage_text)

        st.success(f"Analysis complete — {len(result['gaps'])} gaps identified.")

        # Show gaps as a quick summary
        st.subheader("Identified Gaps")
        for gap in result["gaps"]:
            st.markdown(f"- `{gap}`")

        # Show full report
        st.subheader("Suggested Stimulus")
        st.markdown(result["report"])

        # Download button
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=result["report"],
            file_name="coverage_analysis.md",
            mime="text/markdown"
        )

# ============================================================
# Tab 3: About
# ============================================================
with tab3:
    st.header("About This Project")
    st.markdown("""
    ### Architecture
    - **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, 384-dim)
    - **Vector DB**: ChromaDB (persistent, on-disk)
    - **LLM**: Anthropic Claude Sonnet 4.5
    - **RAG framework**: LangChain (with MMR retrieval)
    - **Agent framework**: LangGraph (stateful multi-node workflow)
    - **UI**: Streamlit

    ### Pipeline
    1. PDF ingestion with header/footer cleaning and front-matter filtering
    2. Recursive chunking (1500 chars, 300 overlap)
    3. Local embedding generation
    4. Persistent vector storage in ChromaDB
    5. MMR-based retrieval for diversified context
    6. LLM synthesis with grounded prompts and citations

    ### Coverage Analyzer Agent (LangGraph)
    Parse Report → [for each gap: Retrieve Spec → Suggest Stimulus] → Assemble Report
    
                
    Built by Dhirajzen Bagawath Geetha Kumaravel — MS Computer Engineering, NYU.
    """)
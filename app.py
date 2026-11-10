import os
import tempfile

import streamlit as st

from agent import run_coverage_agent
from rag import LLM_MODEL, query_spec
from specs import delete_spec, ingest_pdf, list_specs, spec_id_for

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="DV RAG Assistant", page_icon="🔧", layout="wide")

# A spec chosen programmatically (after ingest) is applied here, before the selectbox exists.
if "pending_spec" in st.session_state:
    st.session_state.active_spec = st.session_state.pop("pending_spec")
st.session_state.setdefault("chats", {})          # spec_id -> list of messages
st.session_state.setdefault("uploader_key", 0)    # bump to clear the file uploader

specs = list_specs()
spec_by_id = {s["id"]: s for s in specs}
if st.session_state.get("active_spec") not in spec_by_id:
    # Set explicitly (not just clear) so the selectbox drops a stale label after a delete.
    if specs:
        st.session_state.active_spec = specs[0]["id"]
    else:
        st.session_state.pop("active_spec", None)

# ============================================================
# Sidebar: spec library
# ============================================================
with st.sidebar:
    st.header("📚 Spec Library")

    if specs:
        st.selectbox(
            "Active spec",
            options=list(spec_by_id),
            format_func=lambda sid: spec_by_id[sid]["name"],
            key="active_spec",
        )
    else:
        st.info("No specs yet. Add one below.")

    spec = spec_by_id.get(st.session_state.get("active_spec"))
    if spec:
        st.caption(
            f"📄 {spec['source_file']}  \n"
            f"{spec['pages_indexed']} of {spec['pages']} pages indexed · {spec['chunks']} chunks"
        )

    st.divider()
    with st.expander("➕ Add a spec", expanded=not specs):
        uploaded_pdf = st.file_uploader(
            "Specification PDF", type=["pdf"], key=f"pdf_{st.session_state.uploader_key}"
        )
        default_name = ""
        if uploaded_pdf:
            default_name = os.path.splitext(uploaded_pdf.name)[0].replace("_", " ")
        spec_name = st.text_input("Spec name", value=default_name, placeholder="e.g. APB4, PCIe 6.0")
        skip_pages = st.number_input(
            "Skip first N pages", min_value=0, value=0,
            help="Contents pages and running headers/footers are detected and removed "
                 "automatically. Use this only to force-skip extra front matter (licence, preface).",
        )
        replacing = bool(spec_name.strip()) and spec_id_for(spec_name) in spec_by_id
        if replacing:
            st.warning(f"This replaces the existing '{spec_by_id[spec_id_for(spec_name)]['name']}'.")

        if st.button("Ingest spec", type="primary", disabled=not (uploaded_pdf and spec_name.strip())):
            bar = st.progress(0.0, text="Starting...")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_pdf.getbuffer())
            try:
                new_spec = ingest_pdf(
                    tmp.name, spec_name, source_file=uploaded_pdf.name,
                    skip_first_pages=int(skip_pages),
                    progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg),
                )
            except Exception as e:
                bar.empty()
                st.error(f"Ingestion failed: {e}")
            else:
                st.session_state.chats.pop(new_spec["id"], None)
                st.session_state.pending_spec = new_spec["id"]
                st.session_state.uploader_key += 1
                st.rerun()
            finally:
                os.unlink(tmp.name)

    if spec:
        with st.expander("🗑️ Remove active spec"):
            st.caption(f"Deletes the index for **{spec['name']}**. The PDF itself is not stored here.")
            if st.button("Remove", type="secondary"):
                delete_spec(spec["id"])
                st.session_state.chats.pop(spec["id"], None)
                st.rerun()

# ============================================================
# Header
# ============================================================
st.title("🔧 DV RAG Assistant")
st.markdown(
    "**LLM-powered verification assistant for any specification.** "
    "Upload a spec PDF, ask it questions, or analyze coverage reports for gaps."
)

tab1, tab2, tab3 = st.tabs(["💬 Spec Q&A", "📊 Coverage Analyzer", "ℹ️ About"])


def explain_error(e: Exception) -> str:
    if type(e).__name__ == "AuthenticationError" or "api_key" in str(e).lower():
        return ("Anthropic rejected the API key. Set a valid `ANTHROPIC_API_KEY` in `.env` "
                "and restart the app.")
    return str(e)


def show_sources(sources):
    with st.expander(f"📄 Sources ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**[{i}] page {src['page']}**")
            st.text(src["snippet"] + "...")


# ============================================================
# Tab 1: Spec Q&A
# ============================================================
with tab1:
    if not spec:
        st.info("👈 Add a specification PDF in the sidebar to get started.")
    else:
        st.header(f"Ask the {spec['name']} spec")
        st.caption(f"RAG-powered Q&A grounded in {spec['source_file']} ({spec['pages']} pages), "
                   "with page citations.")

        messages = st.session_state.chats.setdefault(spec["id"], [])
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    show_sources(msg["sources"])

        if question := st.chat_input(f"Ask anything about {spec['name']}..."):
            messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching spec and generating answer..."):
                        result = query_spec(question, spec)
                except Exception as e:
                    st.error(f"Could not answer: {explain_error(e)}")
                else:
                    st.markdown(result["answer"])
                    show_sources(result["sources"])
                    messages.append({"role": "assistant", "content": result["answer"],
                                     "sources": result["sources"]})

# ============================================================
# Tab 2: Coverage Analyzer
# ============================================================
with tab2:
    st.header("Coverage Gap Analyzer")
    if not spec:
        st.info("👈 Add a specification PDF in the sidebar to get started.")
    else:
        st.caption(
            f"Upload or paste a functional coverage report. A LangGraph agent identifies uncovered "
            f"bins, retrieves the relevant **{spec['name']}** spec sections, and suggests UVM "
            f"stimulus to close them."
        )

        uploaded = st.file_uploader("Upload coverage report", type=["txt", "log", "rpt", "md"])
        default_text = uploaded.read().decode("utf-8", errors="replace") if uploaded else ""
        coverage_text = st.text_area(
            "Coverage report contents", value=default_text, height=300,
            placeholder="...or paste your coverage report here",
        )

        if st.button("🚀 Analyze Coverage", type="primary", disabled=not coverage_text.strip()):
            result = None
            with st.status("Running agent...", expanded=True) as status:
                def on_step(node, state):
                    total = len(state["gaps"])
                    if node == "parse":
                        st.write(f"🔎 Found **{total}** uncovered bins")
                    elif node == "suggest":
                        done = state["current_gap_index"]
                        st.write(f"✏️ [{done}/{total}] `{state['gaps'][done - 1]}`")
                try:
                    result = run_coverage_agent(coverage_text, spec, on_step=on_step)
                    status.update(label="Analysis complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Agent failed", state="error")
                    st.error(explain_error(e))
            if result:
                st.session_state.coverage_result = {"spec_id": spec["id"], **result}

        result = st.session_state.get("coverage_result")
        if result and result["spec_id"] == spec["id"]:
            if not result["gaps"]:
                st.success("No uncovered bins found in this report.")
            else:
                st.success(f"{len(result['gaps'])} gaps analyzed against {spec['name']}.")
                st.subheader("Identified Gaps")
                st.markdown("\n".join(f"- `{gap}`" for gap in result["gaps"]))
                st.subheader("Suggested Stimulus")
                st.markdown(result["report"])
                st.download_button(
                    label="📥 Download Report (Markdown)", data=result["report"],
                    file_name="coverage_analysis.md", mime="text/markdown",
                )

# ============================================================
# Tab 3: About
# ============================================================
with tab3:
    st.header("About This Project")
    st.markdown(f"""
    ### Architecture
    - **Spec library**: one ChromaDB collection per uploaded spec, switchable from the sidebar
    - **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local, 384-dim)
    - **Vector DB**: ChromaDB (persistent, on-disk)
    - **LLM**: Anthropic Claude (`{LLM_MODEL}`)
    - **Agent framework**: LangGraph (stateful multi-node workflow)
    - **UI**: Streamlit

    ### Pipeline
    1. PDF ingestion with automatic header/footer detection and contents-page filtering
    2. Recursive chunking (1500 chars, 300 overlap)
    3. Local embedding generation
    4. Persistent vector storage in ChromaDB, one collection per spec
    5. MMR-based retrieval for diversified context
    6. LLM synthesis with grounded prompts and page citations

    ### Coverage Analyzer Agent (LangGraph)
    Parse Report → [for each gap: Retrieve Spec → Suggest Stimulus] → Assemble Report

    Built by Dhirajzen Bagawath Geetha Kumaravel — MS Computer Engineering, NYU.
    """)

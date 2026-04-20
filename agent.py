import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

# --- Config ---
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-5"

# --- Initialize once ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
llm = ChatAnthropic(model=LLM_MODEL, temperature=0, max_tokens=2048)


# ============================================================
# State definition — what flows through the graph
# ============================================================
class AgentState(TypedDict):
    coverage_report: str           # raw input
    gaps: List[str]                # parsed list of uncovered bins
    current_gap_index: int         # loop counter
    spec_context: dict             # gap_name -> retrieved spec text
    suggestions: dict              # gap_name -> stimulus suggestion
    final_report: str              # assembled output


# ============================================================
# Node 1: Parse coverage report → list of gaps
# ============================================================
PARSE_PROMPT = """Extract the list of UNCOVERED bins from this coverage report.
Return ONLY the bin names, one per line, no extra text.

Report:
{report}

Uncovered bins (one per line):"""

def parse_coverage(state: AgentState) -> AgentState:
    print("\n[Node: parse_coverage]")
    prompt = PARSE_PROMPT.format(report=state["coverage_report"])
    response = llm.invoke(prompt).content
    gaps = [line.strip() for line in response.split("\n") if line.strip()]
    print(f"  Found {len(gaps)} coverage gaps:")
    for g in gaps:
        print(f"    - {g}")
    return {**state, "gaps": gaps, "current_gap_index": 0,
            "spec_context": {}, "suggestions": {}}


# ============================================================
# Node 2: Retrieve spec context for current gap
# ============================================================
def retrieve_spec(state: AgentState) -> AgentState:
    idx = state["current_gap_index"]
    gap = state["gaps"][idx]
    print(f"\n[Node: retrieve_spec] Gap {idx+1}/{len(state['gaps'])}: {gap}")

    # Build a search query from the gap name
    query = f"AXI4 protocol behavior for {gap.replace('_', ' ').replace('.', ' ')}"
    docs = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=15)
    context = "\n\n".join([f"[page {d.metadata.get('page','?')}] {d.page_content}"
                           for d in docs])

    print(f"  Retrieved {len(docs)} spec chunks")
    new_context = {**state["spec_context"], gap: context}
    return {**state, "spec_context": new_context}


# ============================================================
# Node 3: Generate stimulus suggestion
# ============================================================
STIMULUS_PROMPT = """You are a senior Design Verification engineer.

A coverage gap has been identified in our AXI4 verification environment.

COVERAGE GAP: {gap}

RELEVANT SPEC CONTEXT:
{spec_context}

Suggest a specific UVM constrained-random stimulus scenario that would
hit this coverage point. Include:
1. A brief description of the scenario (1-2 sentences)
2. A SystemVerilog/UVM code sketch showing the constraint or sequence

Be concrete and protocol-accurate. Cite the spec page if you use specific values.

Suggestion:"""

def suggest_stimulus(state: AgentState) -> AgentState:
    idx = state["current_gap_index"]
    gap = state["gaps"][idx]
    print(f"[Node: suggest_stimulus] Generating stimulus for: {gap}")

    prompt = STIMULUS_PROMPT.format(
        gap=gap,
        spec_context=state["spec_context"][gap]
    )
    response = llm.invoke(prompt).content

    new_suggestions = {**state["suggestions"], gap: response}
    return {**state, "suggestions": new_suggestions,
            "current_gap_index": idx + 1}


# ============================================================
# Conditional edge: more gaps to process?
# ============================================================
def should_continue(state: AgentState) -> str:
    if state["current_gap_index"] < len(state["gaps"]):
        return "continue"
    return "done"


# ============================================================
# Node 4: Assemble final report
# ============================================================
def assemble_report(state: AgentState) -> AgentState:
    print("\n[Node: assemble_report]")
    parts = ["# Coverage Gap Analysis Report\n"]
    parts.append(f"**Total gaps analyzed:** {len(state['gaps'])}\n\n---\n")

    for i, gap in enumerate(state["gaps"], 1):
        parts.append(f"\n## {i}. `{gap}`\n")
        parts.append(state["suggestions"].get(gap, "(no suggestion)"))
        parts.append("\n---\n")

    return {**state, "final_report": "\n".join(parts)}


# ============================================================
# Build the graph
# ============================================================
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("parse", parse_coverage)
    graph.add_node("retrieve", retrieve_spec)
    graph.add_node("suggest", suggest_stimulus)
    graph.add_node("assemble", assemble_report)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "retrieve")
    graph.add_edge("retrieve", "suggest")
    graph.add_conditional_edges(
        "suggest",
        should_continue,
        {"continue": "retrieve", "done": "assemble"}
    )
    graph.add_edge("assemble", END)

    return graph.compile()


# ============================================================
# Main
# ============================================================
def main():
    with open("data/coverage_report.txt") as f:
        report = f.read()

    agent = build_agent()
    result = agent.invoke({
        "coverage_report": report,
        "gaps": [],
        "current_gap_index": 0,
        "spec_context": {},
        "suggestions": {},
        "final_report": ""
    })

    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(result["final_report"])

    # Save it
    with open("coverage_analysis_output.md", "w") as f:
        f.write(result["final_report"])
    print(f"\n✓ Report saved to coverage_analysis_output.md")


if __name__ == "__main__":
    main()

def run_coverage_agent(coverage_report: str):
    """Single-call interface for use from other modules."""
    agent = build_agent()
    result = agent.invoke({
        "coverage_report": coverage_report,
        "gaps": [],
        "current_gap_index": 0,
        "spec_context": {},
        "suggestions": {},
        "final_report": ""
    })
    return {
        "gaps": result["gaps"],
        "suggestions": result["suggestions"],
        "report": result["final_report"]
    }
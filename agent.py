"""Coverage gap analyzer: LangGraph agent that works against whichever spec is selected."""
import sys
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from rag import format_context, get_llm, retrieve
from specs import resolve_spec


# ============================================================
# State definition — what flows through the graph
# ============================================================
class AgentState(TypedDict):
    spec_id: str                   # which spec collection to retrieve from
    spec_name: str                 # display name, used in prompts
    coverage_report: str           # raw input
    gaps: List[str]                # parsed list of uncovered bins
    gap_descriptions: dict         # gap_name -> short description from the report
    current_gap_index: int         # loop counter
    spec_context: dict             # gap_name -> retrieved spec text
    suggestions: dict              # gap_name -> stimulus suggestion
    final_report: str              # assembled output


def _llm():
    return get_llm(max_tokens=2048)


# ============================================================
# Node 1: Parse coverage report → list of gaps
# ============================================================
PARSE_PROMPT = """Extract the list of UNCOVERED bins from this functional coverage report.
Return one bin per line in the form:
<bin name> :: <short plain-English description of what the bin covers>
Use the description from the report if there is one, otherwise infer it from the bin name.
Return ONLY those lines, no extra text. If nothing is uncovered, return nothing.

Report:
{report}

Uncovered bins:"""


def parse_coverage(state: AgentState) -> AgentState:
    response = _llm().invoke(PARSE_PROMPT.format(report=state["coverage_report"])).text
    gaps, descriptions = [], {}
    for line in response.split("\n"):
        name, _, desc = line.strip().lstrip("-*• ").partition("::")
        name = name.strip().strip("`")
        if name and name not in descriptions:
            gaps.append(name)
            descriptions[name] = desc.strip()
    return {**state, "gaps": gaps, "gap_descriptions": descriptions,
            "current_gap_index": 0, "spec_context": {}, "suggestions": {}}


# ============================================================
# Node 2: Retrieve spec context for current gap
# ============================================================
def retrieve_spec(state: AgentState) -> AgentState:
    gap = state["gaps"][state["current_gap_index"]]
    words = gap.replace("_", " ").replace(".", " ")
    query = f"{state['spec_name']}: {words}. {state['gap_descriptions'].get(gap, '')}".strip()
    docs = retrieve(state["spec_id"], query, k=4, fetch_k=15)
    return {**state, "spec_context": {**state["spec_context"], gap: format_context(docs)}}


# ============================================================
# Node 3: Generate stimulus suggestion
# ============================================================
STIMULUS_PROMPT = """You are a senior Design Verification engineer.

A coverage gap has been identified in a verification environment for a design
that implements the specification "{spec_name}".

COVERAGE GAP: {gap}
DESCRIPTION: {description}

RELEVANT SPEC CONTEXT:
{spec_context}

Suggest a specific UVM constrained-random stimulus scenario that would
hit this coverage point. Include:
1. A brief description of the scenario (1-2 sentences)
2. A SystemVerilog/UVM code sketch showing the constraint or sequence

Be concrete and accurate to the spec. Cite the spec page (p. N) when you rely on
specific values or rules. If the spec context does not cover this gap, say so
rather than inventing spec behaviour.

Suggestion:"""


def suggest_stimulus(state: AgentState) -> AgentState:
    idx = state["current_gap_index"]
    gap = state["gaps"][idx]
    prompt = STIMULUS_PROMPT.format(
        spec_name=state["spec_name"],
        gap=gap,
        description=state["gap_descriptions"].get(gap) or "(none given)",
        spec_context=state["spec_context"][gap],
    )
    response = _llm().invoke(prompt).text
    return {**state, "suggestions": {**state["suggestions"], gap: response},
            "current_gap_index": idx + 1}


# ============================================================
# Conditional edges
# ============================================================
def has_gaps(state: AgentState) -> str:
    return "continue" if state["gaps"] else "done"


def should_continue(state: AgentState) -> str:
    return "continue" if state["current_gap_index"] < len(state["gaps"]) else "done"


# ============================================================
# Node 4: Assemble final report
# ============================================================
def assemble_report(state: AgentState) -> AgentState:
    parts = [f"# Coverage Gap Analysis Report — {state['spec_name']}\n",
             f"**Total gaps analyzed:** {len(state['gaps'])}\n\n---\n"]
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
    graph.add_conditional_edges("parse", has_gaps, {"continue": "retrieve", "done": "assemble"})
    graph.add_edge("retrieve", "suggest")
    graph.add_conditional_edges(
        "suggest", should_continue, {"continue": "retrieve", "done": "assemble"}
    )
    graph.add_edge("assemble", END)

    return graph.compile()


def run_coverage_agent(coverage_report: str, spec: dict, on_step=None):
    """Run the agent against one spec.

    spec: registry entry from specs.list_specs().
    on_step: optional callable(node_name, state) fired after each graph node, for progress UIs.
    """
    initial = {
        "spec_id": spec["id"], "spec_name": spec["name"],
        "coverage_report": coverage_report,
        "gaps": [], "gap_descriptions": {}, "current_gap_index": 0,
        "spec_context": {}, "suggestions": {}, "final_report": "",
    }
    state = initial
    # Each gap takes 2 graph steps, so lift the default 25-step recursion limit.
    for update in build_agent().stream(initial, {"recursion_limit": 1000}, stream_mode="updates"):
        for node, state in update.items():
            if on_step:
                on_step(node, state)
    return {"gaps": state["gaps"], "suggestions": state["suggestions"],
            "report": state["final_report"]}


# ============================================================
# CLI: python agent.py [spec name] [coverage report path]
# ============================================================
def main():
    try:
        spec = resolve_spec(sys.argv[1] if len(sys.argv) > 1 else None)
    except LookupError as e:
        sys.exit(str(e))
    report_path = sys.argv[2] if len(sys.argv) > 2 else "data/coverage_report.txt"
    with open(report_path) as f:
        report = f.read()

    def on_step(node, state):
        if node == "parse":
            print(f"Found {len(state['gaps'])} coverage gaps: {', '.join(state['gaps'])}")
        elif node == "suggest":
            print(f"  [{state['current_gap_index']}/{len(state['gaps'])}] "
                  f"{state['gaps'][state['current_gap_index'] - 1]}")

    print(f"Analyzing {report_path} against spec: {spec['name']}")
    result = run_coverage_agent(report, spec, on_step=on_step)
    print("\n" + "=" * 70 + "\n" + result["report"])
    with open("coverage_analysis_output.md", "w") as f:
        f.write(result["report"])
    print("\n✓ Report saved to coverage_analysis_output.md")


if __name__ == "__main__":
    main()

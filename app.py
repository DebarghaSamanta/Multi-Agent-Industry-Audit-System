from langgraph.graph import StateGraph, END
from shared import ResearchState
from brain import researcher_node, auditor_node,planner_node

# 1. Create the Logic Flow
workflow = StateGraph(ResearchState)

# 2. Add our workers to the office
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("auditor", auditor_node)

# 3. Define the rules of the office
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "auditor")

# 4. The Decision Point: Should we finish or loop back?
def router(state: ResearchState):
    if state["is_verified"] or state["iterations"] > 3:
        return "end"
    return "researcher"

workflow.add_conditional_edges("auditor", router, {"end": END, "researcher": "researcher"})

# 5. Compile and Run
app = workflow.compile()

if __name__ == "__main__":
    inputs = {"topic": "Who won the evening match of T20 worldcup  on 7th  February 2026 ", "iterations": 0}
    for output in app.stream(inputs):
        print(output)
        print("-" * 30)
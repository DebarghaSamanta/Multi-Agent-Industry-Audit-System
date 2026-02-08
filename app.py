import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import your custom modules
from shared import ResearchState
from brain import researcher_node, auditor_node, planner_node

# 1. Initialize the Graph
workflow = StateGraph(ResearchState)

# 2. Register our Agents (Nodes)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("auditor", auditor_node)

# 3. Define the Orchestration (The "Office Rules")
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "auditor")

# 4. The Decision Logic (The "Router")
def router(state: ResearchState):
    """
    This function acts as the gatekeeper. 
    It reads the Auditor's 'is_verified' flag and decides the next move.
    """
    # Success Case: Auditor is happy
    if state.get("is_verified"):
        print("\n✅ AUDIT SUCCESSFUL: Fact-checked and verified.")
        return "end"
    
    # Safety Valve: Prevent infinite loops/cost overruns
    if state.get("iterations", 0) >= 3:
        print("\n🚨 CRITICAL: Max audit iterations reached. Outputting best-effort report.")
        return "end"
    
    # Failure Case: Loop back for better data
    print(f"\n❌ AUDIT REJECTED: {state.get('critique', 'Reason unknown')}")
    print("🔄 Restarting Research Cycle...")
    return "researcher"

# 5. Connect the Router to the Graph
workflow.add_conditional_edges(
    "auditor", 
    router, 
    {
        "end": END, 
        "researcher": "researcher"
    }
)

# 6. Compile the Application
app = workflow.compile()

# 7. Execution Block: This is where you write your questions
if __name__ == "__main__":
    load_dotenv()
    
    print("="*60)
    print("🚀 ENTERPRISE DUE DILIGENCE AGENT (VERSION 2026.1)")
    print("="*60)

    # You can change this query to any of the test cases we discussed
    test_query = "Evaluate Glean's competitive position against Microsoft Copilot and Perplexity for Enterprise. Focus on 'Switching Costs' and 'Data Advantage.' Is there evidence of customer churn from legacy intranet providers moving to Glean?"

    initial_input = {
        "topic": test_query,
        "iterations": 0,
        "is_verified": False,
        "plan": [],
        "raw_data": []
    }

    print(f"Target: {test_query}\n")

    # We use streaming so you can see the 'thinking' process in the terminal
    try:
        for output in app.stream(initial_input):
            for node_name, state_update in output.items():
                print(f"📍 Finished Node: {node_name.upper()}")
                
                # Visual debug for your resume/portfolio
                if node_name == "planner":
                    print(f"   Strategy: {state_update.get('plan')}")
                elif node_name == "auditor" and not state_update.get("is_verified"):
                    print(f"   Red Flags: {state_update.get('critique')}")

        # Final Result Extraction
        # We run the app one last time to get the final state
        final_state = app.invoke(initial_input)
        
        print("\n" + "⭐" * 20 + " FINAL VERIFIED REPORT " + "⭐" * 20)
        print(final_state.get("draft_report", "Research failed to generate report."))
        print("="*60)

    except Exception as e:
        print(f"❌ System Crash: {e}")
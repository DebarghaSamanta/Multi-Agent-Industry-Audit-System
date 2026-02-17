import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Import your custom modules
from shared import ResearchState
from brain import evaluator_node, researcher_node, auditor_node, planner_node

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
workflow.add_edge("evaluator", END)
# 4. The Decision Logic (The "Router")
def router(state: ResearchState):
    # --- X-RAY DEBUGGING START ---
    print("\n" + "!"*30)
    print(f"👀 ROUTER RECEIVED TYPE: {type(state)}")
    print(f"👀 ROUTER RECEIVED DATA: {state}")
    print("!"*30 + "\n")
    # --- X-RAY DEBUGGING END ---

    # If this prints <class 'str'>, we know the Auditor returned a string.
    # If this prints <class 'dict'>, then the error is somewhere else.
    
    if isinstance(state, str):
        print("🚨 CRITICAL ERROR: Router received a string. Defaulting to 'planner'.")
        return "planner"

    if state.get("is_verified"):
        print("\n✅ AUDIT SUCCESSFUL.")
        return "end"
    
    if state.get("iterations", 0) >= 3:
        print("\n🚨 CRITICAL: Max iterations reached.")
        return "end"
    
    print(f"\n❌ AUDIT REJECTED: {state.get('critique', 'No critique found')}")
    return "planner"

# 5. Connect the Router to the Graph
workflow.add_conditional_edges(
    "auditor", 
    router, 
    {
        "end": 'evaluator', 
        "planner": "planner"
    }
)
workflow.add_node("evaluator", evaluator_node)
#adding memeory layer
memory_context = SqliteSaver.from_conn_string("checkpoints.db")


# 7. Execution Block: This is where you write your questions
if __name__ == "__main__":
    load_dotenv()
    with memory_context as saver:
        app = workflow.compile(checkpointer=saver,interrupt_before=["researcher"])

        print("="*60)
        print("🚀 ENTERPRISE DUE DILIGENCE AGENT (VERSION 2026.1)")
        print("="*60)
        config = {"configurable": {"thread_id": "audit_001"}}
        # You can change this query to any of the test cases we discussed
        test_query = "Analyze Tesla's 2024-2025 revenue growth and their current market share in the global EV market."

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
            for output in app.stream(initial_input,config=config):
                for node_name, state_update in output.items():
                    print(f"📍 Finished Node: {node_name.upper()}")
                    
                    # Visual debug for your resume/portfolio
                    if node_name == "planner":
                        print(f"   Strategy: {state_update.get('plan')}")
                    elif node_name == "auditor" and not state_update.get("is_verified"):
                        print(f"   Red Flags: {state_update.get('critique')}")
            snapshot = app.get_state(config)
            if snapshot.next: # This checks if the graph is currently PAUSED
                print("\n" + "🛑" * 10 + " HUMAN REVIEW REQUIRED " + "🛑" * 10)
                print(f"The Planner has suggested {len(snapshot.values['plan'])} search queries.")
                
                choice = input("\nType 'GO' to proceed, or 'EXIT' to stop or 'EDIT' to add a query: ").strip().lower()
                if choice == "edit":
                    current_plan = snapshot.values.get("plan", [])
                    new_query = input("Enter your custom search query: ")
                    current_plan.append(new_query)
                    app.update_state(config, {"plan": current_plan})
                    for output in app.stream(None, config=config):
                        for node_name, state_update in output.items():
                            print(f"📍 Finished Node: {node_name.upper()}")
                    print(f"✅ Query Added! Current Plan: {current_plan}")
                elif choice == "go":
                    print("🚀 Resuming Research...")
                    # Passing None tells the graph to pick up from where it was interrupted
                    for output in app.stream(None, config=config):
                        for node_name, state_update in output.items():
                            print(f"📍 Finished Node: {node_name.upper()}")
                else:
                    print("Stopping execution. You can resume this thread later.")
            # Final Result Extraction
            # We run the app one last time to get the final state
            final_state = app.invoke(None,config=config)  # No new input, just get the final state
            
            print("\n" + "⭐" * 20 + " FINAL VERIFIED REPORT " + "⭐" * 20)
            print(final_state.get("draft_report", "Research failed to generate report."))
            print("="*60)

        except Exception as e:
            print(f"❌ System Crash: {e}")
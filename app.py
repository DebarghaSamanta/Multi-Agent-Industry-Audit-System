import os
import time
import uuid
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from shared import ResearchState
from brain import (
    evaluator_node,
    researcher_node,
    auditor_node,
    planner_node,
    reporter_node,
    persist_run,
    print_pass_rate_summary,
    log,
)

# ---------------------------------------------------------------------------
# 1. Build the graph
# ---------------------------------------------------------------------------
workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("auditor", auditor_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("reporter", reporter_node)

workflow.add_edge("evaluator", "reporter")
workflow.add_edge("reporter", END)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "auditor")


# ---------------------------------------------------------------------------
# 2. Router
# ---------------------------------------------------------------------------
def router(state: ResearchState) -> str:
    if state.get("is_verified"):
        log("router_decision", decision="verified", iterations=state.get("iterations"))
        print("\n  AUDIT PASSED — sending to evaluator")
        return "end"

    if state.get("iterations", 0) >= 3:
        log("router_decision", decision="max_iterations_reached")
        print("\n  Max iterations reached — forcing evaluation with current data")
        return "end"

    log("router_decision", decision="retry", critique=state.get("critique", "")[:80])
    print(f"\n  Audit rejected — retrying. Critique: {state.get('critique', '')}")
    return "planner"


workflow.add_conditional_edges(
    "auditor",
    router,
    {"end": "evaluator", "planner": "planner"},
)

# ---------------------------------------------------------------------------
# 3. Memory
# ---------------------------------------------------------------------------
memory_context = SqliteSaver.from_conn_string("checkpoints.db")


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def _drain(app, config):
    """Stream all remaining nodes, printing each one as it finishes."""
    for output in app.stream(None, config=config):
        for node_name, state_update in output.items():
            if node_name == "__interrupt__":
                continue
            print(f"  [Node done] {node_name.upper()}")
            if node_name == "planner":
                print(f"    Queries: {state_update.get('plan')}")
            elif node_name == "auditor" and not state_update.get("is_verified"):
                print(f"    Red flag: {state_update.get('critique')}")


def _handle_interrupt(app, config) -> bool:
    """
    Show the current plan, ask the user what to do.
    Returns True if execution should continue, False if user chose EXIT.
    On every EDIT or GO the graph resumes and may hit another interrupt
    (retry loop) — we keep looping until no more interrupts remain.
    """
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            return True  # graph finished cleanly, nothing to handle

        # Only show the HUMAN REVIEW banner when paused before researcher
        if "researcher" not in snapshot.next:
            # Paused somewhere else (shouldn't normally happen) — just resume
            _drain(app, config)
            continue

        plan = snapshot.values.get("plan", [])
        print("\n" + " HUMAN REVIEW REQUIRED ".center(55, "-"))
        print(f"  Planner proposed {len(plan)} search queries:")
        for i, q in enumerate(plan, 1):
            print(f"    {i}. {q}")

        choice = input(
            "\n  Type GO to proceed, EDIT to add a query, or EXIT to stop: "
        ).strip().lower()

        if choice == "exit":
            print("  Execution stopped. Run again to resume this thread.")
            return False

        if choice == "edit":
            new_query = input("  Enter your custom query: ").strip()
            plan.append(new_query)
            app.update_state(config, {"plan": plan})
            print(f"  Query added. New plan has {len(plan)} queries.")

        # GO or after EDIT — resume and drain until next interrupt or completion
        print("  Resuming research...")
        for output in app.stream(None, config=config):
            for node_name, state_update in output.items():
                if node_name == "__interrupt__":
                    break  # hit another interrupt, fall back to top of while loop
                print(f"  [Node done] {node_name.upper()}")
                if node_name == "planner":
                    print(f"    Queries: {state_update.get('plan')}")
                elif node_name == "auditor" and not state_update.get("is_verified"):
                    print(f"    Red flag: {state_update.get('critique')}")
            else:
                continue
            break  # inner break propagated — go back to top of while to re-check

        # After draining, loop back and check if there is another interrupt


# ---------------------------------------------------------------------------
# 5. Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()

    with memory_context as saver:
        app = workflow.compile(
            checkpointer=saver,
            interrupt_before=["researcher"],
        )

        print("=" * 60)
        print("  ENTERPRISE DUE DILIGENCE AGENT  v2026.2")
        print("=" * 60)

        # Fresh thread ID every run — avoids replaying old checkpoints
        thread_id = f"audit_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        log("run_start", thread_id=thread_id)

        test_query = (
            "Analyze Tesla's 2024-2025 revenue growth and their "
            "current market share in the global EV market."
        )

        initial_input = {
            "topic": test_query,
            "iterations": 0,
            "is_verified": False,
            "plan": [],
            "raw_data": [],
            "faithfulness_score": 0.0,
            "relevancy_score": 0.0,
        }

        print(f"  Target: {test_query}\n")
        run_start = time.perf_counter()

        try:
            # Phase 1 — run until first interrupt (always before researcher)
            for output in app.stream(initial_input, config=config):
                for node_name, state_update in output.items():
                    if node_name == "__interrupt__":
                        continue
                    print(f"  [Node done] {node_name.upper()}")
                    if node_name == "planner":
                        print(f"    Queries planned: {state_update.get('plan')}")

            # Phase 2 — handle ALL interrupts (retry loop may re-interrupt)
            should_continue = _handle_interrupt(app, config)
            if not should_continue:
                raise SystemExit(0)

            # Phase 3 — graph is done, read final state once
            final_state = app.get_state(config).values
            run_latency_ms = int((time.perf_counter() - run_start) * 1000)

            persist_run(final_state, run_latency_ms)

            print("\n" + " FINAL VERIFIED REPORT ".center(55, "="))
            print(final_state.get("draft_report", "No report generated."))
            print("\n" + " VERDICT ".center(55, "="))
            print(f"  Investment grade : {final_state.get('investment_grade', 'N/A')}")
            print(f"  Confidence       : {final_state.get('confidence_pct', 0)}%")
            print(f"  Faithfulness     : {final_state.get('faithfulness_score', 0.0):.2f}")
            print(f"  Data gaps        : {final_state.get('data_gap_count', 0)}")
            print(f"  Risk flags       : {final_state.get('risk_flag_count', 0)}")
            print(f"  Sources cited    : {final_state.get('sources_cited', 0)}")
            print(f"  Total latency    : {run_latency_ms}ms")
            print("=" * 55)

            print_pass_rate_summary()

        except SystemExit:
            pass
        except Exception as exc:
            log("system_crash", error=str(exc))
            print(f"\n  System error: {exc}")
            raise
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
    refusal_node,
    grounding_node,
    persist_run,
    print_pass_rate_summary,
    log,
)

# ---------------------------------------------------------------------------
# 1. Build the graph
# ---------------------------------------------------------------------------
workflow = StateGraph(ResearchState)

workflow.add_node("planner",    planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("auditor",    auditor_node)
workflow.add_node("evaluator",  evaluator_node)
workflow.add_node("refusal",    refusal_node)    # NEW: data sufficiency gate
workflow.add_node("grounding",  grounding_node)  # NEW: claim-evidence verification
workflow.add_node("reporter",   reporter_node)

# Fixed edges
workflow.set_entry_point("planner")
workflow.add_edge("planner",   "researcher")
workflow.add_edge("researcher","auditor")
workflow.add_edge("evaluator", "refusal")       # evaluator → refusal gate

# After refusal gate: if refused → reporter (short-circuits); else → grounding
workflow.add_conditional_edges(
    "refusal",
    lambda s: "reporter" if s.get("should_refuse") else "grounding",
    {"reporter": "reporter", "grounding": "grounding"},
)

workflow.add_edge("grounding", "reporter")
workflow.add_edge("reporter",  END)


# ---------------------------------------------------------------------------
# 2. Audit router: retry or proceed to evaluation
# ---------------------------------------------------------------------------
def router(state: ResearchState) -> str:
    retrieval_failed = state.get("retrieval_failed", False)
    iterations       = state.get("iterations", 0)

    # If retrieval failed and we still have retries, go back to planner
    if retrieval_failed and iterations < 3:
        log("router_decision", decision="retry_retrieval_failed",
            iterations=iterations)
        print(f"\n  Retrieval failed — replanning (iteration {iterations}/3)")
        return "planner"

    if state.get("is_verified"):
        log("router_decision", decision="verified", iterations=iterations)
        print("\n  AUDIT PASSED — sending to evaluator")
        return "end"

    if iterations >= 3:
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
# 4. Human-in-the-loop helpers
# ---------------------------------------------------------------------------
def _drain(app, config):
    for output in app.stream(None, config=config):
        for node_name, state_update in output.items():
            if node_name == "__interrupt__":
                continue
            print(f"  [Node done] {node_name.upper()}")
            if node_name == "planner":
                print(f"    Queries: {state_update.get('plan')}")
            elif node_name == "auditor" and not state_update.get("is_verified"):
                print(f"    Red flag: {state_update.get('critique')}")
            elif node_name == "refusal" and state_update.get("should_refuse"):
                print(f"    REFUSED: {state_update.get('refusal_reason')}")
            elif node_name == "grounding":
                passed = state_update.get("grounding_passed")
                unv    = len(state_update.get("unverified_claims", []))
                print(f"    Grounding: {'PASSED' if passed else 'FAILED'} | {unv} unverified claims")


def _handle_interrupt(app, config) -> bool:
    while True:
        snapshot = app.get_state(config)
        if not snapshot.next:
            return True

        if "researcher" not in snapshot.next:
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

        print("  Resuming research...")
        for output in app.stream(None, config=config):
            for node_name, state_update in output.items():
                if node_name == "__interrupt__":
                    break
                print(f"  [Node done] {node_name.upper()}")
                if node_name == "planner":
                    print(f"    Queries: {state_update.get('plan')}")
                elif node_name == "auditor" and not state_update.get("is_verified"):
                    print(f"    Red flag: {state_update.get('critique')}")
                elif node_name == "refusal" and state_update.get("should_refuse"):
                    print(f"    REFUSED: {state_update.get('refusal_reason')}")
            else:
                continue
            break


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
        print("  ENTERPRISE DUE DILIGENCE AGENT  v2026.3")
        print("  + Retrieval Fallback  + Query Rewriting  + Completeness")
        print("  + Refusal Gate        + Grounding Node")
        print("=" * 60)

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
            "completeness_score": 0.0,
            "hallucination_risk": "HIGH",
            "coverage_summary": "",
            "retrieval_failed": False,
            "failed_queries": [],
            "grounded_report": "",
            "unverified_claims": [],
            "grounding_passed": False,
            "should_refuse": False,
            "refusal_reason": "",
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

            # Phase 3 — read final state
            final_state   = app.get_state(config).values
            run_latency_ms = int((time.perf_counter() - run_start) * 1000)

            persist_run(final_state, run_latency_ms)

            # ---- Print final output ----
            if final_state.get("should_refuse"):
                print("\n" + " REPORT REFUSED ".center(60, "="))
                print(f"  {final_state.get('refusal_reason')}")
                print("=" * 60)
            else:
                print("\n" + " FINAL GROUNDED REPORT ".center(60, "="))
                print(final_state.get("grounded_report") or final_state.get("draft_report", "No report generated."))

                unverified = final_state.get("unverified_claims", [])
                if unverified:
                    print("\n" + " UNVERIFIED CLAIMS ".center(60, "-"))
                    for claim in unverified:
                        print(f"  • {claim}")

            print("\n" + " VERDICT ".center(60, "="))
            print(f"  Investment grade   : {final_state.get('investment_grade', 'N/A')}")
            print(f"  Confidence         : {final_state.get('confidence_pct', 0)}%")
            print(f"  Faithfulness       : {final_state.get('faithfulness_score', 0.0):.2f}")
            print(f"  Relevancy          : {final_state.get('relevancy_score', 0.0):.2f}")
            print(f"  Completeness       : {final_state.get('completeness_score', 0.0):.2f}")
            print(f"  Hallucination risk : {final_state.get('hallucination_risk', 'N/A')}")
            print(f"  Grounding passed   : {final_state.get('grounding_passed', False)}")
            print(f"  Data gaps          : {final_state.get('data_gap_count', 0)}")
            print(f"  Risk flags         : {final_state.get('risk_flag_count', 0)}")
            print(f"  Sources cited      : {final_state.get('sources_cited', 0)}")
            print(f"  Total latency      : {run_latency_ms}ms")
            print(f"  Coverage           : {final_state.get('coverage_summary', 'N/A')}")
            print("=" * 60)

            print_pass_rate_summary()

        except SystemExit:
            pass
        except Exception as exc:
            log("system_crash", error=str(exc))
            print(f"\n  System error: {exc}")
            raise
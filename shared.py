from typing import TypedDict, Annotated, List, Optional
import operator


class ResearchState(TypedDict):
    topic: str

    # operator.add lets agents append without overwriting previous findings
    raw_data: Annotated[List[str], operator.add]

    plan: List[str]
    draft_report: str
    critique: str
    is_verified: bool
    iterations: int

    # --- Evaluation fields (populated by evaluator_node) ---
    eval_score: str           # human-readable RAGAS summary string
    faithfulness_score: float # 0.0 - 1.0
    relevancy_score: float    # 0.0 - 1.0

    # --- Report fields (populated by reporter_node) ---
    investment_grade: str     # A / B+ / B / C / D
    confidence_pct: int       # 0 - 100
    data_gap_count: int
    sources_cited: int
    risk_flag_count: int
    run_latency_ms: int       # total wall-clock ms for the full run
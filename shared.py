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

    # --- Retrieval tracking (for fallback logic) ---
    retrieval_failed: bool          # True if all searches returned empty
    failed_queries: List[str]       # queries that returned no results

    # --- Evaluation fields (populated by evaluator_node) ---
    eval_score: str                 # human-readable summary string
    faithfulness_score: float       # 0.0 - 1.0 (claims backed by sources)
    relevancy_score: float          # 0.0 - 1.0 (answer relevance to topic)
    completeness_score: float       # 0.0 - 1.0 (how complete the answer is)
    hallucination_risk: str         # LOW / MEDIUM / HIGH
    coverage_summary: str           # what was covered vs. what is missing

    # --- Grounding fields (populated by grounding_node) ---
    grounded_report: str            # report with [UNVERIFIED] claims flagged
    unverified_claims: List[str]    # claims that had no source backing
    grounding_passed: bool          # True if enough claims are grounded

    # --- Refusal fields (populated by refusal_node) ---
    should_refuse: bool             # True if data is too thin to generate a report
    refusal_reason: str             # why we refused

    # --- Report fields (populated by reporter_node) ---
    investment_grade: str           # A / B+ / B / C / D
    confidence_pct: int             # 0 - 100
    data_gap_count: int
    sources_cited: int
    risk_flag_count: int
    run_latency_ms: int             # total wall-clock ms for the full run
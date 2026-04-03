import os
import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from httpx import ConnectError, TimeoutException

def _is_network_error(exc):
    return isinstance(exc, (ConnectError, TimeoutException, ConnectionError))

def _is_rate_limit(exc):
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg

from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

from shared import ResearchState

load_dotenv()

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("audit_runs.log"),
    ],
)

def log(event: str, **kwargs):
    record = {"ts": time.time(), "event": event, **kwargs}
    logging.info(json.dumps(record))


# ---------------------------------------------------------------------------
# SQLite eval + latency log
# ---------------------------------------------------------------------------
DB_PATH = "checkpoints.db"

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                REAL,
            topic             TEXT,
            iterations        INTEGER,
            is_verified       INTEGER,
            faithfulness      REAL,
            relevancy         REAL,
            completeness      REAL,
            hallucination_risk TEXT,
            grounding_passed  INTEGER,
            should_refuse     INTEGER,
            unverified_claims INTEGER,
            grade             TEXT,
            confidence        INTEGER,
            data_gaps         INTEGER,
            latency_ms        INTEGER,
            critique          TEXT
        )
    """)
    con.commit()
    con.close()

_init_db()


def persist_run(state: ResearchState, latency_ms: int):
    con = sqlite3.connect(DB_PATH)
    unverified_count = len(state.get("unverified_claims", []))
    con.execute(
        """INSERT INTO runs
           (ts, topic, iterations, is_verified, faithfulness, relevancy,
            completeness, hallucination_risk, grounding_passed, should_refuse,
            unverified_claims, grade, confidence, data_gaps, latency_ms, critique)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            time.time(),
            state.get("topic", ""),
            state.get("iterations", 0),
            int(state.get("is_verified", False)),
            state.get("faithfulness_score", 0.0),
            state.get("relevancy_score", 0.0),
            state.get("completeness_score", 0.0),
            state.get("hallucination_risk", "UNKNOWN"),
            int(state.get("grounding_passed", False)),
            int(state.get("should_refuse", False)),
            unverified_count,
            state.get("investment_grade", "N/A"),
            state.get("confidence_pct", 0),
            state.get("data_gap_count", 0),
            latency_ms,
            state.get("critique", ""),
        ),
    )
    con.commit()
    con.close()


def print_pass_rate_summary():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT
            ROUND(100.0 * SUM(iterations=1) / COUNT(*), 1)          AS first_pass_pct,
            ROUND(AVG(iterations), 2)                                AS avg_iterations,
            ROUND(100.0 * SUM(iterations>=3) / COUNT(*), 1)         AS max_iter_failure_pct,
            ROUND(AVG(faithfulness), 2)                              AS avg_faithfulness,
            ROUND(AVG(completeness), 2)                              AS avg_completeness,
            ROUND(100.0 * SUM(should_refuse=1) / COUNT(*), 1)       AS refusal_rate_pct,
            ROUND(100.0 * SUM(grounding_passed=1) / COUNT(*), 1)    AS grounding_pass_pct,
            ROUND(AVG(latency_ms) / 1000.0, 1)                      AS avg_latency_s,
            COUNT(*)                                                  AS total_runs
        FROM (SELECT * FROM runs ORDER BY id DESC LIMIT 20)
    """).fetchone()
    con.close()
    if rows and rows[8]:
        print("\n" + "=" * 60)
        print("  AUDIT SYSTEM — LAST 20 RUNS DASHBOARD")
        print("=" * 60)
        print(f"  Total runs logged    : {rows[8]}")
        print(f"  First-pass rate      : {rows[0]}%")
        print(f"  Avg iterations/run   : {rows[1]}")
        print(f"  Max-iter failure     : {rows[2]}%")
        print(f"  Avg faithfulness     : {rows[3]}")
        print(f"  Avg completeness     : {rows[4]}")
        print(f"  Refusal rate         : {rows[5]}%")
        print(f"  Grounding pass rate  : {rows[6]}%")
        print(f"  Avg latency          : {rows[7]}s")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
api_key_groq = os.getenv("GROQ_API_KEY")
if not api_key_groq:
    raise ValueError("GROQ_API_KEY not found — check your .env file.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key_groq,
    temperature=0,
)

AUDIT_MODEL = os.getenv("AUDIT_MODEL", "llama-3.3-70b-versatile")
auditor_llm = ChatGroq(
    model=AUDIT_MODEL,
    api_key=api_key_groq,
    temperature=0,
)

search_tool = TavilySearch(max_results=5, topic="general")

def _tavily_search(query: str) -> dict:
    """Call Tavily with retry. Returns empty results on total failure."""
    for attempt in range(3):
        try:
            result = search_tool.invoke({"query": query})
            return result
        except Exception as exc:
            if _is_rate_limit(exc):
                wait = 30
                print(f"  Tavily rate limit — waiting {wait}s...")
                time.sleep(wait)
            elif attempt < 2:
                time.sleep(3)
            else:
                # FIX 1: Retrieval fallback — log and return empty instead of crashing
                log("search_exhausted", query=query, error=str(exc))
                return {"results": [], "error": str(exc)}
    return {"results": []}


# ---------------------------------------------------------------------------
# PLANNER NODE — rewrites queries differently on each retry
# ---------------------------------------------------------------------------

def planner_node(state: ResearchState):
    iteration = state.get("iterations", 0)
    log("node_start", node="planner", iteration=iteration)
    print(f"\n--- AGENT: PLANNER (Iteration {iteration}) ---")

    topic = state["topic"]
    critique = state.get("critique", "")
    failed_queries = state.get("failed_queries", [])

    # FIX 2: Query rewriting — each iteration gets a DIFFERENT strategy
    # so we never send the same queries twice and avoid forced hallucination
    if iteration == 0:
        # First attempt: standard VC checklist
        strategy_hint = """
        ### STRATEGY (Iteration 1 — Broad Discovery)
        - Query 1: Broad financial search: revenue, growth, margins for the company.
        - Query 2: Competitive landscape: market share, top competitors, positioning.
        - Query 3: Risk profile: lawsuits, regulatory scrutiny, SEC filings.
        """
    elif iteration == 1:
        # Second attempt: go deeper with alternative angles
        strategy_hint = f"""
        ### STRATEGY (Iteration 2 — Deep Dive, DIFFERENT angles from last time)
        Previous critique: {critique}
        Failed or thin queries: {failed_queries}

        You MUST generate queries that are DIFFERENT from any prior attempt.
        - Query 1: Dig into primary sources — annual reports, 10-K filings, earnings calls.
        - Query 2: Find analyst reports, industry rankings, or third-party market research.
        - Query 3: Look for news from the past 6 months: partnerships, product launches, M&A.
        """
    else:
        # Third attempt: pivot to alternative sources entirely
        strategy_hint = f"""
        ### STRATEGY (Iteration 3 — Source Pivot, COMPLETELY different from all prior queries)
        Previous critique: {critique}
        All previously tried angles: {failed_queries}

        Pivot to entirely new sources:
        - Query 1: Search government databases, patents, or regulatory filings.
        - Query 2: Search for investor presentations, conference transcripts, or CEO interviews.
        - Query 3: Search for industry association data or academic/think-tank reports.
        """

    prompt = f"""
    ### ROLE
    Lead Investment Strategist — Query Architect.

    ### TASK
    Generate 3 distinct, specific search queries to gather investment data on: "{topic}".

    {strategy_hint}

    ### RULES
    - Every query MUST include the company or entity name from the topic. No placeholders.
    - Each query must target a DIFFERENT data dimension.
    - Queries must be concrete enough to return structured facts (numbers, dates, names).
    - DO NOT reuse any query from: {failed_queries}

    ### OUTPUT
    Return ONLY 3 search queries, one per line. No numbering, no bullets, no preamble.
    """

    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()][:3]
    log("node_done", node="planner", iteration=iteration, queries=queries)
    return {"plan": queries, "iterations": iteration + 1}


# ---------------------------------------------------------------------------
# RESEARCHER NODE — parallel search with retrieval fallback tracking
# ---------------------------------------------------------------------------

def researcher_node(state: ResearchState):
    log("node_start", node="researcher", num_queries=len(state["plan"]))
    print(f"\n--- AGENT: RESEARCHER — {len(state['plan'])} queries (parallel) ---")

    t0 = time.perf_counter()
    failed_queries = list(state.get("failed_queries", []))

    def _search_one(query: str):
        print(f"  Searching: {query}")
        try:
            response = _tavily_search(query)
            if not isinstance(response, dict):
                log("search_warn", query=query, got=type(response).__name__)
                return [], query  # FIX 1: track failed query

            if "error" in response and not response.get("results"):
                log("search_warn", query=query, error=str(response.get("error", "")))
                return [], query  # FIX 1: track failed query

            chunks = []
            for r in response.get("results", []):
                url     = r.get("url", "unknown")
                content = r.get("content", "")
                if content.strip():
                    chunks.append(f"[SOURCE: {url}]\n{content}")

            # FIX 1: If search returned results but all were empty content
            if not chunks:
                return [], query

            return chunks, None  # None = not failed
        except Exception as exc:
            log("search_error", query=query, error=str(exc))
            print(f"  Search failed: {query} — {exc}")
            return [], query  # FIX 1: track failed query

    all_context = []
    with ThreadPoolExecutor(max_workers=len(state["plan"])) as pool:
        futures = {pool.submit(_search_one, q): q for q in state["plan"]}
        for future in as_completed(futures):
            chunks, failed_q = future.result()
            all_context.extend(chunks)
            if failed_q:
                failed_queries.append(failed_q)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # FIX 1: Determine if retrieval totally failed
    retrieval_failed = len(all_context) == 0
    log("search_done", parallel_ms=elapsed_ms, chunks_collected=len(all_context),
        retrieval_failed=retrieval_failed, failed_query_count=len(failed_queries))
    print(f"  Parallel search: {elapsed_ms}ms | {len(all_context)} chunks | "
          f"{'RETRIEVAL FAILED' if retrieval_failed else 'OK'}")

    combined_context = "\n\n".join(all_context)
    sources_cited = sum(1 for c in all_context if c.startswith("[SOURCE:"))
    context_for_llm = combined_context[:4000]

    if retrieval_failed:
        # Don't invoke LLM at all — nothing to extract from
        return {
            "raw_data": [],
            "draft_report": "RETRIEVAL_FAILED: No search results were returned for any query.",
            "sources_cited": 0,
            "retrieval_failed": True,
            "failed_queries": failed_queries,
        }

    extraction_prompt = f"""
    ### ROLE
    Senior Data Extraction Engine (VC Domain).

    ### TASK
    Extract metrics from these search results regarding '{state['topic']}':
    {context_for_llm}

    ### EXTRACTION RULES
    1. Extract only facts explicitly stated in the source text. Do not infer or extrapolate.
    2. Fill the standard VC checklist where data is available: Traction, Moat, Legal risks.
    3. For every metric NOT found in the sources, write exactly: DATA_GAP: [Metric Name]
    4. Every factual claim must end with its source URL in parentheses: (source: <url>)
    5. List every URL you drew a fact from at the end.

    ### OUTPUT FORMAT
    Return a structured bullet-point list. End with a line: SOURCES_USED: N
    """

    response = llm.invoke(extraction_prompt)
    log("node_done", node="researcher", sources_cited=sources_cited)

    return {
        "raw_data": [combined_context],
        "draft_report": response.content,
        "sources_cited": sources_cited,
        "retrieval_failed": False,
        "failed_queries": failed_queries,
    }


# ---------------------------------------------------------------------------
# AUDITOR NODE
# ---------------------------------------------------------------------------

def auditor_node(state: ResearchState):
    log("node_start", node="auditor", model=AUDIT_MODEL)
    print(f"\n--- AGENT: CHIEF AUDITOR (model: {AUDIT_MODEL}) ---")

    draft = state.get("draft_report", "")
    sources_cited = state.get("sources_cited", 0)
    retrieval_failed = state.get("retrieval_failed", False)

    # FIX 1: If retrieval completely failed, reject immediately
    if retrieval_failed:
        log("auditor_skip", reason="retrieval_failed")
        print("  Retrieval failed — auto-rejecting for replanning")
        return {
            "is_verified": False,
            "critique": "All search queries returned empty results. Retrieval completely failed — need different queries.",
        }

    audit_prompt = f"""
    ### ROLE
    Chief Investment Compliance Officer.

    ### TASK
    Review this research report and decide if it is ready to publish.

    Research Report:
    {draft[:1500]}

    Sources cited count: {sources_cited}

    ### VERIFICATION RULES
    1. Mark is_verified=true if the report has at least 3 concrete numbered facts (revenue, market share, deliveries, etc.).
    2. DATA_GAP entries are ACCEPTABLE and expected — do not count them against the report.
    3. Mark is_verified=false ONLY if: fewer than 3 real facts found, OR facts are clearly fabricated with no numbers.
    4. A report with real numbers like "$97B revenue" or "1.64M deliveries" MUST be marked is_verified=true.

    ### OUTPUT — return ONLY valid JSON, no markdown fences:
    {{
        "is_verified": true or false,
        "critique": "one sentence only if false, else empty string",
        "missing_metrics": ["only truly critical missing items, max 2"]
    }}
    """

    response = auditor_llm.invoke(audit_prompt)

    try:
        clean = response.content.replace("```json", "").replace("```", "").strip()
        audit_results = json.loads(clean)
    except json.JSONDecodeError as exc:
        log("auditor_parse_error", raw=response.content[:200], error=str(exc))
        audit_results = {
            "is_verified": sources_cited > 0,
            "critique": "" if sources_cited > 0 else "No sources found.",
            "missing_metrics": [],
        }

    log("node_done", node="auditor", is_verified=audit_results.get("is_verified"))
    return {
        "is_verified": audit_results.get("is_verified", False),
        "critique": audit_results.get("critique", ""),
    }


# ---------------------------------------------------------------------------
# EVALUATOR NODE — faithfulness + relevancy + completeness
# ---------------------------------------------------------------------------

def evaluator_node(state: ResearchState):
    """
    Scores three dimensions:
      - faithfulness:  what fraction of claims are backed by retrieved sources
      - relevancy:     how well the report answers the original research question
      - completeness:  how much of the expected answer surface was actually covered
    Also produces hallucination_risk (LOW/MEDIUM/HIGH) and a coverage_summary.
    """
    log("node_start", node="evaluator")
    print("\n--- AGENT: EVALUATOR — scoring faithfulness / relevancy / completeness ---")

    draft   = state.get("draft_report", "")[:1500]
    sources = " ".join(state.get("raw_data", []))[:2500]
    topic   = state.get("topic", "")

    if not sources.strip():
        log("evaluator_skip", reason="no source data")
        print("  No source data — skipping evaluation")
        return {
            "eval_score": "No sources — all metrics zero",
            "faithfulness_score": 0.0,
            "relevancy_score": 0.0,
            "completeness_score": 0.0,
            "hallucination_risk": "HIGH",
            "coverage_summary": "No sources were retrieved; no coverage possible.",
        }

    scoring_prompt = f"""
    ### ROLE
    Objective Research Quality Auditor. You score three independent dimensions.

    ### INPUTS
    Research Topic / Question:
    {topic}

    Source Data (truncated):
    {sources}

    Research Report:
    {draft}

    ### SCORING RULES

    **faithfulness_score** (0.0 - 1.0):
    - Count every factual claim in the report (ignore DATA_GAP lines).
    - Check each claim against the source data.
    - Score = supported_claims / total_claims
    - If total_claims = 0, score = 0.0

    **relevancy_score** (0.0 - 1.0):
    - Does the report actually answer the specific research topic/question?
    - 1.0 = directly and fully answers the topic
    - 0.5 = partially answers the topic
    - 0.0 = completely off-topic or empty

    **completeness_score** (0.0 - 1.0):
    - For a full investment audit, expect: financials, market position, risk factors, growth outlook.
    - What fraction of those expected dimensions are actually covered with real data (not DATA_GAP)?
    - 1.0 = all four dimensions covered with real numbers
    - 0.75 = three covered
    - 0.5 = two covered
    - 0.25 = one covered
    - 0.0 = none covered

    **hallucination_risk** (string):
    - "LOW"    if faithfulness >= 0.8
    - "MEDIUM" if faithfulness >= 0.5
    - "HIGH"   if faithfulness < 0.5

    **coverage_summary** (one sentence):
    - Describe what topics were covered and what key areas are still missing.

    ### OUTPUT — return ONLY valid JSON, no markdown fences:
    {{
        "total_claims": <integer>,
        "supported_claims": <integer>,
        "faithfulness_score": <float 0.0-1.0>,
        "relevancy_score": <float 0.0-1.0>,
        "completeness_score": <float 0.0-1.0>,
        "hallucination_risk": "LOW" or "MEDIUM" or "HIGH",
        "coverage_summary": "<one sentence>",
        "reasoning": "<one sentence on faithfulness>"
    }}
    """

    try:
        response = llm.invoke(scoring_prompt)
        clean = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        faith      = round(float(result.get("faithfulness_score", 0.0)), 2)
        relevancy  = round(float(result.get("relevancy_score", 0.0)), 2)
        completeness = round(float(result.get("completeness_score", 0.0)), 2)
        hal_risk   = result.get("hallucination_risk", "HIGH")
        coverage   = result.get("coverage_summary", "")
        total      = result.get("total_claims", "?")
        supported  = result.get("supported_claims", "?")

        summary = (f"Faithfulness: {faith:.2f} ({supported}/{total} claims) | "
                   f"Relevancy: {relevancy:.2f} | Completeness: {completeness:.2f} | "
                   f"Hallucination risk: {hal_risk}")

        log("eval_scores", faithfulness=faith, relevancy=relevancy,
            completeness=completeness, hallucination_risk=hal_risk)
        print(f"  Scores — {summary}")
        print(f"  Coverage: {coverage}")

        return {
            "eval_score": summary,
            "faithfulness_score": faith,
            "relevancy_score": relevancy,
            "completeness_score": completeness,
            "hallucination_risk": hal_risk,
            "coverage_summary": coverage,
        }
    except (json.JSONDecodeError, Exception) as exc:
        log("eval_error", error=str(exc))
        print(f"  Scoring failed: {exc}")
        return {
            "eval_score": "Evaluation failed",
            "faithfulness_score": 0.0,
            "relevancy_score": 0.0,
            "completeness_score": 0.0,
            "hallucination_risk": "HIGH",
            "coverage_summary": "Evaluation could not be completed.",
        }


# ---------------------------------------------------------------------------
# REFUSAL NODE — gates answer generation on data sufficiency
# ---------------------------------------------------------------------------

def refusal_node(state: ResearchState):
    """
    Decides whether we have enough data to generate a trustworthy report.
    If not, sets should_refuse=True so the reporter emits a refusal instead
    of a hallucinated answer.

    Refusal triggers (ANY of these is sufficient):
    - Retrieval completely failed (no chunks at all)
    - faithfulness_score < 0.40 (majority of claims are unsupported)
    - completeness_score < 0.25 (less than one dimension of expected data)
    - sources_cited == 0
    - hallucination_risk == HIGH AND completeness < 0.30
    """
    log("node_start", node="refusal")
    print("\n--- AGENT: REFUSAL GATE — checking data sufficiency ---")

    retrieval_failed  = state.get("retrieval_failed", False)
    faithfulness      = state.get("faithfulness_score", 0.0)
    completeness      = state.get("completeness_score", 0.0)
    sources_cited     = state.get("sources_cited", 0)
    hal_risk          = state.get("hallucination_risk", "HIGH")
    draft             = state.get("draft_report", "")

    reasons = []

    if retrieval_failed or "RETRIEVAL_FAILED" in draft:
        reasons.append("all search queries returned empty results")

    if sources_cited == 0:
        reasons.append("zero sources were cited")

    if faithfulness < 0.40 and sources_cited > 0:
        reasons.append(f"faithfulness score too low ({faithfulness:.2f} < 0.40 threshold)")

    if completeness < 0.25:
        reasons.append(f"completeness score too low ({completeness:.2f} < 0.25 threshold — less than one expected dimension covered)")

    if hal_risk == "HIGH" and completeness < 0.30:
        reasons.append("hallucination risk is HIGH with insufficient completeness")

    if reasons:
        refusal_reason = "Refusing to generate report: " + "; ".join(reasons) + "."
        log("refusal_triggered", reasons=reasons)
        print(f"  REFUSING — {refusal_reason}")
        return {
            "should_refuse": True,
            "refusal_reason": refusal_reason,
        }

    log("refusal_passed")
    print("  Data sufficiency check PASSED — proceeding to grounding")
    return {
        "should_refuse": False,
        "refusal_reason": "",
    }


# ---------------------------------------------------------------------------
# GROUNDING NODE — verifies every claim in the draft against sources
# ---------------------------------------------------------------------------

def grounding_node(state: ResearchState):
    """
    Reads the draft report, extracts each factual claim, checks it against
    retrieved source text, and produces a grounded version of the report
    where unverified claims are flagged as [UNVERIFIED].

    Also outputs:
    - unverified_claims: list of claims that could not be grounded
    - grounding_passed: True if >= 70% of claims are grounded
    """
    log("node_start", node="grounding")
    print("\n--- AGENT: GROUNDING — verifying claims against sources ---")

    draft   = state.get("draft_report", "")[:2000]
    sources = " ".join(state.get("raw_data", []))[:3000]

    if not sources.strip() or not draft.strip():
        log("grounding_skip", reason="no sources or draft")
        return {
            "grounded_report": draft,
            "unverified_claims": ["No source data available — all claims unverified"],
            "grounding_passed": False,
        }

    grounding_prompt = f"""
    ### ROLE
    Evidence Grounding Specialist. Your job is fact-anchoring — ensuring every claim
    in a research report can be traced back to a specific piece of retrieved source text.

    ### SOURCE DATA
    {sources}

    ### DRAFT REPORT TO GROUND
    {draft}

    ### TASK
    1. Parse the draft report into individual factual claims (skip DATA_GAP lines).
    2. For each claim, search the source data for supporting evidence.
    3. If a claim IS supported: keep it exactly as-is in the grounded report.
    4. If a claim IS NOT supported by any source text: append [UNVERIFIED] to that line
       in the grounded report and add it to the unverified_claims list.
    5. Compute grounding_rate = supported_claims / total_claims.
    6. Set grounding_passed = true if grounding_rate >= 0.70, else false.

    ### OUTPUT — return ONLY valid JSON, no markdown fences:
    {{
        "total_claims": <integer>,
        "supported_claims": <integer>,
        "grounding_rate": <float 0.0-1.0>,
        "grounding_passed": <boolean>,
        "unverified_claims": ["<claim text>", ...],
        "grounded_report": "<full report text with [UNVERIFIED] appended to unsupported lines>"
    }}
    """

    try:
        response = llm.invoke(grounding_prompt)
        clean = response.content.replace("```json", "").replace("```", "").strip()

        # Handle large JSON safely
        result = json.loads(clean)

        grounding_passed   = result.get("grounding_passed", False)
        unverified_claims  = result.get("unverified_claims", [])
        grounded_report    = result.get("grounded_report", draft)
        grounding_rate     = result.get("grounding_rate", 0.0)
        total              = result.get("total_claims", "?")
        supported          = result.get("supported_claims", "?")

        log("grounding_done", grounding_rate=grounding_rate,
            grounding_passed=grounding_passed, unverified_count=len(unverified_claims))
        print(f"  Grounding: {supported}/{total} claims verified "
              f"(rate: {grounding_rate:.2f}) — {'PASSED' if grounding_passed else 'FAILED'}")

        if unverified_claims:
            print(f"  Unverified claims ({len(unverified_claims)}): "
                  + "; ".join(unverified_claims[:2])
                  + ("..." if len(unverified_claims) > 2 else ""))

        return {
            "grounded_report": grounded_report,
            "unverified_claims": unverified_claims,
            "grounding_passed": grounding_passed,
        }

    except (json.JSONDecodeError, Exception) as exc:
        log("grounding_error", error=str(exc))
        print(f"  Grounding parse failed: {exc} — using raw draft with warning")
        return {
            "grounded_report": draft + "\n\n[WARNING: Grounding verification could not be completed]",
            "unverified_claims": ["Grounding process failed — treat all claims as unverified"],
            "grounding_passed": False,
        }


# ---------------------------------------------------------------------------
# REPORTER NODE
# ---------------------------------------------------------------------------

def reporter_node(state: ResearchState):
    log("node_start", node="reporter")
    print("\n--- AGENT: REPORTER — generating investment verdict ---")

    # If refused, short-circuit with a refusal report
    if state.get("should_refuse"):
        reason = state.get("refusal_reason", "Insufficient data.")
        print(f"\n{'='*60}")
        print(f"  REPORT REFUSED")
        print(f"  Reason: {reason}")
        print(f"{'='*60}\n")
        log("report_refused", reason=reason)
        return {
            "investment_grade": "REFUSED",
            "confidence_pct": 0,
            "data_gap_count": 0,
            "risk_flag_count": 0,
        }

    # Use the grounded report if available, else fall back to draft
    report_text = state.get("grounded_report") or state.get("draft_report", "")

    faith          = state.get("faithfulness_score", 0.0)
    relevancy      = state.get("relevancy_score", 0.0)
    completeness   = state.get("completeness_score", 0.0)
    hal_risk       = state.get("hallucination_risk", "HIGH")
    iterations     = state.get("iterations", 1)
    grounding_ok   = state.get("grounding_passed", False)
    unverified_ct  = len(state.get("unverified_claims", []))

    data_gap_count = report_text.upper().count("DATA_GAP:")
    risk_keywords  = ["lawsuit", "regulatory", "sec", "investigation", "fine", "risk", "recall"]
    risk_flag_count = sum(report_text.lower().count(kw) for kw in risk_keywords)

    # Confidence: weighted combination of quality signals
    raw_confidence    = (faith * 0.40 + relevancy * 0.30 + completeness * 0.30) * 100
    iteration_penalty = (iterations - 1) * 12
    gap_penalty       = data_gap_count * 8
    unverified_penalty = unverified_ct * 5
    grounding_bonus   = 5 if grounding_ok else -10

    confidence_pct = max(0, min(100, int(
        raw_confidence - iteration_penalty - gap_penalty
        - unverified_penalty + grounding_bonus
    )))

    # Investment grade
    if faith >= 0.85 and completeness >= 0.75 and confidence_pct >= 75:
        grade = "A"
    elif faith >= 0.75 and completeness >= 0.60 and confidence_pct >= 60:
        grade = "B+"
    elif faith >= 0.65 and completeness >= 0.50 and confidence_pct >= 50:
        grade = "B"
    elif faith >= 0.50 and completeness >= 0.25 and confidence_pct >= 35:
        grade = "C"
    else:
        grade = "D"

    sources_cited  = state.get("sources_cited", 0)
    coverage       = state.get("coverage_summary", "Not available")

    print(f"\n{'='*60}")
    print(f"  INVESTMENT GRADE  : {grade}")
    print(f"  CONFIDENCE        : {confidence_pct}%")
    print(f"  FAITHFULNESS      : {faith:.2f}  (claims backed by sources)")
    print(f"  RELEVANCY         : {relevancy:.2f}  (answer matches question)")
    print(f"  COMPLETENESS      : {completeness:.2f}  (expected dimensions covered)")
    print(f"  HALLUCINATION RISK: {hal_risk}")
    print(f"  GROUNDING         : {'PASSED' if grounding_ok else 'FAILED'} ({unverified_ct} unverified claims)")
    print(f"  DATA GAPS         : {data_gap_count}")
    print(f"  RISK FLAGS        : {risk_flag_count}")
    print(f"  SOURCES CITED     : {sources_cited}")
    print(f"  ITERATIONS USED   : {iterations} / 3")
    print(f"  COVERAGE          : {coverage}")
    print(f"{'='*60}\n")

    log("report_generated", grade=grade, confidence_pct=confidence_pct,
        data_gaps=data_gap_count, risk_flags=risk_flag_count,
        hallucination_risk=hal_risk, grounding_passed=grounding_ok)

    return {
        "investment_grade": grade,
        "confidence_pct": confidence_pct,
        "data_gap_count": data_gap_count,
        "risk_flag_count": risk_flag_count,
    }
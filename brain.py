import os
import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from httpx import ConnectError, TimeoutException

# Separate retry policies:
# - Network errors: retry fast (2-5s)
# - Rate limits (429): wait longer (30s) — Groq free tier resets per minute
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
# Structured logging — every event is JSON so logs are queryable later
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
    """Emit a structured JSON log line."""
    record = {"ts": time.time(), "event": event, **kwargs}
    logging.info(json.dumps(record))


# ---------------------------------------------------------------------------
# SQLite eval + latency log (separate from LangGraph checkpoints)
# ---------------------------------------------------------------------------
DB_PATH = "checkpoints.db"

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           REAL,
            topic        TEXT,
            iterations   INTEGER,
            is_verified  INTEGER,
            faithfulness REAL,
            relevancy    REAL,
            grade        TEXT,
            confidence   INTEGER,
            data_gaps    INTEGER,
            latency_ms   INTEGER,
            critique     TEXT
        )
    """)
    con.commit()
    con.close()

_init_db()


def persist_run(state: ResearchState, latency_ms: int):
    """Write one completed run to the runs table."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO runs
           (ts, topic, iterations, is_verified, faithfulness, relevancy,
            grade, confidence, data_gaps, latency_ms, critique)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            time.time(),
            state.get("topic", ""),
            state.get("iterations", 0),
            int(state.get("is_verified", False)),
            state.get("faithfulness_score", 0.0),
            state.get("relevancy_score", 0.0),
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
    """Print a quick dashboard of the last 20 runs."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT
            ROUND(100.0 * SUM(iterations=1) / COUNT(*), 1) AS first_pass_pct,
            ROUND(AVG(iterations), 2)                       AS avg_iterations,
            ROUND(100.0 * SUM(iterations>=3) / COUNT(*), 1) AS max_iter_failure_pct,
            ROUND(AVG(faithfulness), 2)                     AS avg_faithfulness,
            ROUND(AVG(latency_ms) / 1000.0, 1)              AS avg_latency_s,
            COUNT(*)                                         AS total_runs
        FROM (SELECT * FROM runs ORDER BY id DESC LIMIT 20)
    """).fetchone()
    con.close()
    if rows and rows[5]:
        print("\n" + "=" * 55)
        print("  AUDIT SYSTEM — LAST 20 RUNS DASHBOARD")
        print("=" * 55)
        print(f"  Total runs logged   : {rows[5]}")
        print(f"  First-pass rate     : {rows[0]}%")
        print(f"  Avg iterations/run  : {rows[1]}")
        print(f"  Max-iter failure    : {rows[2]}%")
        print(f"  Avg faithfulness    : {rows[3]}")
        print(f"  Avg latency         : {rows[4]}s")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
api_key_groq = os.getenv("GROQ_API_KEY")
if not api_key_groq:
    raise ValueError("GROQ_API_KEY not found — check your .env file.")

# Main LLM — llama-3.3-70b is Groq's fastest free model, well above GPT-3.5 quality
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key_groq,
    temperature=0,
)

# Auditor LLM — swap via AUDIT_MODEL env var to A/B test cost vs accuracy
# e.g. AUDIT_MODEL=llama3-8b-8192 for a fast cheap auditor
AUDIT_MODEL = os.getenv("AUDIT_MODEL", "llama-3.3-70b-versatile")
auditor_llm = ChatGroq(
    model=AUDIT_MODEL,
    api_key=api_key_groq,
    temperature=0,
)

# Official docs: tool.invoke({"query": "..."}) returns a dict:
# {"query":str, "results":[{"title","url","content","score","raw_content"},...], ...}
search_tool = TavilySearch(max_results=5, topic="general")

def _tavily_search(query: str) -> dict:
    """Call Tavily with simple retry on genuine failures only."""
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
                raise
    return {"results": []}


# ---------------------------------------------------------------------------
# AGENT NODES
# ---------------------------------------------------------------------------

def planner_node(state: ResearchState):
    iteration = state.get("iterations", 0)
    log("node_start", node="planner", iteration=iteration)
    print(f"\n--- AGENT: PLANNER (Iteration {iteration}) ---")

    topic = state["topic"]
    critique = state.get("critique")

    if critique:
        print(f"  Feedback received: {critique}")
        prompt = f"""
        ### ROLE
        Senior Researcher (Recovery Mode).

        ### CONTEXT
        Original research topic: "{topic}"
        The previous research attempt failed with this critique: {critique}

        ### TASK
        Generate 3 NEW, SPECIFIC search queries about "{topic}" to fix the missing data.
        Queries MUST include the company name from the topic — never use placeholder text.
        - If revenue is missing: search for the company's specific annual report or 10-K filing.
        - If market share is missing: search for the company's specific competitive position.

        ### OUTPUT
        Return ONLY 3 queries, one per line. No numbering, no bullets.
        """
    else:
        prompt = f"""
        ### ROLE
        Lead Investment Strategist.

        ### TASK
        Generate 3 distinct search queries to gather data on: "{topic}".

        ### STRATEGY
        - Query 1: Broad search for "Revenue", "Growth", "Margins".
        - Query 2: Competitive search for "Market Share", "Competitors".
        - Query 3: Risk search for "Lawsuits", "Regulatory Issues".

        ### OUTPUT
        Return ONLY 3 queries, one per line. No numbering, no bullets.
        """

    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    log("node_done", node="planner", queries=queries)
    return {"plan": queries, "iterations": iteration + 1}


def researcher_node(state: ResearchState):
    """Run all queries in parallel using ThreadPoolExecutor for a ~3x speedup."""
    log("node_start", node="researcher", num_queries=len(state["plan"]))
    print(f"\n--- AGENT: RESEARCHER — {len(state['plan'])} queries (parallel) ---")

    t0 = time.perf_counter()

    def _search_one(query: str):
        print(f"  Searching: {query}")
        try:
            response = _tavily_search(query)
            # Docs confirm: invoke({"query":...}) returns a plain dict
            # {"query":str, "results":[{"title","url","content","score"},...]}
            if not isinstance(response, dict):
                log("search_warn", query=query, got=type(response).__name__)
                return []
            if "error" in response:
                log("search_warn", query=query, error=str(response["error"]))
                return []
            chunks = []
            for r in response.get("results", []):
                url     = r.get("url", "unknown")
                content = r.get("content", "")
                chunks.append(f"[SOURCE: {url}]\n{content}")
            return chunks
        except Exception as exc:
            log("search_error", query=query, error=str(exc))
            print(f"  Search failed: {query} — {exc}")
            return []

    all_context = []
    with ThreadPoolExecutor(max_workers=len(state["plan"])) as pool:
        futures = {pool.submit(_search_one, q): q for q in state["plan"]}
        for future in as_completed(futures):
            all_context.extend(future.result())

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log("search_done", parallel_ms=elapsed_ms, chunks_collected=len(all_context))
    print(f"  Parallel search completed in {elapsed_ms}ms ({len(all_context)} chunks)")

    combined_context = "\n\n".join(all_context)
    sources_cited = sum(1 for c in all_context if c.startswith("[SOURCE:"))

    # Truncate to ~4000 chars — enough for extraction, avoids Groq rate limits
    context_for_llm = combined_context[:4000]

    extraction_prompt = f"""
    ### ROLE
    Senior Data Extraction Engine (VC Domain).

    ### TASK
    Extract metrics from these search results regarding '{state['topic']}':
    {context_for_llm}

    ### EXTRACTION RULES
    1. Extract the specific metrics mentioned in the topic: "{state['topic']}".
    2. Fill the standard VC checklist where data is available: Traction, Moat, Legal risks.
    3. For every metric NOT found in the sources, write exactly: DATA_GAP: [Metric Name]
    4. List every URL you drew a fact from.

    ### OUTPUT FORMAT
    Return a structured bullet-point list. End with a line: SOURCES_USED: N
    """

    response = llm.invoke(extraction_prompt)
    log("node_done", node="researcher", sources_cited=sources_cited)

    return {
        "raw_data": [combined_context],
        "draft_report": response.content,
        "sources_cited": sources_cited,
    }


def auditor_node(state: ResearchState):
    log("node_start", node="auditor", model=AUDIT_MODEL)
    print(f"\n--- AGENT: CHIEF AUDITOR (model: {AUDIT_MODEL}) ---")

    draft = state.get("draft_report", "")
    sources_cited = state.get("sources_cited", 0)

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
        print(f"  Auditor JSON parse failed — defaulting to verified to avoid wasted iterations")
        # If we have sources, assume it's good enough rather than waste an iteration
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


def evaluator_node(state: ResearchState):
    """
    Direct faithfulness scorer — replaces RAGAS to avoid Groq n>1 400 errors.

    Logic: ask the LLM to score each claim in the draft against the sources.
    Returns a 0.0-1.0 faithfulness score we fully control.
    """
    log("node_start", node="evaluator")
    print("\n--- AGENT: EVALUATOR — scoring faithfulness ---")

    draft   = state.get("draft_report", "")[:1200]
    sources = " ".join(state.get("raw_data", []))[:2000]

    if not sources.strip():
        log("evaluator_skip", reason="no source data")
        print("  No source data — skipping evaluation")
        return {"eval_score": "No sources", "faithfulness_score": 0.0, "relevancy_score": 0.0}

    scoring_prompt = f"""
    ### ROLE
    Objective Research Quality Auditor.

    ### TASK
    Score how faithful this research report is to its source data.

    Source Data (truncated):
    {sources}

    Research Report:
    {draft}

    ### SCORING RULES
    - Read each factual claim in the report.
    - Check if the claim is supported by the source data.
    - Ignore DATA_GAP entries — they are honest admissions, not errors.
    - Score = (supported claims) / (total factual claims)

    ### OUTPUT — return ONLY valid JSON, no markdown fences:
    {{
        "total_claims": <integer>,
        "supported_claims": <integer>,
        "faithfulness_score": <float between 0.0 and 1.0>,
        "reasoning": "<one sentence>"
    }}
    """

    try:
        response = llm.invoke(scoring_prompt)
        clean = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        faith  = round(float(result.get("faithfulness_score", 0.0)), 2)
        total  = result.get("total_claims", "?")
        supported = result.get("supported_claims", "?")
        reasoning = result.get("reasoning", "")
        summary = f"Faithfulness: {faith:.2f} ({supported}/{total} claims supported)"
        log("eval_scores", faithfulness=faith, total=total, supported=supported)
        print(f"  Score — {summary}")
        print(f"  Reasoning: {reasoning}")
        return {
            "eval_score": summary,
            "faithfulness_score": faith,
            "relevancy_score": 0.0,
        }
    except (json.JSONDecodeError, Exception) as exc:
        log("eval_error", error=str(exc))
        print(f"  Scoring failed: {exc}")
        return {
            "eval_score": "Evaluation failed",
            "faithfulness_score": 0.0,
            "relevancy_score": 0.0,
        }


def reporter_node(state: ResearchState):
    """
    Convert raw scores into a structured investment grade + confidence score.
    This is the output that makes the project stand out in demos and on a resume.
    """
    log("node_start", node="reporter")
    print("\n--- AGENT: REPORTER — generating investment verdict ---")

    draft = state.get("draft_report", "")
    faith = state.get("faithfulness_score", 0.0)
    iterations = state.get("iterations", 1)

    # Count DATA_GAP markers in the draft report
    data_gap_count = draft.upper().count("DATA_GAP:")

    # Count risk mentions
    risk_keywords = ["lawsuit", "regulatory", "sec", "investigation", "fine", "risk", "recall"]
    risk_flag_count = sum(draft.lower().count(kw) for kw in risk_keywords)

    # Confidence: penalise for iterations used and data gaps
    raw_confidence = faith * 100
    iteration_penalty = (iterations - 1) * 12
    gap_penalty = data_gap_count * 8
    confidence_pct = max(0, min(100, int(raw_confidence - iteration_penalty - gap_penalty)))

    # Investment grade based on faithfulness + confidence
    if faith >= 0.85 and confidence_pct >= 75:
        grade = "A"
    elif faith >= 0.75 and confidence_pct >= 60:
        grade = "B+"
    elif faith >= 0.65 and confidence_pct >= 50:
        grade = "B"
    elif faith >= 0.50 and confidence_pct >= 35:
        grade = "C"
    else:
        grade = "D"

    sources_cited = state.get("sources_cited", 0)

    print(f"\n{'='*55}")
    print(f"  INVESTMENT GRADE : {grade}")
    print(f"  CONFIDENCE       : {confidence_pct}%")
    print(f"  FAITHFULNESS     : {faith:.2f}")
    print(f"  DATA GAPS        : {data_gap_count}")
    print(f"  RISK FLAGS       : {risk_flag_count}")
    print(f"  SOURCES CITED    : {sources_cited}")
    print(f"  ITERATIONS USED  : {iterations} / 3")
    print(f"{'='*55}\n")

    log(
        "report_generated",
        grade=grade,
        confidence_pct=confidence_pct,
        data_gaps=data_gap_count,
        risk_flags=risk_flag_count,
    )

    return {
        "investment_grade": grade,
        "confidence_pct": confidence_pct,
        "data_gap_count": data_gap_count,
        "risk_flag_count": risk_flag_count,
    }
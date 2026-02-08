import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from shared import ResearchState
import json
load_dotenv()

# 1. Setup the OpenRouter Brain
api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Safety Check: If this prints 'None', your .env file is the problem
print(f">>> DEBUG: API Key Loaded: {'Yes' if api_key else 'No'}")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found! Check your .env file.")

# 3. Initialize the Brain
llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free", 
    api_key=api_key, # Explicitly pass the variable we just checked
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "AI Auditor Project",
    }
)

# 2. Setup the Search Tool
search_tool = TavilySearch(max_results=2)

# Agent 1: The Researcher (Scout)
def researcher_node(state: ResearchState):
    print(f"--- 🔍 AGENT: RESEARCHER EXECUTING {len(state['plan'])} QUERIES ---")
    
    # 1. Gather all search results into one big context string
    all_raw_data = []
    for search_query in state["plan"]:
        print(f"Searching: {search_query}")
        results = search_tool.invoke(search_query)
        all_raw_data.append(str(results))
    
    combined_context = "\n\n".join(all_raw_data)

    # 2. NOW call the LLM to use your sophisticated extraction prompt
    extraction_prompt = f"""
    ### ROLE
    Senior Data Extraction Engine (VC Domain).

    ### TASK
    Extract metrics from these Search Results regarding '{state['topic']}': 
    {combined_context}

    ### EXTRACTION CHECKLIST
    - Traction: [Revenue, Churn, LTV/CAC, Growth]
    - Moat: [Network Effects, Data Advantage, Switching Costs]
    - Legal: [IP Status, Cap Table, Litigation]

    ### EXTRACTION RULES
    1. If a metric is NOT found, write "DATA_GAP: [Metric Name]". 
    2. Provide a "Source Quote" and a [URL] for every finding.
    3. STRICTLY redact any PII (personal emails or phones).

    ### OUTPUT
    A structured bullet-point list.
    """
    
    # We use the LLM to 'read' the search results
    response = llm.invoke(extraction_prompt)
    
    # 3. Update the state with the EXTRACTED data, not just the raw HTML
    return {
        "raw_data": [combined_context],  # Save for the Auditor to verify
        "draft_report": response.content, # This goes to the Auditor for judging
        "iterations": state.get("iterations", 0) + 1
    }


def planner_node(state: ResearchState):
    print("--- 🧠 AGENT: PLANNER IS STRATEGIZING ---")
    topic = state["topic"]
    
    prompt = f"""
    ### ROLE
    Lead Investment Strategist & Due Diligence Architect.

    ### OBJECTIVE
    Decompose the company/topic '{topic}' into three high-integrity investigative vectors.

    ### VECTOR 1: TRACTION & UNIT ECONOMICS
    Targeted Metrics: Revenue (ARR/MRR), Growth Rate (YoY/MoM), Customer Retention/Churn, LTV/CAC Ratio, and evidence of Signed Contracts/Pilot Results.

    ### VECTOR 2: STRATEGIC MOAT & DEFENSIBILITY
    Targeted Analysis: Network Effects, Data Advantage (Proprietary datasets), Switching Costs, Brand Equity, and Distribution Moats. Find what "stops others" from competing.

    ### VECTOR 3: FOUNDATIONAL & LEGAL COMPLIANCE
    Targeted Documents: Company Incorporation details, IP Ownership/Patents, Founder Agreements, Cap Table structure, Pending Litigation, and Regulatory Approvals.

    ### OUTPUT CONSTRAINT
    - Return ONLY 3 search queries.
    - Each query must be a long-form search string optimized to find the specific metrics listed above.
    - One query per line. No headers.
    """
    
    response = llm.invoke(prompt)
    # Split the response into a list of 3 strings
    queries = response.content.strip().split("\n")
    
    return {"plan": queries}
# Agent 2: The Auditor (Verifier)
def auditor_node(state: ResearchState):
    print("--- ⚖️ AGENT: CHIEF AUDITOR IS REVIEWING ---")
    
    # We compare the Researcher's Draft against the Raw Search Results
    audit_prompt = f"""
    ### ROLE
    Chief Investment Compliance Officer.

    ### CONTEXT
    1. Raw Search Data: {state['raw_data']}
    2. Researcher's Draft: {state['draft_report']}

    ### MANDATORY CHECKLIST
    - CRITICAL METRICS: Are Revenue, Churn, and LTV/CAC present?
    - HALLUCINATION: Does the draft claim a number NOT found in the raw data?
    - CITATIONS: Does every fact have a [URL]?
    - 2026 RELEVANCY: Is the data current?

    ### OUTPUT INSTRUCTIONS
    You must respond in valid JSON format ONLY. 
    If metrics are missing or facts are unverified, set is_verified to false.

    JSON Structure:
    {{
        "is_verified": bool,
        "critique": "Detailed list of what to fix",
        "missing_metrics": ["list", "of", "gaps"]
    }}
    """
    
    # Use the LLM to judge
    response = llm.invoke(audit_prompt)
    
    # Simple JSON Parsing (Senior move: always handle parsing errors)
    try:
        audit_results = json.loads(response.content)
    except:
        # Fallback if the LLM fails to return perfect JSON
        audit_results = {"is_verified": True, "critique": "None", "missing_metrics": []}

    return {
        "is_verified": audit_results["is_verified"],
        "critique": audit_results["critique"],
        "is_verified": audit_results["is_verified"]
    }
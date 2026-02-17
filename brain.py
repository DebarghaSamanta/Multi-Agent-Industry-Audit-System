import os
import json
from dotenv import load_dotenv

# --- THE CRITICAL FIX: Use the specific tool that returns a List, not a String ---
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# RAGAS Imports
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_community.embeddings import DeterministicFakeEmbedding

# Import State
from shared import ResearchState

load_dotenv()

# --- 1. SETUP BRAIN & TOOLS ---
api_key_openrouter = os.getenv("OPENROUTER_API_KEY")

if not api_key_openrouter:
    raise ValueError("OPENROUTER_API_KEY not found! Check your .env file.")

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free", 
    api_key=api_key_openrouter,
    base_url="https://openrouter.ai/api/v1",
    n=1,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "AI Auditor Project",
    }
)

# FIX: Use TavilySearchResults (Returns List[Dict]) instead of TavilySearch (Returns String)
search_tool = TavilySearchResults(max_results=5)

# --- 2. AGENT NODES ---

def planner_node(state: ResearchState):
    print(f"\n--- 🧠 AGENT: PLANNER (Iteration {state.get('iterations', 0)}) ---")
    topic = state["topic"]
    critique = state.get("critique")
    
    # CASE 1: RE-PLANNING (Fixing the Gaps)
    if critique:
        print(f"⚠️ FEEDBACK RECEIVED: {critique}")
        prompt = f"""
        ### ROLE
        Senior Researcher (Recovery Mode).
        
        ### CONTEXT
        The previous research failed.
        Critique: {critique}
        
        ### TASK
        Generate 3 NEW, SPECIFIC search queries to fix the missing data mentioned in the critique.
        - If "Revenue" is missing, search for "Investor Presentation" or "10-K".
        - If "LTV/CAC" is missing, search for "Unit Economics" or "Profitability".
        
        ### OUTPUT
        Return ONLY 3 queries, one per line.
        """
        
    # CASE 2: FIRST RUN (Standard Strategy)
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
        Return ONLY 3 queries, one per line.
        """

    response = llm.invoke(prompt)
    queries = response.content.strip().split("\n")
    
    return {"plan": queries, "iterations": state.get("iterations", 0) + 1}


def researcher_node(state: ResearchState):
    print(f"--- 🔍 AGENT: RESEARCHER EXECUTING {len(state['plan'])} QUERIES ---")
    
    all_context = []

    for search_query in state["plan"]:
        print(f"Searching: {search_query}")
        try:
            # This now returns a LIST of dictionaries guaranteed
            results = search_tool.invoke(search_query)

            for r in results:
                # This loop caused the crash before because 'r' was a letter, not a dict
                all_context.append(f"[SOURCE: {r.get('url')}]\n{r.get('content')}")
                
        except Exception as e:
            print(f"⚠️ Search failed for query '{search_query}': {e}")

    # Build cleaner context
    combined_context = "\n\n".join(all_context)

    # Extraction Logic
    extraction_prompt = f"""
    ### ROLE
    Senior Data Extraction Engine (VC Domain).

    ### TASK
    Extract metrics from these Search Results regarding '{state['topic']}': 
    {combined_context}

    ### EXTRACTION RULES
    1. Look for the specific metrics requested in the original topic: "{state['topic']}".
    2. Additionally, fill out the standard VC Checklist if data is available:
    - Traction, Moat, Legal.
    3. If a specific metric requested by the user is NOT found, write "DATA_GAP: [Metric Name]".

    ### OUTPUT
    A structured bullet-point list.
    """
    
    response = llm.invoke(extraction_prompt)
    
    return {
        "raw_data": [combined_context],
        "draft_report": response.content
    }


def auditor_node(state: ResearchState):
    print("--- ⚖️ AGENT: CHIEF AUDITOR IS REVIEWING ---")
    
    audit_prompt = f"""
    ### ROLE
    Chief Investment Compliance Officer.
    
    ### TASK
    Review these search results against the draft report.
    Raw Data: {state.get('raw_data')[-1] if state.get('raw_data') else 'No Data'}
    Draft Report: {state.get('draft_report')}
    
    ### OUTPUT JSON
    {{
        "is_verified": boolean,
        "critique": "string reason for rejection",
        "missing_metrics": ["list", "of", "metrics"]
    }}
    """
    
    response = llm.invoke(audit_prompt)
    
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        audit_results = json.loads(clean_content)
    except:
        print("⚠️ AUDITOR JSON PARSING FAILED - Defaulting to Retry")
        audit_results = {
            "is_verified": False, 
            "critique": "Auditor failed to parse JSON output.",
            "missing_metrics": []
        }
    
    # Returns a DICTIONARY (Fixes the Router Crash)
    return {
        "is_verified": audit_results.get("is_verified", False),
        "critique": audit_results.get("critique", "Unknown Error")
    }


def evaluator_node(state: ResearchState):
    print("--- 📊 AGENT: RAGAS EVALUATOR IS SCORING ---")

    judge_llm = LangchainLLMWrapper(llm)
    fake_embeddings = DeterministicFakeEmbedding(size=1536)
    
    data_sample = {
        "question": [state["topic"]],
        "answer": [state["draft_report"]],
        "contexts": [state["raw_data"]],
    }
    
    dataset = Dataset.from_dict(data_sample)
    try:
        score = evaluate(
            dataset, 
            metrics=[faithfulness, answer_relevancy],
            llm=judge_llm,
            embeddings=fake_embeddings 
        )
        
        eval_summary = f"Faithfulness: {score['faithfulness']:.2f}, Relevancy: {score['answer_relevancy']:.2f}"
        print(f"📈 EVALUATION SCORES: {eval_summary}")
        return {"eval_score": eval_summary}
        
    except Exception as e:
        print(f"⚠️ RAGAS Scoring skipped due to: {e}")
        return {"eval_score": "Evaluation failed - check API limits"}
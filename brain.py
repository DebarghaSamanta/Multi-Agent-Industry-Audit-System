import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from shared import ResearchState

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
    print("--- 🔍 AGENT: RESEARCHER IS SEARCHING ---")
    query = state["topic"]
    results = search_tool.invoke(query)
    # We save the results into the notebook
    return {"raw_data": [str(results)], "iterations": state.get("iterations", 0) + 1}

# Agent 2: The Auditor (Verifier)
def auditor_node(state: ResearchState):
    print("--- ⚖️ AGENT: AUDITOR IS CHECKING FOR HALLUCINATIONS ---")
    data = state["raw_data"]
    
    prompt = f"Based on this data: {data}, write a 1-sentence summary. If the data is too short, say 'REJECT'."
    response = llm.invoke(prompt)
    
    if "REJECT" in response.content.upper():
        return {"is_verified": False, "critique": "Information too thin."}
    return {"draft_report": response.content, "is_verified": True}
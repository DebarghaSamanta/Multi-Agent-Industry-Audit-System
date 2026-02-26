🕵️‍♂️ **Investment-Grade Due Diligence Auditor
An Autonomous Multi-Agent Research System**

This project automates the high-stakes "First-Pass" due diligence required for Venture Capital investments. Using LangGraph, it coordinates multiple specialized AI agents to investigate a company’s financials, competitive moat, and execution risks while maintaining strict fact-check guardrails.

🏗️ The Architecture: "Reflexive Graph"
Unlike standard RAG (Retrieval-Augmented Generation) which simply summarizes data, this system utilizes a State Machine to ensure data integrity through a self-correction loop.

The system operates as a **State Machine** with 4 distinct nodes:
1.  **🧠 Planner:** Breaks down complex user queries (e.g., "Analyze Tesla 2025") into targeted search vectors.
2.  **🔍 Researcher:** Executes queries using the **Tavily Search API** and extracts structured evidence.
3.  **⚖️ Auditor (The Core Innovation):** Compares the Researcher's draft against raw data.
    * *If Verified:* Passes to Final Review.
    * *If Rejected:* Sends a **Critique** back to the Planner (Feedback Loop).
4.  **📊 Evaluator:** Scores the final report for Faithfulness using **RAGAS** logic.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (Cyclic DAGs)
* **LLM Logic:** Groq / OpenRouter
* **Search:** Tavily API
* **Database:** SQLite (State Persistence)

## ⚡ Key Features
* **Self-Correction:** The system loops until it finds the truth or hits a safety limit.
* **Human-in-the-Loop:** Interrupts execution before sensitive searches to allow human oversight.
* **Domain Adaptive:** Automatically switches strategies for Public Co (10-K), Startups (Crunchbase), or AgriTech (Local Reports).

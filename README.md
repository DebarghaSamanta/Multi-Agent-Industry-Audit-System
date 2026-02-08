🕵️‍♂️ Investment-Grade Due Diligence Auditor
An Autonomous Multi-Agent Research System

This project automates the high-stakes "First-Pass" due diligence required for Venture Capital investments. Using LangGraph, it coordinates multiple specialized AI agents to investigate a company’s financials, competitive moat, and execution risks while maintaining strict fact-check guardrails.

🏗️ The Architecture: "Reflexive Graph"
Unlike standard RAG (Retrieval-Augmented Generation) which simply summarizes data, this system utilizes a State Machine to ensure data integrity through a self-correction loop.

1. Strategic Planner (The Executive)
Decomposes a broad topic into three orthogonal investigative vectors: Financials, Competitive Moat, and Compliance/Legal.

Generates optimized, high-density search queries for deep-web extraction.

2. Recursive Researcher (The Scout)
Executes parallel searches using the Tavily API.

Extracts specific investment-grade metrics: Revenue (ARR), Churn, LTV/CAC, Growth Rate, and IP Status.

Implements Source Attribution, ensuring every fact is backed by a verifiable URL.

3. Guardrail Auditor (The Judge)
Uses a high-reasoning LLM (e.g., Llama 3.3 70B or GPT-OSS 120B) to fact-check the researcher's draft.

Self-Correction: If critical metrics are missing or unverified, the Auditor rejects the report and triggers a recursive re-search with targeted instructions to fill the "Data Gaps."

🛠️ Tech Stack
Orchestration: LangGraph (Stateful Multi-Agent Workflows)

LLM Providers: Groq (LPU-accelerated inference) & OpenRouter

Real-time Search: Tavily (LLM-optimized search engine)

Observability: LangSmith (Full execution tracing and debugging)

📊 Observability with LangSmith
Every audit is fully traceable. My system integrates LangSmith to provide a "Window into the Brain," allowing me to:

Visualize the decision-making process between nodes.

Debug "AuthenticationError" or "RateLimit" failures in real-time.

Monitor latency and token efficiency across complex agentic loops.
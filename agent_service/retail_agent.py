"""
RetailMind Autonomous Research Agent
=====================================
Uses:
  - Groq API       → LLaMA 3.3 70B Versatile (ultra-fast inference)
  - CrewAI         → Multi-agent orchestration (Researcher → Analyst → Writer)
  - DuckDuckGo     → Free web search, no API key required
  - Auto-saves findings to /knowledge_repo/ as .txt files
  - Bridges findings back into the RAG vector store
"""

import os
import re
import logging
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retailmind.agent")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GROQ_API_KEY  = os.environ["GROQ_API_KEY"]
KNOWLEDGE_DIR = Path("./knowledge_repo")
KNOWLEDGE_DIR.mkdir(exist_ok=True)

FAST_MODEL = "llama-3.1-8b-instant"
SMART_MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────
# Groq LLM (shared across all agents)
# ──────────────────────────────────────────────
def get_groq_llm(model_name: str, temperature: float = 0.0) -> LLM:
    """
    Creates a CrewAI LLM instance for a specific Groq model.
    """
    return LLM(
        model=f"groq/{model_name}",
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=4096,
    )


# ──────────────────────────────────────────────
# DuckDuckGo Search Tool (free, no API key)
# ──────────────────────────────────────────────
_ddg_wrapper = DuckDuckGoSearchAPIWrapper(
    region="wt-wt",        # worldwide results
    safesearch="moderate",
    time="y",              # results from the past year
    max_results=2,
)
_ddg_run = DuckDuckGoSearchRun(api_wrapper=_ddg_wrapper)


@tool("DuckDuckGo Retail Search")
def search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo for retail market intelligence.
    Returns a text summary of the top search results.
    Automatically appends 'retail' context to improve result quality.

    Args:
        query: The search query string.

    Returns:
        String of search results from DuckDuckGo.
    """
    # Append 'retail' context if not already present to bias towards relevant sources
    retail_query = query if "retail" in query.lower() else f"{query} retail industry"
    logger.info(f"DuckDuckGo search: '{retail_query}'")

    try:
        results = _ddg_run.run(retail_query)
        if not results:
            return "No results found for this query."
        return results[:4000]

    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}. Retrying without modifier...")
        try:
            retry_results = _ddg_run.run(query)
            if not retry_results:
                return "No results found."
            return retry_results[:4000]
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return "Search failed."


# ──────────────────────────────────────────────
# Agent Definitions
# ──────────────────────────────────────────────
def build_researcher_agent() -> Agent:
    # Use 8B because search results contain a lot of "token noise"
    llm_8b = get_groq_llm(model_name=SMART_MODEL)
    return Agent(
        role="Senior Retail Market Researcher",
        goal="Analyze competitors and market positions.",
        backstory="Expert in understanding brand strategies, strengths and weaknesses in retail industry.",
        tools=[search_tool],
        llm=llm_8b,
        max_iter=1, # Keeps the agent from doing too many loops
        verbose=True,
        allow_delegation=False
    )


def build_analyst_agent() -> Agent:
    # Use 70B for the "Brain" work. It receives the summarized research,
    # so it won't hit the token limit as easily.
    llm_70b = get_groq_llm(model_name=SMART_MODEL)
    return Agent(
        role="Retail Intelligence Analyst",
        goal="Synthesize complex research into strategic trends and pricing .",
        backstory="A master of critical thinking and retail strategy.",
        llm=llm_70b,
        verbose=True,
        allow_delegation=False
    )


def build_writer_agent() -> Agent:
    # Use 8B because formatting a report is a mechanical task
    llm_8b = get_groq_llm(model_name=FAST_MODEL)
    return Agent(
        role="Buisness Report Writer",
        goal="Create a clear final report combining all analysist.",
        backstory="A professional business writer focused on clarity.",
        llm=llm_8b,
        verbose=True,
        allow_delegation=False
    )


# ──────────────────────────────────────────────
# Task Definitions
# ──────────────────────────────────────────────
def build_research_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"""Research the following retail intelligence topic thoroughly:
           QUERY: {query}
            Provide key market facts """
        ),
        expected_output=(
           "Bullet Insights"
        ),
        agent=agent,
    )


def build_analysis_task(agent: Agent, research_task: Task) -> Task:
    return Task(
        description=(
            "Analyze competitors and extract key trends."
            "Provide pricing strategy recommendations."
        ),
        expected_output=(
            "Competitive analysis and Pricing insights"

        ),
        agent=agent,
        context=[research_task],   # receives researcher output as context
    )


def build_writing_task(agent: Agent, analysis_task: Task, ) -> Task:

    return Task(
        description=(
           """ Generate final report including:
                - Executive Summary
                - Key Market Trends
                - Competitive Insights
                - Pricing Strategy
                - Strategic Recommendations
                """
        ),
        expected_output=(
            "Structured professional report"
        ),
        agent=agent,
        context=[analysis_task],   # receives analyst output as context
    )


# ──────────────────────────────────────────────
# Knowledge Repository — File Persistence
# ──────────────────────────────────────────────
def save_to_knowledge_repo(content: str, query: str, model_used: str) -> Path:
    """Save agent report to timestamped .txt file with correct model metadata."""
    safe_query = re.sub(r"[^a-z0-9_]", "_", query.lower())[:40]
    timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename   = f"{timestamp}_{safe_query}.txt"
    filepath   = KNOWLEDGE_DIR / filename

    header = (
        f"RetailMind Intelligence Report\n"
        f"{'='*50}\n"
        f"Query     : {query}\n"
        f"Generated : {datetime.utcnow().isoformat()} UTC\n"
        # Use the passed variable here:
        f"Model     : Groq / {model_used}\n"
        f"{'='*50}\n\n"
    )

    filepath.write_text(header + content, encoding="utf-8")
    logger.info(f"Report saved: {filepath}")
    return filepath


# ──────────────────────────────────────────────
# Main Crew Runner
# ──────────────────────────────────────────────
def run_retail_research_agent(query: str) -> dict:
    """
    Run the full 3-agent CrewAI pipeline for a retail research query.

    Returns:
        {
          "report":    str,       # full report text
          "filepath":  Path,      # saved .txt file path
          "query":     str,
          "model":     str,
          "timestamp": str,
        }
    """
    logger.info(f"Starting RetailMind agent for query: '{query}'")



    # Build agents
    researcher = build_researcher_agent()
    analyst    = build_analyst_agent()
    writer     = build_writer_agent()

    # Build tasks (chained: research → analysis → writing)
    research_task  = build_research_task(researcher, query)
    analysis_task  = build_analysis_task(analyst, research_task)
    writing_task   = build_writing_task(writer, analysis_task, query)

    # Assemble crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,   # tasks run in order, outputs chain
        verbose=True,
        memory=False,                 # stateless per run
    )

    # Kick off
    result     = crew.kickoff()
    report_txt = str(result)

    # Persist to knowledge repo
    model_desc = f"{FAST_MODEL} & {SMART_MODEL}"
    filepath = save_to_knowledge_repo(report_txt, query, model_desc)

    # Bridge into RAG: auto-ingest the new report
    try:
        from rag_service.rag_pipeline import ingest_agent_summary
        ingest_agent_summary(report_txt, filepath.name)
        logger.info("Report ingested into RAG vector store")
    except Exception as e:
        logger.warning(f"RAG bridge skipped: {e}")

    return {
        "report":    report_txt,
        "filepath":  filepath,
        "query":     query,
        "model":     f"{FAST_MODEL} & {SMART_MODEL}",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "Latest trends in AI-powered retail personalisation 2024"
    output = run_retail_research_agent(query)
    print("\n" + "="*60)
    print(output["report"])
    print(f"\nSaved to: {output['filepath']}")

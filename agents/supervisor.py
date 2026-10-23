# =========================================================
# PROJECT PATH FIX
# Must run before importing project modules.
# =========================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# IMPORTS
# =========================================================

import re
import tempfile
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph


# =========================================================
# SHARED AGENT STATE
# =========================================================

class AnalystState(TypedDict, total=False):
    question: str
    dataframe: Any
    dataset_path: str
    column_mapping: dict
    local_answer: str | None

    route: str
    answer: str
    source: str

    # Database Agent output
    sql: str | None
    database_results: Any

    # Web Research Agent output
    web_results: list[dict]
    web_research_question: str
    web_searched_at: str
    web_error: str | None


# =========================================================
# SUPERVISOR NODE
# =========================================================

def supervisor_node(state: AnalystState) -> dict:
    """Route each question to the appropriate specialist agent."""

    question = state.get("question", "").strip().lower()

    if not question:
        return {
            "route": "file",
            "answer": "Please enter a question.",
            "source": "Supervisor",
        }

    database_patterns = [
        r"\bsql\b",
        r"\bdatabase\b",
        r"\bmysql\b",
        r"\bpostgres(?:ql)?\b",
        r"\bsqlite\b",
        r"\bdatabase table\b",
        r"\brun a database query\b",
        r"\bquery the database\b",
        r"\bquery my database\b",
        r"\bfrom the database\b",
        r"\bin the database\b",
    ]

    web_patterns = [
        r"\bsearch the web\b",
        r"\bsearch online\b",
        r"\bweb search\b",
        r"\bweb research\b",
        r"\bonline research\b",
        r"\bexternal research\b",
        r"\blatest news\b",
        r"\bcurrent news\b",
        r"\brecent news\b",
        r"\bsearch for news\b",
        r"\bcompetitor research\b",
        r"\bcompetitors online\b",
        r"\bresearch online\b",
        r"\bfind online\b",
        r"\blook up online\b",
        r"\bsearch the internet\b",
    ]

    if any(
        re.search(pattern, question)
        for pattern in database_patterns
    ):
        route = "database"

    elif any(
        re.search(pattern, question)
        for pattern in web_patterns
    ):
        route = "web"

    else:
        route = "file"

    return {
        "route": route,
        "answer": "",
        "source": "",
        "sql": None,
        "database_results": None,
        "web_results": [],
        "web_research_question": "",
        "web_searched_at": "",
        "web_error": None,
    }


# =========================================================
# FILE AGENT NODE
# =========================================================

def file_agent_node(state: AnalystState) -> dict:
    """Analyze the supplied dataset using the existing File Agent."""

    question = state.get("question", "").strip()
    local_answer = state.get("local_answer")

    # Use the existing local Pandas answer when available.
    if local_answer:
        return {
            "answer": local_answer,
            "source": "Pandas / File Agent",
        }

    dataset_path = state.get("dataset_path")
    dataframe = state.get("dataframe")
    temporary_path = None

    try:
        if dataset_path:
            file_path = str(dataset_path)

            if not Path(file_path).exists():
                return {
                    "answer": f"Dataset file not found: {file_path}",
                    "source": "File Agent",
                }

        elif isinstance(dataframe, pd.DataFrame):
            if dataframe.empty:
                return {
                    "answer": "The dataset contains no rows.",
                    "source": "File Agent",
                }

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                prefix="analyst_",
                encoding="utf-8",
                newline="",
                delete=False,
            ) as temp_file:
                temporary_path = temp_file.name
                dataframe.to_csv(temp_file, index=False)

            file_path = temporary_path

        else:
            return {
                "answer": (
                    "No dataset was supplied. Upload a CSV or Excel "
                    "file before asking a data question."
                ),
                "source": "File Agent",
            }

        from agents.file_agent import analyze_dataset

        result = analyze_dataset(
            file_path,
            user_question=question,
        )

        return {
            "answer": (
                str(result)
                if result
                else "The File Agent did not return an answer."
            ),
            "source": "File Agent",
        }

    except Exception as exc:
        return {
            "answer": f"File Agent error: {exc}",
            "source": "File Agent",
        }

    finally:
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


# =========================================================
# DATABASE AGENT NODE
# =========================================================

def database_agent_node(state: AnalystState) -> dict:
    """Execute the existing SQLite + Gemini Database Agent."""

    question = state.get("question", "").strip()

    try:
        from agents.database_agent import run_database_agent

        result = run_database_agent(question=question)

        return {
            "answer": result.get(
                "answer",
                "The Database Agent returned no answer.",
            ),
            "source": result.get(
                "source",
                "Database Agent",
            ),
            "sql": result.get("sql"),
            "database_results": result.get("results"),
        }

    except Exception as exc:
        return {
            "answer": (
                "The Database Agent could not complete the request.\n\n"
                f"Error: {exc}"
            ),
            "source": "Database Agent",
            "sql": None,
            "database_results": None,
        }


# =========================================================
# WEB RESEARCH AGENT NODE
# =========================================================

def web_agent_node(state: AnalystState) -> dict:
    """Run the real DDGS-powered Web Research Agent."""

    question = state.get("question", "").strip()

    try:
        from agents.web_agent import run_web_agent

        result = run_web_agent(
            question=question,
            max_results=6,
        )

        return {
            "answer": result.get(
                "answer",
                "The Web Research Agent returned no answer.",
            ),
            "source": result.get(
                "source",
                "DDGS Web Research Agent",
            ),
            "web_results": result.get("results", []),
            "web_research_question": result.get(
                "research_question",
                question,
            ),
            "web_searched_at": result.get(
                "searched_at",
                "",
            ),
            "web_error": result.get("error"),
        }

    except Exception as exc:
        return {
            "answer": (
                "The Web Research Agent could not complete the search.\n\n"
                f"Error: {exc}\n\n"
                "Check that the ddgs package is installed and that "
                "your internet connection is working."
            ),
            "source": "DDGS Web Research Agent",
            "web_results": [],
            "web_research_question": question,
            "web_searched_at": "",
            "web_error": str(exc),
        }


# =========================================================
# SYNTHESIZER NODE
# =========================================================

def synthesizer_node(state: AnalystState) -> dict:
    """Return the selected agent's answer without inventing facts."""

    answer = state.get("answer", "").strip()

    if not answer:
        answer = "No answer was returned by the selected agent."

    return {
        "answer": answer,
        "source": state.get("source", "Supervisor"),
    }


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

def build_supervisor_graph():
    workflow = StateGraph(AnalystState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("file_agent", file_agent_node)
    workflow.add_node("database_agent", database_agent_node)
    workflow.add_node("web_agent", web_agent_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("route", "file"),
        {
            "file": "file_agent",
            "database": "database_agent",
            "web": "web_agent",
        },
    )

    workflow.add_edge("file_agent", "synthesizer")
    workflow.add_edge("database_agent", "synthesizer")
    workflow.add_edge("web_agent", "synthesizer")

    workflow.add_edge("synthesizer", END)

    return workflow.compile()


supervisor_graph = build_supervisor_graph()


# =========================================================
# PUBLIC FUNCTION FOR STREAMLIT
# =========================================================

def run_supervisor(
    question: str,
    dataframe: pd.DataFrame | None = None,
    column_mapping: dict | None = None,
    local_answer: str | None = None,
    dataset_path: str | None = None,
) -> dict:
    """Execute the multi-agent workflow and return its results."""

    initial_state: AnalystState = {
        "question": question,
        "dataframe": dataframe,
        "column_mapping": column_mapping or {},
        "local_answer": local_answer,
        "route": "file",
        "answer": "",
        "source": "",
    }

    if dataset_path:
        initial_state["dataset_path"] = dataset_path

    result = supervisor_graph.invoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "route": result.get("route", "file"),
        "source": result.get("source", "Supervisor"),

        # Database Agent output
        "sql": result.get("sql"),
        "database_results": result.get("database_results"),

        # Web Research Agent output
        "web_results": result.get("web_results", []),
        "web_research_question": result.get(
            "web_research_question",
            "",
        ),
        "web_searched_at": result.get(
            "web_searched_at",
            "",
        ),
        "web_error": result.get("web_error"),
    }


# =========================================================
# INTEGRATION TEST
# =========================================================

if __name__ == "__main__":
    print("=" * 65)
    print("LANGGRAPH MULTI-AGENT INTEGRATION TEST")
    print("=" * 65)
    print(f"Project root: {PROJECT_ROOT}")

    sample_path = (
        PROJECT_ROOT / "data" / "sample" / "sample.csv"
    )

    if not sample_path.exists():
        print(f"Sample dataset not found: {sample_path}")
        raise SystemExit(1)

    sample_df = pd.read_csv(sample_path)

    test_questions = [
        "Which product has the highest profit?",
        "Compare sales and profit by region",
        "Run a SQL query on my database",
        "Which product has the highest profit in the database?",
        "What are total sales by region in the database?",
        "Search the web for latest news about AI in data analytics",
    ]

    for question in test_questions:
        print("\n" + "-" * 65)
        print(f"Question: {question}")

        result = run_supervisor(
            question=question,
            dataframe=sample_df,
        )

        print(f"Route: {result['route']}")
        print(f"Source: {result['source']}")

        if result.get("sql"):
            print(f"Generated SQL: {result['sql']}")

        print(f"Answer:\n{result['answer']}")

        if result.get("web_results"):
            print("\nWeb source links:")

            for item in result["web_results"]:
                print(f"- {item.get('title', 'Untitled')}")
                print(f"  {item.get('url', '')}")

        if result.get("web_error"):
            print(f"Web search error: {result['web_error']}")

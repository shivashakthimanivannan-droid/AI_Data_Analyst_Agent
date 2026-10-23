
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.mysql_tools import (
    DEFAULT_DATABASE_PATH,
    execute_read_only_query,
    get_database_schema,
    initialize_database,
)

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
MAX_RETRIES = 3
MAX_RESULT_ROWS = 200


# =========================================================
# GEMINI
# =========================================================

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    return genai.Client(api_key=api_key)


def _is_quota_error(exc):
    message = str(exc).lower()

    return any(
        marker in message
        for marker in (
            "resource_exhausted",
            "generaterequestsperdaypermodel",
            "generate_content_free_tier_requests",
            "quota exceeded",
            "billing details",
        )
    )


def _generate_content_with_retry(client, prompt):
    retryable_markers = (
        "500",
        "502",
        "503",
        "504",
        "unavailable",
        "internal",
        "deadline_exceeded",
        "temporarily unavailable",
        "high demand",
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
            )

            text = getattr(response, "text", None)

            if not text or not text.strip():
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return text.strip()

        except Exception as exc:
            if _is_quota_error(exc):
                raise

            message = str(exc).lower()

            retryable = any(
                marker in message
                for marker in retryable_markers
            )

            if not retryable or attempt == MAX_RETRIES - 1:
                raise

            delay = 2 ** attempt

            print(
                f"Temporary Gemini error. Retrying in {delay}s..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini request failed after retries."
    )


# =========================================================
# SCHEMA HELPERS
# =========================================================

def _quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def _get_schema_columns(schema, table_name):
    table_info = schema[table_name]

    if isinstance(table_info, dict):
        columns = table_info.get("columns", [])
    else:
        columns = table_info

    names = []

    for column in columns:
        if isinstance(column, dict):
            name = column.get("name")

            if name:
                names.append(str(name))

        elif isinstance(column, str):
            names.append(column)

    return names


def _find_table_and_columns(schema):
    if not schema:
        raise RuntimeError(
            "No database tables were found."
        )

    table_name = (
        "sales_data"
        if "sales_data" in schema
        else next(iter(schema))
    )

    column_names = _get_schema_columns(
        schema,
        table_name,
    )

    columns = {
        name.lower(): name
        for name in column_names
    }

    return table_name, columns


def _column(columns, *names):
    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]

    return None


# =========================================================
# QUESTION HELPERS
# =========================================================

def _normalize_question(question):
    return re.sub(
        r"\s+",
        " ",
        question.lower(),
    ).strip()


def _ranking_direction(question):
    q = _normalize_question(question)

    if re.search(
        r"\b(lowest|least|minimum|smallest|worst|bottom)\b",
        q,
    ):
        return "ASC"

    if re.search(
        r"\b(highest|most|maximum|largest|greatest|top|best)\b",
        q,
    ):
        return "DESC"

    return None


def _ranking_label(question):
    direction = _ranking_direction(question)

    if direction == "ASC":
        return "lowest"

    if direction == "DESC":
        return "highest"

    return None


def _requested_dimension(question):
    q = _normalize_question(question)

    if re.search(
        r"\b(region|regions|territory|territories|area|areas)\b",
        q,
    ):
        return "region"

    if re.search(
        r"\b(product|products|category|categories)\b",
        q,
    ):
        return "product"

    return None


def _requested_metric(question):
    q = _normalize_question(question)

    if re.search(r"\bprofit\b", q):
        return "profit"

    if re.search(r"\b(sales|revenue)\b", q):
        return "sales"

    if re.search(r"\b(quantity|units)\b", q):
        return "quantity"

    return None


# =========================================================
# LOCAL SQL GENERATION
# =========================================================

def _local_sql_for_question(question, schema):
    q = _normalize_question(question)

    table, columns = _find_table_and_columns(schema)
    quoted_table = _quote_identifier(table)

    sales = _column(
        columns,
        "sales",
        "revenue",
        "total_sales",
    )

    profit = _column(
        columns,
        "profit",
        "net_profit",
    )

    quantity = _column(
        columns,
        "quantity",
        "units",
    )

    product = _column(
        columns,
        "product",
        "category",
        "product_name",
    )

    region = _column(
        columns,
        "region",
        "territory",
        "area",
    )

    def col(name):
        return _quote_identifier(name)

    metrics = {
        "sales": sales,
        "profit": profit,
        "quantity": quantity,
    }

    aliases = {
        "sales": "total_sales",
        "profit": "total_profit",
        "quantity": "total_quantity",
    }

    dimensions = {
        "region": region,
        "product": product,
    }

    broad_phrases = (
        "run a sql query",
        "run sql",
        "query my database",
        "query the database",
        "run a query on my database",
        "show me my database",
        "give me a database summary",
        "summarize my database",
        "summarise my database",
        "show database data",
        "show all database data",
        "what is in my database",
        "database overview",
    )

    if any(phrase in q for phrase in broad_phrases):
        selected = [
            "COUNT(*) AS total_records"
        ]

        for metric_name, column_name in metrics.items():
            if column_name:
                selected.append(
                    f"SUM({col(column_name)}) "
                    f"AS {aliases[metric_name]}"
                )

        return (
            f"SELECT {', '.join(selected)} "
            f"FROM {quoted_table}"
        )

    # Profit margin
    if "profit margin" in q and sales and profit:
        return (
            f"SELECT SUM({col(profit)}) AS total_profit, "
            f"SUM({col(sales)}) AS total_sales, "
            f"ROUND("
            f"100.0 * SUM({col(profit)}) / "
            f"NULLIF(SUM({col(sales)}), 0), 2"
            f") AS profit_margin_pct "
            f"FROM {quoted_table}"
        )

    direction = _ranking_direction(question)
    dimension_name = _requested_dimension(question)
    metric_name = _requested_metric(question)

    dimension_column = dimensions.get(
        dimension_name
    )

    metric_column = metrics.get(
        metric_name
    )

    # Ranking: handles BOTH product and region correctly.
    if (
        direction
        and dimension_column
        and metric_column
    ):
        alias = aliases[metric_name]

        return (
            f"SELECT {col(dimension_column)} "
            f"AS {dimension_name}, "
            f"SUM({col(metric_column)}) AS {alias} "
            f"FROM {quoted_table} "
            f"GROUP BY {col(dimension_column)} "
            f"ORDER BY {alias} {direction} "
            f"LIMIT 1"
        )

    # Grouped aggregation
    if (
        dimension_column
        and metric_column
        and re.search(
            r"\b(by|per|each)\s+"
            r"(region|regions|product|products|"
            r"category|categories|territory|area)\b",
            q,
        )
    ):
        alias = aliases[metric_name]

        return (
            f"SELECT {col(dimension_column)} "
            f"AS {dimension_name}, "
            f"SUM({col(metric_column)}) AS {alias} "
            f"FROM {quoted_table} "
            f"GROUP BY {col(dimension_column)} "
            f"ORDER BY {alias} DESC"
        )

    # Simple totals
    if (
        metric_column
        and re.search(
            r"\b(total|sum)\b",
            q,
        )
    ):
        alias = aliases[metric_name]

        return (
            f"SELECT SUM({col(metric_column)}) "
            f"AS {alias} "
            f"FROM {quoted_table}"
        )

    # Row count
    if re.search(
        r"\b(how many|number of|count|row count)\b",
        q,
    ):
        return (
            f"SELECT COUNT(*) AS total_records "
            f"FROM {quoted_table}"
        )

    # Data preview
    preview_phrases = (
        "show all records",
        "show database records",
        "show me the records",
        "show sample data",
        "preview the database",
        "display database rows",
    )

    if any(phrase in q for phrase in preview_phrases):
        return (
            f"SELECT * FROM {quoted_table} LIMIT 20"
        )

    return None


# =========================================================
# SQL FORMATTING AND GEMINI GENERATION
# =========================================================

def _extract_sql(text):
    text = text.strip()

    match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        text = match.group(1).strip()

    return text.rstrip(";").strip() + ";"


def _format_sql(sql):
    sql = sql.strip().rstrip(";")

    formatted = re.sub(
        r"\s+(FROM|WHERE|GROUP BY|HAVING|ORDER BY|LIMIT)\b",
        lambda match: "\n" + match.group(1),
        sql,
        flags=re.IGNORECASE,
    )

    return formatted + ";"


def _generate_sql(client, question, schema):
    prompt = f"""
You are a SQLite query-generation assistant.

SCHEMA:
{json.dumps(schema, indent=2, ensure_ascii=False)}

QUESTION:
{question}

Return exactly one SQLite SELECT or read-only WITH query.
Return SQL only, without Markdown or explanation.
Use only table and column names from the schema.
Never generate write operations, DDL, PRAGMA, or multiple statements.
Use GROUP BY and SUM for aggregate business questions.
Use LIMIT for individual records.
"""

    response = _generate_content_with_retry(
        client,
        prompt,
    )

    return _extract_sql(response)


# =========================================================
# RESULT FORMATTING
# =========================================================

def _format_value(value):
    if value is None:
        return "N/A"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    return str(value)


def _local_summary(question, sql, results):
    columns = results.get("columns", [])
    rows = results.get("rows", [])

    if not rows:
        return (
            "The query completed successfully, "
            "but no matching records were found."
        )

    ranking = _ranking_label(question)

    if len(rows) == 1:
        values = dict(zip(columns, rows[0]))

        # Generic dimension + metric summary
        for dimension in ("region", "product"):
            if dimension not in values:
                continue

            for metric in (
                "total_profit",
                "total_sales",
                "total_quantity",
            ):
                if metric not in values:
                    continue

                entity = values[dimension]
                value = _format_value(
                    values[metric]
                )

                metric_label = metric.replace(
                    "total_",
                    "",
                )

                if ranking:
                    return (
                        f"The {dimension} with the "
                        f"**{ranking} {metric_label}** "
                        f"is **{entity}**, with total "
                        f"{metric_label} of **{value}**."
                    )

                return (
                    f"**{entity}** has total "
                    f"{metric_label} of **{value}**."
                )

        # Profit margin
        if "profit_margin_pct" in values:
            margin = _format_value(
                values["profit_margin_pct"]
            )

            profit = _format_value(
                values.get("total_profit")
            )

            sales = _format_value(
                values.get("total_sales")
            )

            return (
                f"**Profit Margin: {margin}%**\n\n"
                f"- Total Profit: **{profit}**\n"
                f"- Total Sales: **{sales}**"
            )

        # Database summary
        if "total_records" in values:
            lines = [
                "## Database Summary",
                "",
            ]

            for key, value in values.items():
                label = key.replace(
                    "_",
                    " ",
                ).title()

                lines.append(
                    f"- **{label}:** "
                    f"{_format_value(value)}"
                )

            return "\n".join(lines)

        # Single aggregate value
        if len(columns) == 1:
            label = columns[0].replace(
                "_",
                " ",
            ).title()

            return (
                f"**{label}:** "
                f"{_format_value(rows[0][0])}"
            )

    # Multi-row results
    lines = [
        "## Query Results",
        "",
    ]

    if columns:
        lines.append(
            "| " + " | ".join(map(str, columns)) + " |"
        )

        lines.append(
            "| " + " | ".join(
                ["---"] * len(columns)
            ) + " |"
        )

        for row in rows[:20]:
            cells = [
                _format_value(value).replace(
                    "|",
                    "\\|",
                )
                for value in row
            ]

            lines.append(
                "| " + " | ".join(cells) + " |"
            )

    lines.extend([
        "",
        f"Rows returned: **{results.get('row_count', len(rows))}**.",
    ])

    return "\n".join(lines)


def _summarize_with_gemini(client, question, sql, results):
    prompt = f"""
Answer the user's question using only these actual SQL results.

QUESTION:
{question}

SQL:
{sql}

RESULTS:
{json.dumps(results, default=str, ensure_ascii=False)}

Be concise and factual. Do not invent data.
"""

    return _generate_content_with_retry(
        client,
        prompt,
    )


# =========================================================
# MAIN DATABASE AGENT
# =========================================================

def run_database_agent(
    question,
    database_path=None,
    csv_path=None,
):
    if not isinstance(question, str) or not question.strip():
        return {
            "answer": "Please enter a database question.",
            "route": "database",
            "source": "Database Agent",
            "sql": None,
            "results": None,
        }

    db_path = Path(
        database_path or DEFAULT_DATABASE_PATH
    )

    sql = None
    client = None

    try:
        initialize_database(
            csv_path=csv_path,
            database_path=db_path,
        )

        schema = get_database_schema(
            database_path=db_path
        )

        if not schema:
            raise RuntimeError(
                "No database tables were found."
            )

        sql = _local_sql_for_question(
            question,
            schema,
        )

        used_local_sql = sql is not None

        if not used_local_sql:
            try:
                client = _get_client()

                sql = _generate_sql(
                    client,
                    question,
                    schema,
                )

            except Exception as exc:
                if _is_quota_error(exc):
                    return {
                        "answer": (
                            "Gemini's daily API quota has been exhausted. "
                            "This question could not be translated "
                            "into SQL automatically."
                        ),
                        "route": "database",
                        "source": "Local Database Agent",
                        "sql": None,
                        "results": None,
                        "error": "Gemini API quota exhausted",
                    }

                raise

        results = execute_read_only_query(
            sql=sql,
            database_path=db_path,
            max_rows=MAX_RESULT_ROWS,
        )

        if used_local_sql:
            answer = _local_summary(
                question,
                sql,
                results,
            )

            source = "SQLite (Local SQL)"

        else:
            try:
                answer = _summarize_with_gemini(
                    client=client,
                    question=question,
                    sql=sql,
                    results=results,
                )

                source = "Gemini + SQLite Database Agent"

            except Exception as exc:
                print(
                    "Gemini summarization failed; "
                    "using local fallback:",
                    exc,
                )

                answer = _local_summary(
                    question,
                    sql,
                    results,
                )

                source = "SQLite (Local Result Fallback)"

        return {
            "answer": answer,
            "route": "database",
            "source": source,
            "sql": _format_sql(sql),
            "results": results,
        }

    except Exception as exc:
        return {
            "answer": (
                "The Database Agent could not complete "
                "the request.\n\n"
                f"Details: {exc}"
            ),
            "route": "database",
            "source": "Database Agent",
            "sql": sql,
            "results": None,
            "error": str(exc),
        }


# =========================================================
# REGRESSION TESTS
# =========================================================

if __name__ == "__main__":
    print("=" * 65)
    print("DATABASE AGENT - REGRESSION TEST")
    print("=" * 65)

    questions = [
        "Run a SQL query on my database",
        "Which product has the highest profit in the database?",
        "Which product has the lowest profit in the database?",
        "Which product has the highest sales?",
        "Which region has the lowest sales?",
        "Which region has the highest profit?",
        "Which region has the lowest profit?",
        "What are total sales by region?",
        "What is the total profit?",
        "What is the profit margin?",
    ]

    for index, question in enumerate(questions, start=1):
        print("\n" + "-" * 65)
        print(f"Test {index}/{len(questions)}")
        print("Question:", question)

        result = run_database_agent(question)

        print("\nSQL:")
        print(
            result.get("sql")
            or "No SQL generated"
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nSource:")
        print(result.get("source"))

        if result.get("error"):
            print("\nError:", result["error"])

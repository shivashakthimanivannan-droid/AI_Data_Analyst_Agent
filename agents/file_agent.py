
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_MODEL = "gemini-3.6-flash"

# =========================================================
# DATASET LOADING
# =========================================================

def load_dataset(file_path):
    """Load CSV or Excel data and clean column names."""
    file_path = str(file_path)
    extension = Path(file_path).suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(file_path)
    elif extension in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Upload a CSV or Excel file.")

    df.columns = [str(column).strip() for column in df.columns]
    return df


# =========================================================
# COLUMN AND FORMATTING HELPERS
# =========================================================

def _find_column(df, candidates):
    """Find a column by case-insensitive exact name."""
    columns = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


def _numeric_series(series):
    """Convert formatted numeric values into numbers."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def _to_numeric(df, column):
    return _numeric_series(df[column])


def _format_number(value):
    if pd.isna(value):
        return "N/A"

    value = float(value)

    if value.is_integer():
        return f"{value:,.0f}"

    return f"{value:,.2f}"


def _format_table(df, max_rows=20):
    """Format a DataFrame as Markdown without tabulate."""
    if df is None or df.empty:
        return "No results were found."

    display_df = df.head(max_rows).copy()
    headers = [str(column) for column in display_df.columns]

    def clean_cell(value):
        if pd.isna(value):
            return "N/A"

        if isinstance(value, (int, float)):
            return _format_number(value)

        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        [clean_cell(value) for value in row]
        for row in display_df.itertuples(index=False, name=None)
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    if len(df) > max_rows:
        lines.extend(["", f"*Showing {max_rows} of {len(df)} rows.*"])

    return "\n".join(lines)


def _get_metric_columns(df):
    """Identify common business metric columns."""
    return {
        "sales": _find_column(
            df,
            [
                "Sales", "Revenue", "Total Sales",
                "Total Revenue", "Turnover", "Amount",
            ],
        ),
        "profit": _find_column(
            df,
            [
                "Profit", "Net Profit", "Total Profit",
                "Gross Profit", "Earnings",
            ],
        ),
        "quantity": _find_column(
            df,
            ["Quantity", "Units Sold", "Units", "Items Sold"],
        ),
    }


def _requested_metrics(df, question):
    """
    Detect every metric explicitly requested.

    For example, "compare sales and profit" returns both
    columns instead of silently selecting only one.
    """
    q = question.lower()
    available = _get_metric_columns(df)
    requested = []

    patterns = {
        "sales": r"\b(sales|revenue|turnover)\b",
        "profit": r"\b(profit|earnings)\b",
        "quantity": r"\b(quantity|units sold|items sold)\b",
    }

    for metric_name, pattern in patterns.items():
        column = available.get(metric_name)

        if column is not None and re.search(pattern, q):
            requested.append((metric_name, column))

    # Margin is derived from profit and sales, not a raw metric.
    if re.search(r"\b(profit margin|profitability|margin)\b", q):
        if available["sales"] is not None and available["profit"] is not None:
            requested = [
                ("sales", available["sales"]),
                ("profit", available["profit"]),
            ]

    return requested


def _find_metric_column(df, question):
    """Return one metric for single-metric questions."""
    requested = _requested_metrics(df, question)

    if len(requested) == 1:
        return requested[0][1], requested[0][0]

    if len(requested) > 1:
        # Keep older callers working; combined comparisons are
        # handled separately by answer_local_question.
        q = question.lower()

        if re.search(r"\b(profit|earnings)\b", q):
            col = _get_metric_columns(df)["profit"]
            if col is not None:
                return col, "profit"

        if re.search(r"\b(sales|revenue|turnover)\b", q):
            col = _get_metric_columns(df)["sales"]
            if col is not None:
                return col, "sales"

        return requested[0][1], requested[0][0]

    return None, None


def _find_group_column(df, question):
    """Identify the grouping dimension requested by the user."""
    q = question.lower()

    candidates = [
        (
            r"\b(products?|items?)\b",
            ["Product", "Product Name", "Item", "Item Name", "SKU"],
        ),
        (
            r"\b(categories|category)\b",
            ["Category", "Product Category", "Sub-Category"],
        ),
        (
            r"\b(regions?|areas?|territories|markets?)\b",
            ["Region", "Area", "Territory", "Market"],
        ),
        (r"\b(cities|city|towns|town)\b", ["City", "Town"]),
        (r"\b(countries|country|nations?)\b", ["Country", "Nation"]),
        (
            r"\b(customers?|clients?)\b",
            ["Customer", "Customer Name", "Customer ID", "Client"],
        ),
        (
            r"\b(segments?)\b",
            ["Segment", "Customer Segment", "Market Segment"],
        ),
        (
            r"\b(departments?)\b",
            ["Department", "Dept"],
        ),
        (
            r"\b(orders?)\b",
            ["Order ID", "Order", "Order Number"],
        ),
    ]

    for pattern, names in candidates:
        if re.search(pattern, q):
            column = _find_column(df, names)
            if column is not None:
                return column

    # Match a real column name mentioned directly in the question.
    for column in df.columns:
        name = str(column).strip()
        if len(name) >= 2 and re.search(
            rf"(?<!\w){re.escape(name.lower())}(?!\w)", q
        ):
            if column not in _get_metric_columns(df).values():
                return column

    # Fallback for simple questions without a named dimension.
    return _find_column(
        df,
        ["Product", "Product Name", "Category", "Region", "City"],
    )


def _find_date_column(df):
    return _find_column(
        df,
        [
            "Date", "Order Date", "Transaction Date",
            "Invoice Date", "Purchase Date", "Created At",
            "Timestamp",
        ],
    )


def _group_metrics(df, group_col, metrics):
    """Aggregate multiple numeric metrics by a grouping column."""
    columns = [column for _, column in metrics]

    temp = df[[group_col] + columns].copy()

    for column in columns:
        temp[column] = _numeric_series(temp[column])

    temp = temp.dropna(subset=[group_col])

    if temp.empty:
        return pd.DataFrame()

    grouped = temp.groupby(group_col, dropna=False)[columns].sum()

    # Use readable names in the output.
    rename = {}
    for metric_name, column in metrics:
        rename[column] = metric_name.title()

    grouped = grouped.rename(columns=rename).reset_index()

    # Sort by the first requested metric, highest first.
    sort_column = rename[metrics[0][1]]
    grouped = grouped.sort_values(sort_column, ascending=False)

    return grouped


def _build_monthly_table(df, metrics):
    """Build a monthly aggregate table for one or more metrics."""
    date_col = _find_date_column(df)

    if date_col is None:
        return None

    temp = df.copy()
    temp["_ParsedDate"] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=["_ParsedDate"])

    if temp.empty:
        return None

    temp["_Month"] = temp["_ParsedDate"].dt.to_period("M").astype(str)

    for _, column in metrics:
        temp[column] = _numeric_series(temp[column])

    columns = [column for _, column in metrics]
    monthly = temp.groupby("_Month")[columns].sum().reset_index()

    rename = {"_Month": "Month"}
    for metric_name, column in metrics:
        rename[column] = metric_name.title()

    return monthly.rename(columns=rename)


# =========================================================
# LOCAL PANDAS QUESTION ANSWERING
# =========================================================

def answer_local_question(df, question):
    """
    Answer supported questions locally using Pandas.
    Return None when Gemini is needed.
    """
    if df.empty:
        return "The uploaded dataset is empty."

    q = re.sub(r"\s+", " ", question.lower().strip())
    metric_columns = _get_metric_columns(df)

    sales_col = metric_columns["sales"]
    profit_col = metric_columns["profit"]
    quantity_col = metric_columns["quantity"]

    # -----------------------------------------------------
    # DATASET SUMMARY
    # -----------------------------------------------------

    if any(phrase in q for phrase in [
        "dataset summary",
        "summarize the dataset",
        "summary of the dataset",
        "dataset overview",
        "describe the dataset",
        "tell me about this dataset",
        "how many rows",
        "how many columns",
        "data quality",
    ]):
        lines = [
            "## Dataset Summary",
            "",
            f"- **Rows:** {len(df):,}",
            f"- **Columns:** {len(df.columns):,}",
            f"- **Missing cells:** {int(df.isna().sum().sum()):,}",
            f"- **Duplicate rows:** {int(df.duplicated().sum()):,}",
            "",
            "### Column Details",
            "",
            "| Column | Data Type | Missing Values |",
            "|---|---|---:|",
        ]

        for col in df.columns:
            lines.append(
                f"| {col} | {df[col].dtype} | "
                f"{int(df[col].isna().sum()):,} |"
            )

        return "\n".join(lines)

    # -----------------------------------------------------
    # MULTI-METRIC MONTHLY COMPARISON
    # Run before single-metric handling.
    # -----------------------------------------------------

    monthly_requested = any(word in q for word in [
        "monthly", "month-wise", "month wise",
        "monthly trend", "by month", "each month",
        "over time", "month by month",
    ])

    if monthly_requested:
        requested = _requested_metrics(df, q)

        # If the user asks for a trend without naming a metric,
        # show available sales and profit together.
        if not requested:
            requested = [
                (name, col)
                for name, col in [
                    ("sales", sales_col),
                    ("profit", profit_col),
                ]
                if col is not None
            ]

        if requested:
            monthly = _build_monthly_table(df, requested)

            if monthly is None:
                return (
                    "I couldn't build a monthly trend. Check whether "
                    "the dataset has a recognizable date column with "
                    "valid date values."
                )

            if monthly.empty:
                return "No valid monthly data was found."

            lines = ["## Monthly Metric Comparison", "", _format_table(monthly)]

            sales_label = next(
                (name for name, _ in requested if name == "sales"), None
            )
            profit_label = next(
                (name for name, _ in requested if name == "profit"), None
            )

            if sales_label and profit_label:
                best_sales_row = monthly.loc[monthly["Sales"].idxmax()]
                best_profit_row = monthly.loc[monthly["Profit"].idxmax()]

                lines.extend([
                    "",
                    "### Key Findings",
                    "",
                    f"- **Highest-sales month:** {best_sales_row['Month']} "
                    f"({_format_number(best_sales_row['Sales'])}).",
                    f"- **Highest-profit month:** {best_profit_row['Month']} "
                    f"({_format_number(best_profit_row['Profit'])}).",
                ])

            lines.extend([
                "",
                "Values are aggregated from the dataset using Pandas.",
            ])

            return "\n".join(lines)

    # -----------------------------------------------------
    # MULTI-METRIC GROUP COMPARISON
    # Example: "Compare sales and profit by region."
    # -----------------------------------------------------

    requested_metrics = _requested_metrics(df, q)

    explicitly_compares_metrics = (
        len(requested_metrics) >= 2
        and (
            re.search(r"\b(compare|comparison|versus|vs)\b", q)
            or re.search(r"\bby\b", q)
            or re.search(r"\bacross\b", q)
        )
    )

    if explicitly_compares_metrics:
        group_col = _find_group_column(df, q)

        if group_col is not None:
            grouped = _group_metrics(df, group_col, requested_metrics)

            if not grouped.empty:
                lines = [
                    f"## Metric Comparison by {group_col}",
                    "",
                    _format_table(grouped),
                ]

                sales_name = next(
                    (name.title() for name, _ in requested_metrics
                     if name == "sales"),
                    None,
                )
                profit_name = next(
                    (name.title() for name, _ in requested_metrics
                     if name == "profit"),
                    None,
                )

                if sales_name and profit_name:
                    best_sales = grouped.loc[grouped[sales_name].idxmax()]
                    best_profit = grouped.loc[grouped[profit_name].idxmax()]

                    lines.extend([
                        "",
                        "### Key Findings",
                        "",
                        f"- **Highest sales:** {best_sales[group_col]} "
                        f"({_format_number(best_sales[sales_name])}).",
                        f"- **Highest profit:** {best_profit[group_col]} "
                        f"({_format_number(best_profit[profit_name])}).",
                    ])

                lines.extend([
                    "",
                    "These are descriptive comparisons, not proof of "
                    "what caused the differences.",
                ])

                return "\n".join(lines)

    # -----------------------------------------------------
    # PROFIT MARGIN
    # -----------------------------------------------------

    margin_requested = bool(
        re.search(r"\b(profit margin|margin|profitability)\b", q)
    )

    if margin_requested:
        if sales_col is None or profit_col is None:
            return (
                "I couldn't calculate profit margin because the dataset "
                "needs recognizable Sales/Revenue and Profit columns."
            )

        sales_values = _to_numeric(df, sales_col)
        profit_values = _to_numeric(df, profit_col)

        total_sales = sales_values.sum()
        total_profit = profit_values.sum()

        if sales_values.notna().sum() == 0 or profit_values.notna().sum() == 0:
            return "Profit margin cannot be calculated because one of the required columns has no valid numeric values."

        if total_sales == 0:
            return "Profit margin cannot be calculated because total sales are zero."

        margin = total_profit / total_sales * 100

        return (
            "## Overall Profit Margin\n\n"
            f"- **Total profit:** {_format_number(total_profit)}\n"
            f"- **Total sales:** {_format_number(total_sales)}\n"
            f"- **Profit margin:** {margin:.2f}%\n\n"
            "**Formula:** Total profit ÷ Total sales × 100.\n\n"
            "Calculated locally using Pandas."
        )

    # -----------------------------------------------------
    # TOTAL SALES, PROFIT, OR QUANTITY
    # -----------------------------------------------------

    metric_col, metric = _find_metric_column(df, q)

    total_requested = (
        any(word in q.split() for word in [
            "total", "overall", "sum", "combined",
        ])
        or "how much sales" in q
        or "how much profit" in q
        or "sales generated" in q
        or "what is the sales" in q
        or "what is the profit" in q
        or "how many units" in q
    )

    if total_requested and metric_col is not None:
        values = _to_numeric(df, metric_col)
        total = values.sum()

        return (
            f"## Total {metric.title()}\n\n"
            f"**{metric.title()}: {_format_number(total)}**\n\n"
            f"Calculated locally from the `{metric_col}` column."
        )

    # -----------------------------------------------------
    # TOP N / HIGHEST / LOWEST / BEST / WORST
    # -----------------------------------------------------

    ranking_requested = any(word in q for word in [
        "highest", "lowest", "maximum", "minimum",
        "most", "least", "best", "worst", "top", "rank",
    ])

    if ranking_requested and metric_col is not None:
        group_col = _find_group_column(df, q)

        if group_col is not None:
            temp = df[[group_col, metric_col]].copy()
            temp[metric_col] = _numeric_series(temp[metric_col])
            temp = temp.dropna(subset=[group_col, metric_col])

            if not temp.empty:
                grouped = temp.groupby(group_col)[metric_col].sum()

                lowest_requested = any(word in q for word in [
                    "lowest", "minimum", "least", "worst",
                ])

                grouped = grouped.sort_values(ascending=lowest_requested)

                if re.search(r"\btop\s+\d+\b", q) or re.search(r"\brank\b", q):
                    match = re.search(r"\btop\s+(\d+)\b", q)
                    limit = int(match.group(1)) if match else 5
                    limit = max(1, min(limit, 50))

                    result = grouped.head(limit).reset_index()
                    result.columns = [group_col, metric.title()]
                    result.insert(0, "Rank", range(1, len(result) + 1))

                    return (
                        f"## Top {len(result)} by {metric.title()}\n\n"
                        f"{_format_table(result)}"
                    )

                name = grouped.index[0]
                value = grouped.iloc[0]
                label = "lowest" if lowest_requested else "highest"

                comparison = (
                    grouped.rename(metric.title())
                    .reset_index()
                )

                return (
                    f"## {label.title()} {metric.title()} by {group_col}\n\n"
                    f"**{name}** has the {label} {metric}: "
                    f"**{_format_number(value)}**.\n\n"
                    "### Full Comparison\n\n"
                    f"{_format_table(comparison)}"
                )

    # -----------------------------------------------------
    # GROUPED SINGLE-METRIC COMPARISON
    # -----------------------------------------------------

    grouped_requested = (
        re.search(r"\bby\b", q) is not None
        or any(phrase in q for phrase in [
            "breakdown", "compare", "comparison",
            "distribution", "across each",
        ])
    )

    if grouped_requested and metric_col is not None:
        group_col = _find_group_column(df, q)

        if group_col is not None:
            temp = df[[group_col, metric_col]].copy()
            temp[metric_col] = _numeric_series(temp[metric_col])
            temp = temp.dropna(subset=[group_col, metric_col])

            grouped = (
                temp.groupby(group_col)[metric_col]
                .sum()
                .sort_values(ascending=False)
                .rename(metric.title())
                .reset_index()
            )

            return (
                f"## {metric.title()} by {group_col}\n\n"
                f"{_format_table(grouped)}"
            )

    # -----------------------------------------------------
    # BASIC NUMERIC STATISTICS
    # -----------------------------------------------------

    if any(word in q for word in [
        "average", "mean", "median", "minimum",
        "maximum", "statistics",
    ]) and metric_col is not None:
        values = _to_numeric(df, metric_col).dropna()

        if not values.empty:
            return (
                f"## {metric.title()} Statistics\n\n"
                f"- **Mean:** {_format_number(values.mean())}\n"
                f"- **Median:** {_format_number(values.median())}\n"
                f"- **Minimum:** {_format_number(values.min())}\n"
                f"- **Maximum:** {_format_number(values.max())}\n"
                f"- **Valid values:** {len(values):,}"
            )

    # Unsupported or ambiguous questions use Gemini.
    return None


# =========================================================
# CHART DATA HELPERS
# =========================================================

def _build_grouped_chart(df, group_col, metric_col):
    temp = df[[group_col, metric_col]].copy()
    temp[metric_col] = _numeric_series(temp[metric_col])
    temp = temp.dropna(subset=[group_col, metric_col])

    grouped = (
        temp.groupby(group_col)[metric_col]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        "labels": [str(value) for value in grouped.index],
        "values": [float(value) for value in grouped.values],
        "group_column": group_col,
        "metric_column": metric_col,
    }


def _build_monthly_chart(df):
    date_col = _find_date_column(df)
    metrics = _get_metric_columns(df)
    sales_col = metrics["sales"]
    profit_col = metrics["profit"]

    if date_col is None:
        return None

    temp = df.copy()
    temp["_Date"] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=["_Date"])

    if temp.empty:
        return None

    temp["_Month"] = temp["_Date"].dt.to_period("M").astype(str)

    metric_columns = [
        col for col in [sales_col, profit_col] if col is not None
    ]

    if not metric_columns:
        return None

    for col in metric_columns:
        temp[col] = _numeric_series(temp[col])

    monthly = temp.groupby("_Month")[metric_columns].sum().reset_index()

    return {
        "labels": monthly["_Month"].astype(str).tolist(),
        "series": {
            col: monthly[col].astype(float).tolist()
            for col in metric_columns
        },
    }


def get_visualization_data(file_path, user_question):
    """Return chart data without calling Gemini."""
    df = load_dataset(file_path)
    q = user_question.lower()

    if any(word in q for word in [
        "monthly", "month-wise", "month wise",
        "trend", "over time", "by month",
    ]):
        result = _build_monthly_chart(df)
        if result is not None:
            return result

    metric_col, _ = _find_metric_column(df, q)
    group_col = _find_group_column(df, q)

    if metric_col is not None and group_col is not None:
        return _build_grouped_chart(df, group_col, metric_col)

    return None


# =========================================================
# GEMINI FALLBACK
# =========================================================

def _build_dataset_context(df):
    lines = [
        f"Rows: {len(df):,}",
        f"Columns: {list(df.columns)}",
        "",
        "Data types:",
        df.dtypes.astype(str).to_string(),
        "",
        "Missing values:",
        df.isna().sum().to_string(),
        "",
        "Sample records:",
        df.head(10).to_string(index=False),
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        lines.extend([
            "",
            "Numeric statistics:",
            df[numeric_cols].describe().to_string(),
        ])

    return "\n".join(lines)


def safe_gemini_response(client, question, df):
    prompt = f"""
You are an AI Data Analyst Agent.

Answer the user's question using only the dataset context below.

Rules:
- Never invent data, totals, rankings, correlations, or calculations.
- If the user asks for calculations that cannot be reliably derived
  from this context, say what additional computation is required.
- Distinguish observed patterns from hypotheses.
- Do not claim that one factor caused another without evidence.
- For "why" questions, describe possible explanations as hypotheses,
  and clearly state that the dataset alone may not establish causality.
- Use clear Markdown headings, lists, and tables when useful.
- State clearly when the context is insufficient.

DATASET CONTEXT:
{_build_dataset_context(df)}

USER QUESTION:
{question}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = getattr(response, "text", None)

        if answer and answer.strip():
            return answer.strip()

        return "Gemini returned an empty response. Please rephrase your question."

    except Exception as exc:
        error = str(exc)
        lower = error.lower()

        if any(term in lower for term in [
            "quota exceeded", "resource_exhausted",
            "resource exhausted", "rate limit", "429",
        ]):
            return (
                "### Gemini API quota or rate limit reached\n\n"
                "Gemini could not complete this question because the "
                "API quota or rate limit appears to have been reached. "
                "Check your API usage or retry after the limit resets."
            )

        if any(term in lower for term in [
            "api key not valid", "invalid api key",
            "unauthenticated", "permission denied",
        ]):
            return (
                "### Gemini API authentication error\n\n"
                "Check your local `.env` file, API key validity, and "
                "model access. Do not share your API key."
            )

        if "404" in lower or "not found" in lower:
            return (
                "### Gemini model error\n\n"
                f"Check whether `{GEMINI_MODEL}` is available to your API key."
            )

        print(f"Gemini API error: {error}")

        return (
            "### Gemini API error\n\n"
            "The request could not be completed. Check the terminal "
            "for the detailed error message."
        )


# =========================================================
# MAIN ENTRY POINT
# =========================================================

def analyze_dataset(file_path, user_question=None):
    """Answer supported questions locally; use Gemini otherwise."""
    try:
        df = load_dataset(file_path)
    except Exception as exc:
        return f"Could not load the dataset: {exc}"

    if df.empty:
        return "The uploaded dataset is empty."

    question = (user_question or "").strip()

    if not question:
        return "Please enter a question about your dataset."

    try:
        local_answer = answer_local_question(df, question)

        if local_answer is not None:
            return local_answer

    except Exception as exc:
        print(f"Local analysis error: {exc}")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        return (
            "### Gemini API key is missing\n\n"
            "Add your API key to the `.env` file in the project root:\n\n"
            "```text\n"
            "GEMINI_API_KEY=your_actual_api_key_here\n"
            "```\n\n"
            "Keep your real API key private."
        )

    try:
        client = genai.Client(api_key=api_key.strip())
    except Exception as exc:
        print(f"Gemini client initialization error: {exc}")
        return "Could not initialize Gemini. Check your SDK and environment."

    return safe_gemini_response(client, question, df)

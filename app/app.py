import os

import re

import tempfile

from pathlib import Path



import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st

from dotenv import load_dotenv



import sys

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent



if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



from dotenv import load_dotenv

from agents.supervisor import run_supervisor



load_dotenv()



# ---------------------------------------------------------

# PAGE CONFIGURATION

# ---------------------------------------------------------



st.set_page_config(

    page_title="AI Data Analyst Agent",

    page_icon="📊",

    layout="wide",

    initial_sidebar_state="expanded",

)



st.title("📊 AI Data Analyst Agent")

st.caption(

    "Interactive business analytics powered by Pandas, Plotly, and Gemini."

)



# ---------------------------------------------------------

# COLUMN DETECTION

# ---------------------------------------------------------



ALIASES = {

    "date": [

        "date", "order date", "transaction date", "created at",

        "datetime", "month", "timestamp",

    ],

    "sales": [

        "sales", "revenue", "total sales", "total revenue",

        "turnover", "amount", "sales amount",

    ],

    "profit": [

        "profit", "earnings", "net profit", "total profit",

        "profit amount",

    ],

    "quantity": [

        "quantity", "qty", "units", "units sold",

        "total quantity",

    ],

    "region": [

        "region", "area", "territory", "location", "state",

    ],

    "product": [

        "product", "product name", "item", "category",

        "product category",

    ],

}





def normalize_name(value):

    """Normalize column names for reliable matching."""

    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()





def detect_columns(df):

    """Automatically identify common business data columns."""

    result = {}



    normalized_columns = {

        normalize_name(column): column

        for column in df.columns

    }



    for role, aliases in ALIASES.items():

        candidates = [normalize_name(alias) for alias in aliases]



        match = None



        for candidate in candidates:

            if candidate in normalized_columns:

                match = normalized_columns[candidate]

                break



        if match is None:

            for normalized, original in normalized_columns.items():

                if any(

                    candidate in normalized

                    for candidate in candidates

                    if candidate

                ):

                    match = original

                    break



        result[role] = match



    return result





def clean_numeric(series):

    """Convert numeric-looking strings and currency values to numbers."""

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(series, errors="coerce")



    cleaned = (

        series.astype(str)

        .str.replace(",", "", regex=False)

        .str.replace("$", "", regex=False)

        .str.replace("₹", "", regex=False)

        .str.replace("£", "", regex=False)

        .str.replace("€", "", regex=False)

        .str.replace("%", "", regex=False)

        .str.strip()

    )



    return pd.to_numeric(cleaned, errors="coerce")





def parse_dates(series):

    """

    Parse dates consistently.



    Ambiguous slash-separated dates such as 1/5/2026 are treated

    as month/day/year, matching the sample dashboard.

    """

    if pd.api.types.is_datetime64_any_dtype(series):

        return pd.to_datetime(series, errors="coerce")



    return pd.to_datetime(

        series,

        errors="coerce",

        format="mixed",

        dayfirst=False,

    )





def prepare_dataset(df, mapping):

    """Prepare a clean working copy without modifying the original."""

    data = df.copy()



    for role in ("sales", "profit", "quantity"):

        column = mapping.get(role)

        if column and column in data.columns:

            data[column] = clean_numeric(data[column])



    date_column = mapping.get("date")

    if date_column and date_column in data.columns:

        data[date_column] = parse_dates(data[date_column])



    return data





def money(value):

    """Format numeric values for display."""

    if pd.isna(value):

        return "N/A"



    return f"{value:,.0f}"







def monthly_aggregate(data, date_column, metrics):

    """

    Aggregate valid numeric values by month.



    Months with no valid values for a metric return NaN,

    not an incorrect zero.

    """

    if not date_column or date_column not in data.columns:

        return pd.DataFrame()



    valid = data.dropna(subset=[date_column]).copy()



    if valid.empty:

        return pd.DataFrame()



    valid["_month"] = (

        valid[date_column]

        .dt.to_period("M")

        .astype(str)

    )



    valid_metrics = [

        column

        for column in metrics

        if column and column in valid.columns

    ]



    if not valid_metrics:

        return pd.DataFrame()



    for column in valid_metrics:

        valid[column] = pd.to_numeric(

            valid[column],

            errors="coerce",

        )



    result = (

        valid.groupby("_month")[valid_metrics]

        .sum(min_count=1)

        .reset_index()

        .sort_values("_month")

        .rename(columns={"_month": "Month"})

        .reset_index(drop=True)

    )



    return result



# ---------------------------------------------------------

# DATA LOADING

# ---------------------------------------------------------



st.sidebar.header("📂 Dataset")



uploaded_file = st.sidebar.file_uploader(

    "Upload CSV or Excel",

    type=["csv", "xlsx", "xls"],

)



if uploaded_file is not None:

    try:

        suffix = Path(uploaded_file.name).suffix.lower()



        if suffix == ".csv":

            original_df = pd.read_csv(uploaded_file)

        else:

            original_df = pd.read_excel(uploaded_file)



        source_name = uploaded_file.name



    except Exception as exc:

        st.error(f"Could not load the uploaded dataset: {exc}")

        st.stop()



else:

    sample_path = (

        Path(__file__).resolve().parents[1]

        / "data"

        / "sample"

        / "sample.csv"

    )



    if not sample_path.exists():

        st.warning(

            "Upload a CSV or Excel file, or place sample.csv in "

            "data/sample/sample.csv."

        )

        st.stop()



    original_df = pd.read_csv(sample_path)

    source_name = sample_path.name



if original_df.empty:

    st.warning("The dataset contains no rows.")

    st.stop()



if len(original_df.columns) == 0:

    st.warning("The dataset contains no columns.")

    st.stop()



st.sidebar.success(f"Loaded: {source_name}")



# ---------------------------------------------------------


# ---------------------------------------------------------
# DATA CLEANING STUDIO (APPROVED WORKING COPY)
# ---------------------------------------------------------
from data_cleaning_studio import render_data_cleaning_studio

original_df = render_data_cleaning_studio(original_df, source_name)

# COLUMN SETTINGS

# ---------------------------------------------------------



st.sidebar.header("⚙️ Column Settings")



detected = detect_columns(original_df)

column_options = ["None"] + list(original_df.columns)





def column_selector(label, role):

    current = detected.get(role)



    default_index = (

        list(original_df.columns).index(current) + 1

        if current in original_df.columns

        else 0

    )



    selected = st.sidebar.selectbox(

        label,

        options=column_options,

        index=default_index,

        key=f"column_{role}",

    )



    return None if selected == "None" else selected





mapping = {

    "date": column_selector("Date", "date"),

    "region": column_selector("Region", "region"),

    "product": column_selector("Product / Category", "product"),

    "sales": column_selector("Sales / Revenue", "sales"),

    "profit": column_selector("Profit", "profit"),

    "quantity": column_selector("Quantity", "quantity"),

}



data = prepare_dataset(original_df, mapping)



# ---------------------------------------------------------

# FILTERS

# ---------------------------------------------------------



st.sidebar.header("🔎 Filters")



filtered = data.copy()



date_column = mapping["date"]



if date_column and date_column in filtered.columns:

    valid_dates = filtered[date_column].dropna()



    if not valid_dates.empty:

        min_date = valid_dates.min().date()

        max_date = valid_dates.max().date()



        date_range = st.sidebar.date_input(

            "Date range",

            value=(min_date, max_date),

            min_value=min_date,

            max_value=max_date,

            key="date_filter",

        )



        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

            start_date, end_date = date_range



            filtered = filtered[

                filtered[date_column].notna()

                & (filtered[date_column].dt.date >= start_date)

                & (filtered[date_column].dt.date <= end_date)

            ]



for role in ("region", "product"):

    column = mapping.get(role)



    if column and column in filtered.columns:

        choices = sorted(

            filtered[column].dropna().unique().tolist(),

            key=lambda item: str(item),

        )



        selected = st.sidebar.multiselect(

            f"{role.title()}",

            options=choices,

            default=choices,

            key=f"filter_{role}",

        )



        filtered = filtered[filtered[column].isin(selected)]



st.caption(

    f"Dataset loaded: **{source_name}** · "

    f"Showing **{len(filtered):,} of {len(original_df):,} rows** "

    "after filtering."

)



if filtered.empty:

    st.warning("No rows match the selected filters.")

    st.stop()



# ---------------------------------------------------------

# DATASET PREVIEW

# ---------------------------------------------------------



with st.expander("📋 Dataset Preview", expanded=False):

    st.dataframe(filtered, use_container_width=True, hide_index=True)



    col_a, col_b, col_c = st.columns(3)

    col_a.metric("Rows", f"{len(filtered):,}")

    col_b.metric("Columns", f"{len(filtered.columns):,}")

    col_c.metric(

        "Duplicate rows",

        f"{int(filtered.duplicated().sum()):,}",

    )



# ---------------------------------------------------------

# KPI CARDS

# ---------------------------------------------------------



st.subheader("📌 Key Performance Indicators")



sales_col = mapping["sales"]

profit_col = mapping["profit"]

quantity_col = mapping["quantity"]



def safe_total(dataframe, column):

    """

    Return None when the column is unavailable or

    contains no valid numeric values.

    """

    if not column or column not in dataframe.columns:

        return None



    numeric_values = pd.to_numeric(

        dataframe[column],

        errors="coerce",

    )



    if numeric_values.notna().sum() == 0:

        return None



    return numeric_values.sum(min_count=1)





total_sales = safe_total(filtered, sales_col)

total_profit = safe_total(filtered, profit_col)

total_quantity = safe_total(filtered, quantity_col)





for role, column in (

    ("Sales", sales_col),

    ("Profit", profit_col),

    ("Quantity", quantity_col),

):

    if column and column in filtered.columns:

        missing_count = int(filtered[column].isna().sum())



        if missing_count > 0:

            st.warning(

                f"{role}: {missing_count:,} of "

                f"{len(filtered):,} records have missing "

                "or invalid numeric values. "

                "Aggregations exclude these values."

            )





if (

    total_sales is not None

    and total_profit is not None

    and pd.notna(total_sales)

    and total_sales != 0

):

    profit_margin = total_profit / total_sales * 100

else:

    profit_margin = None



k1, k2, k3, k4 = st.columns(4)



k1.metric(

    "Total Sales / Revenue",

    money(total_sales) if total_sales is not None else "N/A",

)



k2.metric(

    "Total Profit",

    money(total_profit) if total_profit is not None else "N/A",

)



k3.metric(

    "Profit Margin",

    f"{profit_margin:.2f}%" if profit_margin is not None else "N/A",

)



k4.metric(

    "Total Quantity",

    money(total_quantity) if total_quantity is not None else "N/A",

)



# ---------------------------------------------------------

# SHARED AGGREGATIONS

# ---------------------------------------------------------



monthly_metrics = [

    column for column in (sales_col, profit_col)

    if column and column in filtered.columns

]



monthly_df = monthly_aggregate(

    filtered,

    date_column,

    monthly_metrics,

)







def group_aggregate(dataframe, group_column, metric_column):

    """

    Aggregate metrics by category without treating

    entirely missing groups as zero.

    """

    if (

        not group_column

        or not metric_column

        or group_column not in dataframe.columns

        or metric_column not in dataframe.columns

    ):

        return pd.DataFrame()



    result = (

        dataframe.groupby(

            group_column,

            dropna=False,

        )[metric_column]

        .sum(min_count=1)

        .sort_values(ascending=False)

        .rename("Value")

        .reset_index()

    )



    return result







def best_group(dataframe, group_column, metric_column, highest=True):

    """

    Return the highest or lowest category with valid data.

    """

    result = group_aggregate(

        dataframe,

        group_column,

        metric_column,

    )



    if result.empty:

        return None, None



    result = result.dropna(subset=["Value"])



    if result.empty:

        return None, None



    if highest:

        row = result.loc[result["Value"].idxmax()]

    else:

        row = result.loc[result["Value"].idxmin()]



    return row[group_column], row["Value"]





# ---------------------------------------------------------

# AUTOMATIC BUSINESS INSIGHTS

# ---------------------------------------------------------



st.subheader("💡 Business Insights & Executive Summary")



insights_tab, executive_tab = st.tabs(

    ["Automatic Insights", "AI Executive Summary"]

)



insights = []



if total_sales is not None:

    insights.append(

        f"Total sales: {money(total_sales)} across {len(filtered):,} rows."

    )



if total_profit is not None:

    insights.append(f"Total profit: {money(total_profit)}.")



if profit_margin is not None:

    insights.append(f"Profit margin: {profit_margin:.2f}%.")



if total_quantity is not None:

    insights.append(f"Total quantity: {money(total_quantity)}.")



product_col = mapping["product"]

region_col = mapping["region"]



if product_col and profit_col:

    name, value = best_group(filtered, product_col, profit_col, True)

    if name is not None:

        insights.append(

            f"Top product by profit: {name} ({money(value)})."

        )



    name, value = best_group(filtered, product_col, profit_col, False)

    if name is not None:

        insights.append(

            f"Lowest product by profit: {name} ({money(value)})."

        )



if region_col and sales_col:

    name, value = best_group(filtered, region_col, sales_col, True)

    if name is not None:

        insights.append(

            f"Leading region by sales: {name} ({money(value)})."

        )



# Only rank months with actual numeric observations.

for metric_column, metric_name in (

    (sales_col, "sales"),

    (profit_col, "profit"),

):

    if not metric_column or monthly_df.empty or metric_column not in monthly_df.columns:

        continue



    valid_months = monthly_df.dropna(subset=[metric_column])

    if valid_months.empty:

        continue



    best_row = valid_months.loc[valid_months[metric_column].idxmax()]

    worst_row = valid_months.loc[valid_months[metric_column].idxmin()]



    insights.append(

        f"Strongest month by {metric_name}: {best_row['Month']} "

        f"({money(best_row[metric_column])})."

    )

    insights.append(

        f"Lowest month by {metric_name}: {worst_row['Month']} "

        f"({money(worst_row[metric_column])})."

    )



with insights_tab:

    st.caption(

        "These findings are calculated locally with Pandas "

        "from your currently filtered data."

    )



    for item in insights:

        st.markdown(f"- {item}")



    st.markdown("#### Data-backed comparisons")



    left, right = st.columns(2)



    with left:

        if product_col and profit_col:

            product_profit = group_aggregate(

                filtered, product_col, profit_col

            )



            if not product_profit.dropna(subset=["Value"]).empty:

                product_profit = product_profit.dropna(subset=["Value"])

                fig = px.bar(

                    product_profit,

                    x=product_col,

                    y="Value",

                    title="Profit by Product",

                    labels={"Value": "Profit"},

                )

                st.plotly_chart(fig, use_container_width=True)



    with right:

        if region_col and sales_col:

            region_sales = group_aggregate(

                filtered, region_col, sales_col

            )



            if not region_sales.dropna(subset=["Value"]).empty:

                region_sales = region_sales.dropna(subset=["Value"])

                fig = px.bar(

                    region_sales,

                    x=region_col,

                    y="Value",

                    title="Sales by Region",

                    labels={"Value": "Sales"},

                )

                st.plotly_chart(fig, use_container_width=True)



    if not monthly_df.empty:

        st.markdown("#### Monthly Sales and Profit")



        monthly_long = monthly_df.melt(

            id_vars="Month",

            value_vars=monthly_metrics,

            var_name="Metric",

            value_name="Amount",

        )



        monthly_long = monthly_long.dropna(subset=["Amount"])

        if monthly_long.empty:

            st.info("No valid monthly numeric values are available to plot.")

        else:

            fig = px.line(

                monthly_long,

            x="Month",

            y="Amount",

            color="Metric",

            markers=True,

            title="Monthly Sales and Profit",

        )



            st.plotly_chart(fig, use_container_width=True)



with executive_tab:

    st.markdown("#### Executive Summary")



    for item in insights:

        st.markdown(f"- {item}")



    st.info(

        "The metrics above are calculated from the filtered dataset. "

        "The AI-generated narrative, when available, is an explanation "

        "of these figures—not a replacement for the calculations."

    )



    if st.button("Generate AI Executive Summary", key="generate_summary"):

        api_key = os.getenv("GEMINI_API_KEY")



        if not api_key:

            st.warning(

                "GEMINI_API_KEY was not found. Add it to your .env file."

            )

        else:

            try:

                from google import genai



                client = genai.Client(api_key=api_key)



                prompt = (

                    "You are a business data analyst. Write a concise "

                    "executive summary using only these calculated facts. "

                    "Do not invent numbers, causes, or recommendations.\n\n"

                    + "\n".join(insights)

                )



                with st.spinner("Generating executive summary..."):

                    response = client.models.generate_content(

                        model=os.getenv(

                            "GEMINI_MODEL", "gemini-3-flash-preview"

                        ),

                        contents=prompt,

                    )



                st.markdown(response.text or "No summary was returned.")



            except Exception as exc:

                st.error(f"Could not generate the AI summary: {exc}")



# ---------------------------------------------------------

# INTERACTIVE ANALYTICS

# ---------------------------------------------------------



st.subheader("📈 Interactive Analytics")



category_tab, comparison_tab, trends_tab, quality_tab = st.tabs(

    [

        "Category Analysis",

        "Metric Comparison",

        "Time Trends",

        "Data Quality",

    ]

)



with category_tab:

    categorical_columns = filtered.select_dtypes(

        include=["object", "category", "string"]

    ).columns.tolist()



    if not categorical_columns:

        st.info("No categorical columns are available.")

    else:

        default_group = (

            product_col if product_col in categorical_columns

            else categorical_columns[0]

        )



        group_column = st.selectbox(

            "Group by",

            categorical_columns,

            index=categorical_columns.index(default_group),

            key="category_group",

        )



        numeric_columns = filtered.select_dtypes(

            include="number"

        ).columns.tolist()



        if not numeric_columns:

            st.info("No numeric columns are available.")

        else:

            default_metric = (

                sales_col if sales_col in numeric_columns

                else numeric_columns[0]

            )



            metric_column = st.selectbox(

                "Metric",

                numeric_columns,

                index=numeric_columns.index(default_metric),

                key="category_metric",

            )



            chart_type = st.selectbox(

                "Chart type",

                ["Auto", "Bar", "Line", "Pie"],

                key="category_chart_type",

            )



            aggregation = st.selectbox(

                "Aggregation",

                ["Sum", "Mean", "Count", "Min", "Max"],

                key="category_aggregation",

            )



            top_n = st.slider(

                "Number of groups to show",

                min_value=1,

                max_value=30,

                value=10,

                key="category_top_n",

            )



            agg_functions = {

                "Sum": "sum",

                "Mean": "mean",

                "Count": "count",

                "Min": "min",

                "Max": "max",

            }



            grouped_series = filtered.groupby(group_column, dropna=False)[metric_column]

            if aggregation == "Sum":

                grouped_series = grouped_series.sum(min_count=1)

            else:

                grouped_series = grouped_series.agg(agg_functions[aggregation])

            grouped = (

                grouped_series.dropna()

                .sort_values(ascending=False)

                .head(top_n)

                .reset_index()

            )



            st.caption(

                f"Showing {len(grouped)} groups using "

                f"{aggregation.lower()}."

            )



            if not grouped.empty:

                if chart_type == "Pie":

                    fig = px.pie(

                        grouped,

                        names=group_column,

                        values=metric_column,

                        title=f"{aggregation} of {metric_column} by "

                              f"{group_column}",

                    )

                elif chart_type == "Line":

                    fig = px.line(

                        grouped,

                        x=group_column,

                        y=metric_column,

                        markers=True,

                    )

                else:

                    fig = px.bar(

                        grouped,

                        x=group_column,

                        y=metric_column,

                        title=f"{aggregation} of {metric_column} by "

                              f"{group_column}",

                    )



                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(grouped, use_container_width=True, hide_index=True)



with comparison_tab:

    numeric_columns = filtered.select_dtypes(

        include="number"

    ).columns.tolist()



    if len(numeric_columns) < 2:

        st.info("At least two numeric columns are required.")

    else:

        selected_metrics = st.multiselect(

            "Metrics to compare",

            numeric_columns,

            default=[

                col for col in (sales_col, profit_col)

                if col in numeric_columns

            ],

            key="comparison_metrics",

        )



        categorical_columns = filtered.select_dtypes(

            include=["object", "category", "string"]

        ).columns.tolist()



        if categorical_columns:

            compare_by = st.selectbox(

                "Compare by",

                categorical_columns,

                key="comparison_group",

            )



            if selected_metrics:

                comparison = (

                    filtered.groupby(compare_by, dropna=False)[selected_metrics]

                    .sum(min_count=1)

                    .reset_index()

                )



                st.dataframe(

                    comparison,

                    use_container_width=True,

                    hide_index=True,

                )



                comparison_long = comparison.melt(

                    id_vars=compare_by,

                    value_vars=selected_metrics,

                    var_name="Metric",

                    value_name="Value",

                ).dropna(subset=["Value"])



                if comparison_long.empty:

                    st.info("No valid values are available for comparison.")

                else:

                    fig = px.bar(

                        comparison_long,

                    x=compare_by,

                    y="Value",

                    color="Metric",

                    barmode="group",

                )



                    st.plotly_chart(fig, use_container_width=True)

        else:

            st.info("No categorical columns are available for grouping.")



with trends_tab:

    if date_column and date_column in filtered.columns:

        valid_trends = filtered.dropna(subset=[date_column]).copy()



        if valid_trends.empty:

            st.info("No valid dates are available for time trends.")

        else:

            available_trend_metrics = [

                col for col in (sales_col, profit_col, quantity_col)

                if col and col in valid_trends.columns

            ]



            if available_trend_metrics:

                trend_metric = st.selectbox(

                    "Trend metric",

                    available_trend_metrics,

                    key="trend_metric",

                )



                trend_data = (

                    valid_trends

                    .assign(

                        _month=valid_trends[date_column]

                        .dt.to_period("M").astype(str)

                    )

                    .groupby("_month", as_index=False)[trend_metric]

                    .sum(min_count=1)

                    .rename(columns={"_month": "Month"})

                    .sort_values("Month")

                )



                trend_plot = trend_data.dropna(subset=[trend_metric])

                if trend_plot.empty:

                    st.info("No valid numeric values are available for this trend.")

                else:

                    fig = px.line(

                        trend_plot,

                    x="Month",

                    y=trend_metric,

                    markers=True,

                    title=f"Monthly {trend_metric}",

                )



                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(

                    trend_data,

                    use_container_width=True,

                    hide_index=True,

                )

            else:

                st.info("No numeric metrics are available for trends.")

    else:

        st.info("Select or identify a date column to view time trends.")



with quality_tab:

    st.markdown("#### Data Quality Overview")



    quality_summary = pd.DataFrame({

        "Column": filtered.columns,

        "Data Type": [

            str(filtered[col].dtype) for col in filtered.columns

        ],

        "Missing Values": [

            int(filtered[col].isna().sum()) for col in filtered.columns

        ],

        "Missing %": [

            round(filtered[col].isna().mean() * 100, 2)

            for col in filtered.columns

        ],

        "Unique Values": [

            int(filtered[col].nunique(dropna=True))

            for col in filtered.columns

        ],

    })



    q1, q2, q3 = st.columns(3)



    q1.metric("Rows", f"{len(filtered):,}")

    q2.metric("Columns", f"{len(filtered.columns):,}")

    q3.metric("Duplicate Rows", f"{int(filtered.duplicated().sum()):,}")



    st.dataframe(

        quality_summary,

        use_container_width=True,

        hide_index=True,

    )



# ---------------------------------------------------------

# NATURAL-LANGUAGE Q&A

# ---------------------------------------------------------



def answer_question_locally(dataframe, question, column_mapping):

    """

    Answer common business questions directly from Pandas.



    Returns a string for locally answered questions, or None when

    Gemini fallback may be useful.

    """

    q = question.lower().strip()



    if not q:

        return "Please enter a question."



    sales = column_mapping.get("sales")

    profit = column_mapping.get("profit")

    quantity = column_mapping.get("quantity")

    date = column_mapping.get("date")

    region = column_mapping.get("region")

    product = column_mapping.get("product")



    # Handle profit margin.

    if "margin" in q:

        if (

            sales and profit

            and sales in dataframe.columns

            and profit in dataframe.columns

        ):

            sales_total = safe_total(dataframe, sales)



            profit_total = safe_total(dataframe, profit)

            if sales_total is None or profit_total is None or sales_total == 0:

                return "Profit margin is unavailable because valid sales or profit values are missing, or sales total is zero."



            margin = profit_total / sales_total * 100



            return (

                f"**Profit margin:** {margin:.2f}%\n\n"

                f"- Total sales: {money(sales_total)}\n"

                f"- Total profit: {money(safe_total(dataframe, profit))}"

            )



        return "I couldn't identify both sales and profit columns."



    # Handle monthly comparison questions.

    asks_monthly = bool(

        re.search(r"\b(month|monthly|month-wise|month wise)\b", q)

    )



    asks_compare = bool(

        re.search(r"\b(compare|comparison|versus|vs\.?)\b", q)

    )



    asks_sales = bool(

        re.search(r"\b(sales|revenue|turnover)\b", q)

    )



    asks_profit = bool(

        re.search(r"\b(profit|earnings)\b", q)

    )



    if asks_monthly and asks_compare and asks_sales and asks_profit:

        if not date or date not in dataframe.columns:

            return "I couldn't identify a date column for monthly comparison."



        monthly = monthly_aggregate(

            dataframe,

            date,

            [sales, profit],

        )



        if monthly.empty or not sales or not profit or sales not in monthly or profit not in monthly:

            return "No valid dated records are available for monthly comparison."

        if monthly[sales].notna().sum() == 0 or monthly[profit].notna().sum() == 0:

            return "Monthly sales/profit comparison is unavailable because one metric has no valid values."



        return (

            "**Monthly Sales and Profit**\n\n"

            + monthly.to_markdown(index=False, floatfmt=",.2f")

            + "\n\n"

            + f"**Highest sales:** "

            + f"{monthly.loc[monthly[sales].idxmax(), 'Month']} "

            + f"({money(monthly[sales].max())}).\n\n"

            + f"**Highest profit:** "

            + f"{monthly.loc[monthly[profit].idxmax(), 'Month']} "

            + f"({money(monthly[profit].max())})."

        )



    if asks_monthly and (asks_sales or asks_profit):

        metric = sales if asks_sales else profit



        if not date or date not in dataframe.columns:

            return "I couldn't identify a date column for monthly analysis."



        monthly = monthly_aggregate(dataframe, date, [metric])



        if monthly.empty or not metric or metric not in monthly or monthly[metric].notna().sum() == 0:

            return "No valid numeric observations are available for monthly analysis."



        if re.search(r"\b(lowest|minimum|worst|least)\b", q):

            row = monthly.loc[monthly[metric].idxmin()]

            description = "Lowest"

        else:

            row = monthly.loc[monthly[metric].idxmax()]

            description = "Highest"



        return (

            f"**{description} monthly {metric}:** {row['Month']} "

            f"({money(row[metric])}).\n\n"

            + monthly.to_markdown(index=False)

        )



    # Handle comparisons of two metrics by category.

    if asks_compare and asks_sales and asks_profit:

        group_column = None



        if region and region in dataframe.columns and "region" in q:

            group_column = region

        elif product and product in dataframe.columns and "product" in q:

            group_column = product

        elif region and region in dataframe.columns:

            group_column = region

        elif product and product in dataframe.columns:

            group_column = product



        if (

            group_column

            and sales and profit

            and sales in dataframe.columns

            and profit in dataframe.columns

        ):

            comparison = (

                dataframe.groupby(group_column, dropna=False)[[sales, profit]]

                .sum()

                .reset_index()

                .sort_values(sales, ascending=False)

            )



            highest_sales = comparison.loc[comparison[sales].idxmax()]

            highest_profit = comparison.loc[comparison[profit].idxmax()]



            return (

                f"**Metric Comparison by {group_column.title()}**\n\n"

                + comparison.to_markdown(index=False)

                + "\n\n"

                + f"**Highest sales:** {highest_sales[group_column]} "

                + f"({money(highest_sales[sales])}).\n\n"

                + f"**Highest profit:** {highest_profit[group_column]} "

                + f"({money(highest_profit[profit])})."

            )



    # Handle overall totals.

    if re.search(r"\b(total|overall|sum)\b", q):

        if asks_sales and sales and sales in dataframe.columns:

            return f"**Total sales:** {money(safe_total(dataframe, sales))}"



        if asks_profit and profit and profit in dataframe.columns:

            return f"**Total profit:** {money(safe_total(dataframe, profit))}"



        if quantity and quantity in dataframe.columns and re.search(

            r"\b(quantity|units|qty)\b", q

        ):

            return f"**Total quantity:** {money(safe_total(dataframe, quantity))}"



    # Handle highest / lowest product, region, or category.

    wants_lowest = bool(

        re.search(r"\b(lowest|least|minimum|worst)\b", q)

    )



    wants_highest = bool(

        re.search(r"\b(highest|top|best|maximum|most)\b", q)

    )



    metric = None



    if asks_profit and profit in dataframe.columns:

        metric = profit

    elif asks_sales and sales in dataframe.columns:

        metric = sales

    elif quantity and quantity in dataframe.columns and re.search(

        r"\b(quantity|units|qty)\b", q

    ):

        metric = quantity



    if metric and (wants_highest or wants_lowest):

        group_column = None



        if product and product in dataframe.columns and "product" in q:

            group_column = product

        elif region and region in dataframe.columns and "region" in q:

            group_column = region

        elif product and product in dataframe.columns:

            group_column = product

        elif region and region in dataframe.columns:

            group_column = region



        if group_column:

            grouped = group_aggregate(dataframe, group_column, metric)



            grouped = grouped.dropna(subset=["Value"])

            if not grouped.empty:

                if wants_lowest:

                    row = grouped.loc[grouped["Value"].idxmin()]

                    label = "Lowest"

                else:

                    row = grouped.loc[grouped["Value"].idxmax()]

                    label = "Highest"



                return (

                    f"**{label} {metric} by {group_column}:** "

                    f"{row[group_column]} ({money(row['Value'])}).\n\n"

                    + grouped.to_markdown(index=False)

                )



    # Handle generic dataset summary.

    if re.search(r"\b(summary|summarize|overview|describe)\b", q):

        parts = [

            f"- Rows: {len(dataframe):,}",

            f"- Columns: {len(dataframe.columns):,}",

        ]



        if sales and sales in dataframe.columns:

            parts.append(f"- Total sales: {money(safe_total(dataframe, sales))}")



        if profit and profit in dataframe.columns:

            parts.append(f"- Total profit: {money(safe_total(dataframe, profit))}")



        if quantity and quantity in dataframe.columns:

            parts.append(

                f"- Total quantity: {money(safe_total(dataframe, quantity))}"

            )



        return "**Dataset Summary**\n\n" + "\n".join(parts)



    return None





def ask_gemini(dataframe, question, column_mapping):

    """Use Gemini for questions that local Pandas rules do not cover."""

    api_key = os.getenv("GEMINI_API_KEY")



    if not api_key:

        return (

            "I couldn't match that question to a supported local analysis. "

            "For AI-assisted answers, add GEMINI_API_KEY to your .env file."

        )



    try:

        from google import genai



        client = genai.Client(api_key=api_key)



        # Send a compact dataset profile and a limited preview.

        preview = dataframe.head(30).copy()



        if len(preview.columns) > 20:

            preview = preview.iloc[:, :20]



        prompt = f"""

You are an AI data analyst.



Answer the user's question using the dataset information below.

Do not invent calculations or claim you inspected rows not included in the

preview. If the preview is insufficient to calculate an answer reliably,

say so and suggest a question that can be answered.



Dataset shape: {dataframe.shape}

Column mapping: {column_mapping}



Column types:

{dataframe.dtypes.astype(str).to_dict()}



Numeric summary:

{dataframe.describe(include="all").fillna("").to_string()}



First 30 rows:

{preview.to_string(index=False)}



User question:

{question}

"""



        response = client.models.generate_content(

            model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),

            contents=prompt,

        )



        return response.text or "Gemini did not return a response."



    except Exception as exc:

        return f"Gemini could not answer this question: {exc}"





# ---------------------------------------------------------

# CHAT INTERFACE — LANGGRAPH SUPERVISOR INTEGRATION

# ---------------------------------------------------------





st.subheader("💬 Ask Your Data Analyst")



st.caption(

    "Ask questions about your uploaded data. The LangGraph Supervisor "

    "routes requests to the appropriate agent."

)



if "messages" not in st.session_state:

    st.session_state.messages = []





def render_agent_details(message):

    """Render SQL, database result tables, and web research links."""



    # Agent routing information

    if message.get("route"):

        st.caption(

            f"Agent: {message.get('source', 'Supervisor')} "

            f"· Route: {str(message['route']).title()}"

        )



    # Display SQL generated by the Database Agent

    sql = message.get("sql")



    if isinstance(sql, str) and sql.strip():

        with st.expander("View SQL query", expanded=False):

            st.code(sql, language="sql")



            st.download_button(

                label="Download SQL",

                data=sql,

                file_name="database_query.sql",

                mime="text/plain",

                key=f"download_sql_{message.get('message_id', 'current')}",

            )



    # Display database query results

    database_results = message.get("database_results")



    if isinstance(database_results, dict):

        columns = database_results.get("columns", [])

        rows = database_results.get("rows", [])



        if rows:

            import pandas as pd



            st.markdown("**Database query results**")



            try:

                if isinstance(rows[0], dict):

                    results_df = pd.DataFrame(rows)

                else:

                    results_df = pd.DataFrame(

                        rows,

                        columns=columns if columns else None,

                    )



                st.dataframe(

                    results_df,

                    use_container_width=True,

                    hide_index=True,

                )



                row_count = database_results.get("row_count")



                if row_count is not None:

                    st.caption(f"Rows returned: {row_count}")



                if database_results.get("truncated"):

                    st.info(

                        "The result set was limited. "

                        "Not all matching rows may be displayed."

                    )



            except (ValueError, TypeError) as exc:

                st.warning(

                    f"Could not format the database results as a table: {exc}"

                )



        elif columns:

            st.info("The query ran, but no matching rows were returned.")



    # Display clickable web research results

    web_results = message.get("web_results")



    if isinstance(web_results, dict):

        web_results = web_results.get(

            "results",

            web_results.get("items", []),

        )



    if isinstance(web_results, list) and web_results:

        st.markdown("**Web research sources**")



        for index, item in enumerate(web_results, start=1):

            if not isinstance(item, dict):

                continue



            title = (

                item.get("title")

                or item.get("name")

                or f"Research result {index}"

            )



            url = (

                item.get("url")

                or item.get("link")

                or item.get("href")

            )



            snippet = (

                item.get("snippet")

                or item.get("body")

                or item.get("description")

            )



            if isinstance(url, str) and url.startswith(("https://", "http://")):

                st.markdown(f"{index}. [{title}]({url})")

            else:

                st.markdown(f"{index}. **{title}**")



            if isinstance(snippet, str) and snippet.strip():

                st.caption(snippet)



    # Surface errors from the web agent when available

    web_error = message.get("web_error")



    if web_error:

        st.warning(f"Web research notice: {web_error}")





suggestions = [

    "Which product has the highest profit?",

    "Compare sales and profit by region",

    "Compare monthly sales and profit and identify the best month",

    "What is the total revenue?",

    "Give me a dataset summary",

]



suggestion_columns = st.columns(len(suggestions))



for index, suggestion in enumerate(suggestions):

    with suggestion_columns[index]:

        if st.button(

            suggestion,

            key=f"suggestion_{index}",

            use_container_width=True,

        ):

            st.session_state.pending_question = suggestion





# Display previous messages, including their saved results.

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])



        if message["role"] == "assistant":

            render_agent_details(message)





pending_question = st.session_state.pop("pending_question", None)

typed_question = st.chat_input("Ask a question about your data...")



question = typed_question or pending_question



if question:

    st.session_state.messages.append(

        {

            "role": "user",

            "content": question,

        }

    )



    with st.chat_message("user"):

        st.markdown(question)



    # Defaults prevent missing metadata when an agent fails.

    answer = ""

    route = "error"

    source = "Error handler"

    result = {}



    with st.chat_message("assistant"):

        with st.spinner("Supervisor is routing your request..."):

            try:

                # First try deterministic Pandas analysis.

                # This uses the currently filtered dataset.

                local_answer = answer_question_locally(

                    filtered,

                    question,

                    mapping,

                )



                # Route the request through the Supervisor.

                result = run_supervisor(

                    question=question,

                    dataframe=filtered,

                    column_mapping=mapping,

                    local_answer=local_answer,

                )



                if not isinstance(result, dict):

                    raise TypeError(

                        "The Supervisor returned an unexpected result."

                    )



                answer = result.get(

                    "answer",

                    "The Supervisor did not return an answer.",

                )

                route = result.get("route", "file")

                source = result.get("source", "Supervisor")



            except Exception as exc:

                st.warning(

                    "The Supervisor encountered an error. "

                    "Trying the regular analysis fallback."

                )



                try:

                    answer = answer_question_locally(

                        filtered,

                        question,

                        mapping,

                    )



                    if answer is None:

                        answer = ask_gemini(

                            filtered,

                            question,

                            mapping,

                        )



                    route = "fallback"

                    source = "Local analysis / Gemini"



                except Exception as fallback_exc:

                    answer = (

                        "I couldn't complete the analysis.\n\n"

                        f"Supervisor error: {exc}\n\n"

                        f"Fallback error: {fallback_exc}"

                    )

                    route = "error"

                    source = "Error handler"



        st.markdown(str(answer))



        # Render additional output returned by the Supervisor.

        message_id = f"answer_{len(st.session_state.messages)}"



        assistant_message = {

            "role": "assistant",

            "content": str(answer),

            "route": route,

            "source": source,

            "sql": result.get("sql"),

            "database_results": result.get("database_results"),

            "web_results": result.get("web_results"),

            "web_error": result.get("web_error"),

            "message_id": message_id,

        }



        render_agent_details(assistant_message)



    # Save the full response so tables and links remain available

    # when Streamlit reruns the application.

    st.session_state.messages.append(assistant_message)





# Clear chat history

if st.session_state.messages:

    if st.button("Clear chat history", key="clear_chat"):

        st.session_state.messages = []

        st.rerun()

import pandas as pd


def dataset_summary(df: pd.DataFrame):
    """Return a basic summary of the dataset."""

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

    return summary


def total_sales(df: pd.DataFrame):
    """Calculate total sales."""

    return float(df["Sales"].sum())


def total_profit(df: pd.DataFrame):
    """Calculate total profit."""

    return float(df["Profit"].sum())


def total_quantity(df: pd.DataFrame):
    """Calculate total quantity sold."""

    return int(df["Quantity"].sum())


def profit_margin(df: pd.DataFrame):
    """Calculate overall profit margin."""

    sales = df["Sales"].sum()
    profit = df["Profit"].sum()

    if sales == 0:
        return 0.0

    return round((profit / sales) * 100, 2)


def sales_by_region(df: pd.DataFrame):
    """Calculate sales by region."""

    return (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )


def sales_by_product(df: pd.DataFrame):
    """Calculate sales by product."""

    return (
        df.groupby("Product")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )


def profit_by_region(df: pd.DataFrame):
    """Calculate profit by region."""

    return (
        df.groupby("Region")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )


def profit_by_product(df: pd.DataFrame):
    """Calculate profit by product."""

    return (
        df.groupby("Product")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )


def monthly_sales_profit(df: pd.DataFrame):
    """Calculate monthly sales and profit."""

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.dropna(subset=["Date"])

    monthly = (
        data.groupby(data["Date"].dt.to_period("M"))
        .agg({
            "Sales": "sum",
            "Profit": "sum"
        })
    )

    monthly.index = monthly.index.astype(str)

    return monthly.to_dict(orient="index")


# ============================================================
# GENERAL SAFE COLUMN ANALYSIS
# ============================================================

def analyze_column(
    df: pd.DataFrame,
    column: str,
    operation: str
):
    """
    Perform safe statistical operations on a dataset column.

    Allowed operations:
    sum, mean, min, max, count, median
    """

    if column not in df.columns:
        return {
            "error": f"Column '{column}' does not exist.",
            "available_columns": list(df.columns)
        }

    allowed_operations = {
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "median"
    }

    if operation not in allowed_operations:
        return {
            "error": f"Operation '{operation}' is not allowed.",
            "allowed_operations": sorted(allowed_operations)
        }

    series = df[column]

    if operation == "sum":
        result = series.sum()

    elif operation == "mean":
        result = series.mean()

    elif operation == "min":
        result = series.min()

    elif operation == "max":
        result = series.max()

    elif operation == "count":
        result = series.count()

    elif operation == "median":
        result = series.median()

    try:
        result = float(result)
    except (TypeError, ValueError):
        result = str(result)

    return {
        "column": column,
        "operation": operation,
        "result": result
    }

def chart_data(df: pd.DataFrame, column: str, metric: str):
    """
    Prepare grouped data for visualization.

    Supported metrics:
    sum, mean, count
    """

    if column not in df.columns:
        return {
            "error": f"Column '{column}' does not exist.",
            "available_columns": list(df.columns)
        }

    allowed_metrics = {
        "sum",
        "mean",
        "count"
    }

    if metric not in allowed_metrics:
        return {
            "error": f"Metric '{metric}' is not supported.",
            "allowed_metrics": sorted(allowed_metrics)
        }

    if metric == "sum":
        result = df.groupby(column).sum(numeric_only=True)

    elif metric == "mean":
        result = df.groupby(column).mean(numeric_only=True)

    elif metric == "count":
        result = df.groupby(column).count()

    return {
        "column": column,
        "metric": metric,
        "data": result.to_dict(orient="index")
    }
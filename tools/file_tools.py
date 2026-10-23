import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset.
    """

    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)

    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Please use CSV or Excel."
        )

    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a structured profile of the dataset.
    """

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in df.isnull().sum().items()
            if value > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
        "datetime_columns": df.select_dtypes(
            include=["datetime"]
        ).columns.tolist(),
    }

    return profile
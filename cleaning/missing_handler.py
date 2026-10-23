
import pandas as pd


class MissingValueHandler:
    """
    Detect and handle missing values using explicit rules.
    The original DataFrame is never modified.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        self.original = dataframe.copy(deep=True)
        self.df = dataframe.copy(deep=True)
        self.audit_log = []

    def missing_report(self):
        rows = []

        for column in self.df.columns:
            missing = int(self.df[column].isna().sum())

            rows.append({
                "column": str(column),
                "missing_count": missing,
                "missing_percent": round(
                    missing / len(self.df) * 100, 2
                ) if len(self.df) else 0.0,
                "dtype": str(self.df[column].dtype),
            })

        return pd.DataFrame(rows)

    def recommend_treatment(self):
        recommendations = []

        for column in self.df.columns:
            series = self.df[column]
            missing = int(series.isna().sum())

            if missing == 0:
                recommendation = "No action required"

            elif pd.api.types.is_numeric_dtype(series):
                recommendation = (
                    "Review missing values; consider median "
                    "imputation only if justified"
                )

            elif pd.api.types.is_datetime64_any_dtype(series):
                recommendation = (
                    "Investigate missing dates; avoid inventing dates"
                )

            else:
                recommendation = (
                    "Review source; consider explicit Unknown "
                    "category if appropriate"
                )

            recommendations.append({
                "column": str(column),
                "missing_count": missing,
                "recommendation": recommendation,
            })

        return pd.DataFrame(recommendations)

    def apply_treatment(
        self,
        column: str,
        strategy: str,
        fill_value=None,
    ):
        if column not in self.df.columns:
            raise ValueError(f"Column not found: {column}")

        supported = {
            "keep",
            "drop_rows",
            "mean",
            "median",
            "mode",
            "constant",
        }

        if strategy not in supported:
            raise ValueError(
                f"Unsupported strategy: {strategy}"
            )

        before = int(self.df[column].isna().sum())
        rows_before = len(self.df)

        if strategy == "keep":
            pass

        elif strategy == "drop_rows":
            self.df = self.df.dropna(
                subset=[column]
            ).copy()

        elif strategy in {"mean", "median"}:
            if not pd.api.types.is_numeric_dtype(
                self.df[column]
            ):
                raise TypeError(
                    f"{strategy} requires a numeric column."
                )

            values = self.df[column].dropna()

            if values.empty:
                raise ValueError(
                    "Cannot impute a column with no valid values."
                )

            replacement = (
                values.mean()
                if strategy == "mean"
                else values.median()
            )

            self.df[column] = self.df[column].fillna(
                replacement
            )

        elif strategy == "mode":
            modes = self.df[column].mode(dropna=True)

            if modes.empty:
                raise ValueError(
                    "Cannot calculate mode without valid values."
                )

            self.df[column] = self.df[column].fillna(
                modes.iloc[0]
            )

        elif strategy == "constant":
            if fill_value is None:
                raise ValueError(
                    "Provide a non-null fill_value."
                )

            self.df[column] = self.df[column].fillna(
                fill_value
            )

        after = int(self.df[column].isna().sum())

        self.audit_log.append({
            "column": column,
            "strategy": strategy,
            "missing_before": before,
            "missing_after": after,
            "rows_before": rows_before,
            "rows_after": len(self.df),
        })

        return self.df.copy(deep=True)

    def get_cleaned_data(self):
        return self.df.copy(deep=True)

    def get_audit_log(self):
        return pd.DataFrame(self.audit_log)

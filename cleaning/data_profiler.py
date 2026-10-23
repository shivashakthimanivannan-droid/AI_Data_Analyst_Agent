
import pandas as pd
import numpy as np


class DataProfiler:
    """Read-only data-quality profiling for Pandas DataFrames."""

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        self.df = dataframe.copy(deep=True)

    def dataset_overview(self):
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "total_missing_cells": int(self.df.isna().sum().sum()),
            "memory_usage_mb": round(
                self.df.memory_usage(deep=True).sum() / 1024**2, 3
            ),
        }

    def column_profile(self):
        profiles = []

        for column in self.df.columns:
            series = self.df[column]
            missing = int(series.isna().sum())

            profiles.append({
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_percent": round(
                    missing / len(self.df) * 100, 2
                ) if len(self.df) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [
                    str(value)
                    for value in series.dropna().unique()[:5]
                ],
            })

        return pd.DataFrame(profiles)

    def duplicate_report(self):
        if self.df.empty:
            return {
                "duplicate_count": 0,
                "duplicate_percent": 0.0,
            }

        count = int(self.df.duplicated().sum())

        return {
            "duplicate_count": count,
            "duplicate_percent": round(
                count / len(self.df) * 100, 2
            ),
        }

    def numeric_profile(self):
        numeric = self.df.select_dtypes(include="number")

        if numeric.empty:
            return pd.DataFrame()

        return numeric.describe().transpose()

    def outlier_report(self):
        results = []

        numeric = self.df.select_dtypes(include="number")

        for column in numeric.columns:
            values = numeric[column].dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = values[
                (values < lower) | (values > upper)
            ]

            results.append({
                "column": str(column),
                "outlier_count": len(outliers),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            })

        return pd.DataFrame(results)

    def generate_report(self):
        return {
            "overview": self.dataset_overview(),
            "columns": self.column_profile(),
            "duplicates": self.duplicate_report(),
            "numeric": self.numeric_profile(),
            "outliers": self.outlier_report(),
        }

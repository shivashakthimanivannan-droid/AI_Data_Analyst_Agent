
import numpy as np
import pandas as pd


class OutlierDetector:
    """
    Read-only statistical outlier detection.

    Supported methods:
      - IQR
      - Z-score
      - Modified Z-score (MAD)

    No records are automatically deleted or changed.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if not dataframe.columns.is_unique:
            raise ValueError("Column names must be unique.")

        self.df = dataframe.copy(deep=True)

    def _get_numeric(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Column not found: {column}")

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(
                f"Column '{column}' must be numeric."
            )

        if pd.api.types.is_bool_dtype(self.df[column]):
            raise TypeError(
                "Boolean columns cannot be analyzed as amounts."
            )

        values = pd.to_numeric(
            self.df[column], errors="coerce"
        ).astype("float64")

        return values.replace(
            [np.inf, -np.inf], np.nan
        )

    def detect_iqr(self, column, multiplier=1.5):
        if multiplier <= 0:
            raise ValueError("IQR multiplier must be positive.")

        values = self._get_numeric(column)
        valid = values.dropna()

        mask = pd.Series(
            False, index=self.df.index, dtype=bool
        )

        if len(valid) < 4:
            return mask, {
                "method": "IQR",
                "reason": "Insufficient observations",
            }

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return mask, {
                "method": "IQR",
                "reason": "Zero interquartile range",
            }

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask = ((values < lower) | (values > upper)).fillna(False)

        return mask, {
            "method": "IQR",
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "multiplier": multiplier,
        }

    def detect_zscore(self, column, threshold=3.0):
        if threshold <= 0:
            raise ValueError("Threshold must be positive.")

        values = self._get_numeric(column)
        valid = values.dropna()

        mask = pd.Series(
            False, index=self.df.index, dtype=bool
        )

        if len(valid) < 3:
            return mask, {
                "method": "Z-score",
                "reason": "Insufficient observations",
            }

        mean = valid.mean()
        std = valid.std(ddof=0)

        if std == 0 or not np.isfinite(std):
            return mask, {
                "method": "Z-score",
                "reason": "Zero or invalid standard deviation",
            }

        scores = (values - mean).abs() / std
        mask = (scores > threshold).fillna(False)

        return mask, {
            "method": "Z-score",
            "mean": float(mean),
            "std": float(std),
            "threshold": threshold,
        }

    def detect_modified_zscore(
        self, column, threshold=3.5
    ):
        if threshold <= 0:
            raise ValueError("Threshold must be positive.")

        values = self._get_numeric(column)
        valid = values.dropna()

        mask = pd.Series(
            False, index=self.df.index, dtype=bool
        )

        if len(valid) < 3:
            return mask, {
                "method": "Modified Z-score",
                "reason": "Insufficient observations",
            }

        median = valid.median()
        mad = (valid - median).abs().median()

        if mad == 0 or not np.isfinite(mad):
            return mask, {
                "method": "Modified Z-score",
                "reason": "Zero or invalid MAD",
            }

        scores = 0.6745 * (values - median).abs() / mad
        mask = (scores > threshold).fillna(False)

        return mask, {
            "method": "Modified Z-score",
            "median": float(median),
            "mad": float(mad),
            "threshold": threshold,
        }

    def detect(self, column, method="iqr", threshold=None):
        methods = {
            "iqr": self.detect_iqr,
            "zscore": self.detect_zscore,
            "modified_zscore": self.detect_modified_zscore,
        }

        if method not in methods:
            raise ValueError(
                f"Unsupported method: {method}"
            )

        if threshold is None:
            return methods[method](column)

        if method == "iqr":
            return methods[method](
                column, multiplier=threshold
            )

        return methods[method](
            column, threshold=threshold
        )

    def preview_outliers(
        self, column, method="iqr", threshold=None
    ):
        mask, details = self.detect(
            column, method, threshold
        )

        return {
            "column": column,
            "method": method,
            "outlier_count": int(mask.sum()),
            "outlier_percent": round(
                float(mask.sum()) / len(self.df) * 100, 2
            ) if len(self.df) else 0.0,
            "details": details,
            "outlier_rows": self.df.loc[
                mask
            ].copy(deep=True),
        }

    def summary(self, method="iqr", threshold=None):
        rows = []

        numeric_columns = self.df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            if pd.api.types.is_bool_dtype(self.df[column]):
                continue

            result = self.preview_outliers(
                column, method, threshold
            )

            rows.append({
                "column": column,
                "method": method,
                "outlier_count": result["outlier_count"],
                "outlier_percent": result["outlier_percent"],
                "details": str(result["details"]),
            })

        return pd.DataFrame(rows)

    def get_original_data(self):
        return self.df.copy(deep=True)

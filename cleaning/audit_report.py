
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cleaning.cleaning_engine import CleaningEngine


class DataQualityReport:
    """Generate an auditable report from a CleaningEngine."""

    def __init__(self, engine: CleaningEngine):
        if not isinstance(engine, CleaningEngine):
            raise TypeError("Expected a CleaningEngine instance.")

        self.engine = engine

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {
                str(k): DataQualityReport._json_safe(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                DataQualityReport._json_safe(v)
                for v in value
            ]

        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()

        if isinstance(value, np.ndarray):
            return DataQualityReport._json_safe(
                value.tolist()
            )

        if isinstance(value, np.generic):
            return DataQualityReport._json_safe(
                value.item()
            )

        if isinstance(value, float) and not np.isfinite(value):
            return None

        if pd.api.types.is_scalar(value) and pd.isna(value):
            return None

        return value

    def _column_quality(self, dataframe):
        rows = len(dataframe)
        result = []

        for column in dataframe.columns:
            series = dataframe[column]
            missing = int(series.isna().sum())

            result.append({
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_percent": (
                    round(missing / rows * 100, 2)
                    if rows else 0.0
                ),
                "unique_values": int(
                    series.nunique(dropna=True)
                ),
            })

        return result

    def generate(
        self,
        rules=None,
        outlier_columns=None,
        outlier_method="iqr",
    ):
        raw = self.engine.get_original_data()
        cleaned = self.engine.get_cleaned_data()
        quality = self.engine.quality_summary()

        report = {
            "report_type": "Data Quality and Cleaning Audit",
            "generated_at_utc": pd.Timestamp.now(
                tz="UTC"
            ).isoformat(),
            "quality_comparison": quality,
            "raw_column_profile": self._column_quality(raw),
            "cleaned_column_profile": self._column_quality(
                cleaned
            ),
            "transformations": self.engine.get_audit_log()
                .to_dict(orient="records"),
            "business_validation": None,
            "outlier_analysis": [],
            "warnings": [],
        }

        if rules is not None:
            validation = self.engine.validate(rules)

            report["business_validation"] = {
                "summary": validation["summary"],
                "issues": validation["report"].to_dict(
                    orient="records"
                ),
            }

        if outlier_columns:
            from cleaning.outlier_detector import OutlierDetector

            detector = OutlierDetector(cleaned)

            for column in outlier_columns:
                result = detector.preview_outliers(
                    column,
                    method=outlier_method,
                )

                report["outlier_analysis"].append({
                    "column": column,
                    "method": outlier_method,
                    "outlier_count": result["outlier_count"],
                    "outlier_percent": result["outlier_percent"],
                    "details": result["details"],
                })

        if quality["after"]["missing_cells"] > 0:
            report["warnings"].append(
                "The cleaned dataset still contains "
                "missing values."
            )

        if quality["after"]["exact_duplicates"] > 0:
            report["warnings"].append(
                "Exact duplicate records remain."
            )

        validation = report["business_validation"]

        if (
            validation is not None
            and validation["summary"]["total_issue_occurrences"] > 0
        ):
            report["warnings"].append(
                "Business-rule violations remain unresolved."
            )

        report["warnings"].append(
            "Completeness is not an overall data-quality "
            "score. Statistical outliers require review."
        )

        return self._json_safe(report)

    def export_json(self, filepath, **report_options):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate(**report_options)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return path


import copy
import pandas as pd

from cleaning.data_profiler import DataProfiler
from cleaning.missing_handler import MissingValueHandler
from cleaning.duplicate_detector import DuplicateDetector
from cleaning.standardizer import DataStandardizer
from cleaning.outlier_detector import OutlierDetector
from cleaning.business_validator import BusinessValidator


class CleaningEngine:
    """
    Controlled data-cleaning workflow.

    - Preserves the original dataset.
    - Applies approved transformations only.
    - Records operations and affected row counts.
    - Supports profiling, outliers and validation.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if not dataframe.columns.is_unique:
            raise ValueError("Column names must be unique.")

        self.original = dataframe.copy(deep=True)
        self.current = dataframe.copy(deep=True)
        self.audit_log = []

    def _record(self, operation, parameters, before, after):
        self.audit_log.append({
            "step": len(self.audit_log) + 1,
            "operation": operation,
            "parameters": copy.deepcopy(parameters),
            "rows_before": len(before),
            "rows_after": len(after),
            "missing_before": int(before.isna().sum().sum()),
            "missing_after": int(after.isna().sum().sum()),
        })

    def apply(self, operation, **parameters):
        """
        Apply one approved cleaning operation.

        Supported operations:
        missing_values
        remove_duplicates
        standardize_text
        standardize_categories
        standardize_numeric
        standardize_dates
        """
        before = self.current.copy(deep=True)

        if operation == "missing_values":
            handler = MissingValueHandler(before)
            handler.apply_treatment(**parameters)
            after = handler.get_cleaned_data()

        elif operation == "remove_duplicates":
            detector = DuplicateDetector(before)
            after = detector.remove_duplicates(**parameters)

        elif operation in {
            "standardize_text",
            "standardize_categories",
            "standardize_numeric",
            "standardize_dates",
        }:
            standardizer = DataStandardizer(before)

            method = getattr(standardizer, operation)
            after = method(**parameters)

        else:
            raise ValueError(
                f"Unsupported cleaning operation: {operation}"
            )

        # Commit only after the operation succeeds.
        self.current = after.copy(deep=True)

        self._record(
            operation,
            parameters,
            before,
            self.current,
        )

        return self.get_cleaned_data()

    def profile(self, original=False):
        dataset = (
            self.original if original else self.current
        )
        return DataProfiler(dataset).generate_report()

    def outlier_report(
        self,
        method="iqr",
        threshold=None,
    ):
        detector = OutlierDetector(self.current)
        return detector.summary(
            method=method,
            threshold=threshold,
        )

    def validate(self, rules):
        """
        Run configured business rules.

        Example:
        [
            {
                "method": "validate_required",
                "params": {"column": "Product"}
            }
        ]
        """
        allowed = {
            "validate_required",
            "validate_numeric_range",
            "validate_profit_relationship",
            "validate_date_order",
            "validate_unique",
        }

        validator = BusinessValidator(self.current)

        for rule in rules:
            method_name = rule["method"]

            if method_name not in allowed:
                raise ValueError(
                    f"Unsupported validation: {method_name}"
                )

            params = rule.get("params", {})
            getattr(validator, method_name)(**params)

        return {
            "report": validator.get_report(),
            "summary": validator.get_summary(),
        }

    def quality_summary(self):
        """
        Descriptive quality metrics, not a weighted
        enterprise data-quality score.
        """
        def calculate(df):
            rows = len(df)
            columns = len(df.columns)
            total_cells = rows * columns

            missing = int(df.isna().sum().sum())
            duplicates = int(df.duplicated().sum())

            completeness = (
                round(
                    (total_cells - missing)
                    / total_cells * 100,
                    2,
                )
                if total_cells else None
            )

            return {
                "rows": rows,
                "columns": columns,
                "missing_cells": missing,
                "exact_duplicates": duplicates,
                "completeness_percent": completeness,
            }

        return {
            "before": calculate(self.original),
            "after": calculate(self.current),
        }

    def get_cleaned_data(self):
        return self.current.copy(deep=True)

    def get_original_data(self):
        return self.original.copy(deep=True)

    def get_audit_log(self):
        return pd.DataFrame(
            self.audit_log,
            columns=[
                "step",
                "operation",
                "parameters",
                "rows_before",
                "rows_after",
                "missing_before",
                "missing_after",
            ],
        )

    def reset(self):
        """Restore the working copy to the raw data."""
        self.current = self.original.copy(deep=True)
        self.audit_log = []
        return self.get_cleaned_data()

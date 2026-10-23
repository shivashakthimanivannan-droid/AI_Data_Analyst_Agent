
import pandas as pd


class BusinessValidator:
    """
    Read-only business rule validation.

    Supported rules:
    - Required fields
    - Numeric minimum / maximum
    - Financial relationships
    - Date ordering
    - Unique identifiers

    Rules must be explicitly configured.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if not dataframe.columns.is_unique:
            raise ValueError("Column names must be unique.")

        self.df = dataframe.copy(deep=True)
        self.issues = []

    def _check_columns(self, *columns):
        missing = [
            column for column in columns
            if column not in self.df.columns
        ]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _record(self, rule, column, mask, severity="error"):
        if len(mask) != len(self.df):
            raise ValueError("Invalid validation mask.")

        mask = pd.Series(
            mask.to_numpy(dtype=bool),
            index=self.df.index
        )

        count = int(mask.sum())

        self.issues.append({
            "rule": rule,
            "column": column,
            "severity": severity,
            "issue_count": count,
            "affected_positions": [
                int(pos)
                for pos in range(len(mask))
                if mask.iloc[pos]
            ],
        })

        return count

    def validate_required(self, column):
        self._check_columns(column)

        series = self.df[column]
        mask = series.isna()

        if pd.api.types.is_string_dtype(series):
            mask = mask | series.astype(
                "string"
            ).str.strip().eq("").fillna(False)

        return self._record(
            "required_field", column, mask
        )

    def validate_numeric_range(
        self, column, minimum=None, maximum=None
    ):
        self._check_columns(column)

        if minimum is None and maximum is None:
            raise ValueError("Provide a minimum or maximum.")

        values = pd.to_numeric(
            self.df[column], errors="coerce"
        )

        mask = pd.Series(
            False, index=self.df.index
        )

        if minimum is not None:
            mask |= values < minimum

        if maximum is not None:
            mask |= values > maximum

        # Nonmissing values that cannot be interpreted
        # as numbers are also invalid.
        mask |= self.df[column].notna() & values.isna()

        return self._record(
            "numeric_range", column, mask.fillna(False)
        )

    def validate_profit_relationship(
        self, sales_col, profit_col, tolerance=0.01
    ):
        """
        Optional business rule:
        Profit must not exceed Sales.

        Negative profits are allowed.
        This rule assumes comparable accounting definitions.
        """
        self._check_columns(sales_col, profit_col)

        if tolerance < 0:
            raise ValueError("Tolerance cannot be negative.")

        sales = pd.to_numeric(
            self.df[sales_col], errors="coerce"
        )
        profit = pd.to_numeric(
            self.df[profit_col], errors="coerce"
        )

        mask = (
            sales.notna()
            & profit.notna()
            & (profit > sales + tolerance)
        )

        return self._record(
            "profit_exceeds_sales",
            f"{profit_col} / {sales_col}",
            mask.fillna(False),
        )

    def validate_date_order(
        self, start_col, end_col, date_format=None
    ):
        self._check_columns(start_col, end_col)

        start = pd.to_datetime(
            self.df[start_col],
            format=date_format,
            errors="coerce",
        )
        end = pd.to_datetime(
            self.df[end_col],
            format=date_format,
            errors="coerce",
        )

        invalid_start = (
            self.df[start_col].notna() & start.isna()
        )
        invalid_end = (
            self.df[end_col].notna() & end.isna()
        )

        mask = (
            invalid_start
            | invalid_end
            | (
                start.notna()
                & end.notna()
                & (end < start)
            )
        )

        return self._record(
            "invalid_date_order",
            f"{start_col} / {end_col}",
            mask.fillna(False),
        )

    def validate_unique(self, column):
        self._check_columns(column)

        series = self.df[column]

        # Missing IDs are handled by required-field checks.
        mask = (
            series.notna()
            & series.duplicated(keep=False)
        )

        return self._record(
            "unique_identifier", column, mask
        )

    def get_report(self):
        return pd.DataFrame(
            self.issues,
            columns=[
                "rule",
                "column",
                "severity",
                "issue_count",
                "affected_positions",
            ],
        )

    def get_summary(self):
        report = self.get_report()

        if report.empty:
            return {
                "rules_checked": 0,
                "failed_rules": 0,
                "total_issue_occurrences": 0,
            }

        return {
            "rules_checked": len(report),
            "failed_rules": int(
                (report["issue_count"] > 0).sum()
            ),
            "total_issue_occurrences": int(
                report["issue_count"].sum()
            ),
        }

    def get_original_data(self):
        return self.df.copy(deep=True)

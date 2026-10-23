
import re
import pandas as pd


class DataStandardizer:
    """Apply explicitly approved data-standardization rules."""

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")
        if not dataframe.columns.is_unique:
            raise ValueError("Column names must be unique.")

        self.original = dataframe.copy(deep=True)
        self.df = dataframe.copy(deep=True)
        self.audit_log = []

    def _check_column(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Column not found: {column}")

    def _record(self, column, operation, before, after):
        changed = (
            before.astype("string").fillna("<MISSING>")
            != after.astype("string").fillna("<MISSING>")
        )

        self.audit_log.append({
            "column": column,
            "operation": operation,
            "changed_rows": int(changed.sum()),
            "missing_before": int(before.isna().sum()),
            "missing_after": int(after.isna().sum()),
        })

    def standardize_text(
        self,
        column,
        case="title",
        strip_whitespace=True,
        collapse_spaces=True,
    ):
        self._check_column(column)

        if case not in {"title", "lower", "upper", "preserve"}:
            raise ValueError("Invalid case option.")

        before = self.df[column].copy()
        values = before.astype("string")

        if strip_whitespace:
            values = values.str.strip()

        if collapse_spaces:
            values = values.str.replace(
                r"\s+", " ", regex=True
            )

        if case == "title":
            values = values.str.title()
        elif case == "lower":
            values = values.str.lower()
        elif case == "upper":
            values = values.str.upper()

        self.df[column] = values

        self._record(
            column, f"standardize_text:{case}",
            before, self.df[column]
        )

        return self.get_cleaned_data()

    def standardize_categories(self, column, mapping):
        self._check_column(column)

        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dictionary.")

        before = self.df[column].copy()
        self.df[column] = before.replace(mapping)

        self._record(
            column, "standardize_categories",
            before, self.df[column]
        )

        return self.get_cleaned_data()

    def standardize_numeric(
        self,
        column,
        currency_symbols=("₹", "$", "€", "£"),
        percentage=False,
        strict=True,
    ):
        self._check_column(column)

        before = self.df[column].copy()

        def parse(value):
            if pd.isna(value):
                return float("nan")

            if isinstance(value, bool):
                raise ValueError("Boolean is not a numeric value.")

            text = str(value).strip()

            if not text:
                return float("nan")

            for symbol in currency_symbols:
                text = text.replace(symbol, "")

            text = text.replace(",", "").strip()

            has_percent = text.endswith("%")

            if has_percent:
                text = text[:-1].strip()

            if not re.fullmatch(
                r"[+-]?(?:\d+\.?\d*|\.\d+)",
                text
            ):
                raise ValueError(
                    f"Invalid numeric value: {value!r}"
                )

            number = float(text)

            if percentage and has_percent:
                number /= 100.0

            if has_percent and not percentage:
                raise ValueError(
                    "Percentage found; set percentage=True."
                )

            return number

        converted = []
        invalid = []

        for index, value in enumerate(before.tolist()):
            try:
                converted.append(parse(value))
            except (ValueError, TypeError) as error:
                invalid.append((index, str(error)))
                converted.append(float("nan"))

        if strict and invalid:
            raise ValueError(
                f"Invalid values in {column}: {invalid[:5]}. "
                "No changes were applied."
            )

        self.df[column] = pd.Series(
            converted,
            index=self.df.index,
            dtype="float64",
        )

        self._record(
            column, "standardize_numeric",
            before, self.df[column]
        )

        return self.get_cleaned_data()

    def standardize_dates(
        self,
        column,
        date_format,
        strict=True,
    ):
        self._check_column(column)

        if not date_format:
            raise ValueError(
                "An explicit date_format is required."
            )

        before = self.df[column].copy()

        converted = pd.to_datetime(
            before,
            format=date_format,
            errors="coerce",
        )

        invalid = before.notna() & converted.isna()

        if strict and invalid.any():
            raise ValueError(
                f"{int(invalid.sum())} invalid dates found. "
                "No changes were applied."
            )

        self.df[column] = converted

        self._record(
            column, "standardize_dates",
            before, self.df[column]
        )

        return self.get_cleaned_data()

    def get_cleaned_data(self):
        return self.df.copy(deep=True)

    def get_audit_log(self):
        return pd.DataFrame(self.audit_log)

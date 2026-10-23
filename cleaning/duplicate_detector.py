
import pandas as pd


class DuplicateDetector:
    """
    Detect exact and business-key duplicates.

    Detection does not change data.
    Removal requires an explicit method call.
    Original input remains unchanged.
    """

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Expected a Pandas DataFrame.")

        if not dataframe.columns.is_unique:
            raise ValueError("Column names must be unique.")

        self.original = dataframe.copy(deep=True)
        self.df = dataframe.copy(deep=True)
        self.audit_log = []

    def _validate_keys(self, key_columns):
        if not key_columns:
            raise ValueError(
                "Provide at least one business-key column."
            )

        if isinstance(key_columns, str):
            key_columns = [key_columns]

        key_columns = list(key_columns)

        if not key_columns:
            raise ValueError(
                "Provide at least one business-key column."
            )

        missing = [
            col for col in key_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Unknown key columns: {missing}"
            )

        if len(key_columns) != len(set(key_columns)):
            raise ValueError(
                "Business-key columns must be unique."
            )

        return key_columns

    def _duplicate_mask(
        self,
        key_columns=None,
        keep=False,
        ignore_missing_keys=True,
    ):
        if key_columns is None:
            return self.df.duplicated(keep=keep)

        keys = self._validate_keys(key_columns)

        mask = self.df.duplicated(
            subset=keys,
            keep=keep,
        )

        if ignore_missing_keys:
            valid_keys = self.df[keys].notna().all(axis=1)
            mask = mask & valid_keys

            # Recalculate duplicates only among complete keys.
            mask = pd.Series(
                False,
                index=self.df.index,
            )

            valid_subset = self.df.loc[valid_keys, keys]

            mask.loc[valid_keys] = (
                valid_subset.duplicated(keep=keep).to_numpy()
            )

        return mask

    def exact_duplicate_report(self):
        mask = self._duplicate_mask(keep="first")

        return {
            "duplicate_count": int(mask.sum()),
            "duplicate_percent": round(
                mask.sum() / len(self.df) * 100, 2
            ) if len(self.df) else 0.0,
            "affected_rows": int(
                self._duplicate_mask(keep=False).sum()
            ),
        }

    def business_key_report(
        self,
        key_columns,
        ignore_missing_keys=True,
    ):
        keys = self._validate_keys(key_columns)

        duplicate_mask = self._duplicate_mask(
            keys,
            keep="first",
            ignore_missing_keys=ignore_missing_keys,
        )

        affected_mask = self._duplicate_mask(
            keys,
            keep=False,
            ignore_missing_keys=ignore_missing_keys,
        )

        return {
            "key_columns": keys,
            "duplicate_count": int(duplicate_mask.sum()),
            "affected_rows": int(affected_mask.sum()),
            "duplicate_percent": round(
                duplicate_mask.sum() / len(self.df) * 100, 2
            ) if len(self.df) else 0.0,
            "ignored_missing_keys": ignore_missing_keys,
        }

    def preview_duplicates(
        self,
        key_columns=None,
        ignore_missing_keys=True,
    ):
        mask = self._duplicate_mask(
            key_columns=key_columns,
            keep=False,
            ignore_missing_keys=ignore_missing_keys,
        )

        return self.df.loc[mask].copy(deep=True)

    def remove_duplicates(
        self,
        key_columns=None,
        keep="first",
        ignore_missing_keys=True,
    ):
        if keep not in {"first", "last"}:
            raise ValueError(
                "keep must be 'first' or 'last'."
            )

        keys = (
            self._validate_keys(key_columns)
            if key_columns is not None
            else None
        )

        before = len(self.df)

        mask = self._duplicate_mask(
            key_columns=keys,
            keep=keep,
            ignore_missing_keys=ignore_missing_keys,
        )

        self.df = self.df.loc[~mask].copy(deep=True)

        removed = before - len(self.df)

        self.audit_log.append({
            "operation": (
                "remove_exact_duplicates"
                if keys is None
                else "remove_business_key_duplicates"
            ),
            "key_columns": keys,
            "keep": keep,
            "rows_before": before,
            "rows_after": len(self.df),
            "rows_removed": removed,
            "ignore_missing_keys": ignore_missing_keys,
        })

        return self.df.copy(deep=True)

    def get_cleaned_data(self):
        return self.df.copy(deep=True)

    def get_audit_log(self):
        return pd.DataFrame(self.audit_log)

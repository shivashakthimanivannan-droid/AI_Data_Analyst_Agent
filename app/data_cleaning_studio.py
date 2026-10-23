
"""Human-approved Data Cleaning Studio for the AI Data Analyst Agent."""

import hashlib
import json

import pandas as pd
import streamlit as st

from cleaning.cleaning_engine import CleaningEngine
from cleaning.audit_report import DataQualityReport
from cleaning.outlier_detector import OutlierDetector


# =========================================================
# HELPERS
# =========================================================

BUSINESS_NUMERIC_COLUMNS = {
    "sales",
    "revenue",
    "total_sales",
    "profit",
    "net_profit",
    "quantity",
    "units",
    "amount",
    "price",
    "cost",
}


def _fingerprint(df: pd.DataFrame, source_name: str) -> str:
    try:
        content = pd.util.hash_pandas_object(
            df, index=True
        ).to_numpy().tobytes()
    except (TypeError, ValueError):
        content = df.to_json(
            orient="split",
            date_format="iso",
            default_handler=str,
        ).encode()

    digest = hashlib.sha256()
    digest.update(source_name.encode("utf-8", errors="replace"))
    digest.update(str(df.dtypes.to_dict()).encode())
    digest.update(content)
    return digest.hexdigest()


def _options(df, numeric=False):
    if numeric:
        return [
            col
            for col in df.select_dtypes(include="number").columns
            if not pd.api.types.is_bool_dtype(df[col])
        ]

    return list(df.columns)


def _blank_mask(series):
    """Identify empty and whitespace-only text."""
    return series.map(
        lambda value: isinstance(value, str)
        and not value.strip()
    ).fillna(False)


def _is_numeric_business_column(df, column):
    """
    Identify numeric columns, including familiar business
    metrics accidentally loaded as object/string columns.
    """
    if column not in df.columns:
        return False

    if pd.api.types.is_bool_dtype(df[column]):
        return False

    if pd.api.types.is_numeric_dtype(df[column]):
        return True

    normalized_name = str(column).strip().lower()

    return normalized_name in BUSINESS_NUMERIC_COLUMNS


def _parse_numeric_constant(value):
    """
    Return a numeric value or raise ValueError.
    Never convert arbitrary text to a number.
    """
    text = str(value).strip()

    if not text:
        raise ValueError("Enter a numeric replacement value.")

    try:
        number = float(text.replace(",", ""))
    except ValueError:
        raise ValueError(
            "Numeric columns require a valid number."
        )

    if not pd.notna(number) or not (
        float("-inf") < number < float("inf")
    ):
        raise ValueError(
            "Numeric replacement must be a finite number."
        )

    return number


def _numeric_issues(df):
    """
    Find invalid numeric text in recognized business metrics.
    Does not modify the dataset.
    """
    issues = []

    for column in df.columns:
        if not _is_numeric_business_column(df, column):
            continue

        series = df[column]

        numeric = pd.to_numeric(
            series, errors="coerce"
        )

        invalid = series.notna() & numeric.isna()

        count = int(invalid.sum())

        if count:
            examples = (
                series.loc[invalid]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )

            issues.append({
                "Column": column,
                "Invalid values": count,
                "Examples": ", ".join(examples),
            })

    return issues


def _profit_sales_check(df, sales_col, profit_col):
    """Flag records where profit exceeds sales."""
    sales = pd.to_numeric(
        df[sales_col], errors="coerce"
    )
    profit = pd.to_numeric(
        df[profit_col], errors="coerce"
    )

    mask = (
        sales.notna()
        & profit.notna()
        & (profit > sales)
    )

    positions = [
        int(i)
        for i, flagged in enumerate(mask.to_numpy())
        if flagged
    ]

    details = df.loc[mask].copy()

    details.insert(
        0,
        "Source row position",
        positions,
    )

    return {
        "rule": "profit_exceeds_sales",
        "sales_column": sales_col,
        "profit_column": profit_col,
        "issue_count": len(positions),
        "affected_positions": positions,
        "details": details,
    }


def _missing_numeric_check(df):
    """Summarize true nulls in recognized numeric columns."""
    issues = []

    for column in df.columns:
        if not _is_numeric_business_column(df, column):
            continue

        count = int(df[column].isna().sum())

        if count:
            issues.append({
                "Column": column,
                "Missing values": count,
            })

    return issues


# =========================================================
# MAIN STUDIO
# =========================================================

def render_data_cleaning_studio(
    raw_df: pd.DataFrame,
    source_name: str,
):
    """
    Render the cleaning interface.

    Return either the original dataset or the approved
    working dataset for downstream analytics.
    """

    signature = _fingerprint(raw_df, source_name)

    if st.session_state.get("dc_signature") != signature:
        st.session_state.dc_signature = signature
        st.session_state.dc_engine = CleaningEngine(raw_df)
        st.session_state.dc_use_cleaned = False
        st.session_state.dc_confirmed = False
        st.session_state.dc_rules = []
        st.session_state.dc_extra_checks = []
        st.session_state.messages = []

    engine = st.session_state.dc_engine
    current = engine.get_cleaned_data()

    with st.expander(
        "🧹 Data Cleaning Studio — review and approve changes",
        expanded=False,
    ):
        st.caption(
            "Original data is preserved. Every transformation "
            "affects only the working copy and requires approval."
        )

        comparison = engine.quality_summary()
        before = comparison["before"]
        after = comparison["after"]

        a, b, c, d = st.columns(4)

        a.metric(
            "Original rows",
            f"{before['rows']:,}",
        )

        b.metric(
            "Working rows",
            f"{after['rows']:,}",
            delta=after["rows"] - before["rows"],
        )

        c.metric(
            "Completeness",
            (
                f"{after['completeness_percent']}%"
                if after["completeness_percent"] is not None
                else "N/A"
            ),
        )

        d.metric(
            "Exact duplicates",
            str(after["exact_duplicates"]),
        )

        (
            overview_tab,
            clean_tab,
            validate_tab,
            export_tab,
        ) = st.tabs([
            "Data audit",
            "Approve cleaning",
            "Validate & outliers",
            "Compare & export",
        ])

        # =====================================================
        # TAB 1: DATA AUDIT
        # =====================================================

        with overview_tab:
            st.markdown("**Raw data column profile**")

            st.dataframe(
                engine.profile(original=True)["columns"],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Current working data profile**")

            st.dataframe(
                engine.profile()["columns"],
                use_container_width=True,
                hide_index=True,
            )

            blank_counts = {
                col: int(_blank_mask(current[col]).sum())
                for col in current.columns
            }

            blank_counts = {
                col: count
                for col, count in blank_counts.items()
                if count > 0
            }

            if blank_counts:
                st.warning(
                    "Empty or whitespace-only text detected. "
                    "These values may not be counted as nulls."
                )

                st.dataframe(
                    pd.DataFrame([
                        {
                            "Column": col,
                            "Blank text values": count,
                        }
                        for col, count in blank_counts.items()
                    ]),
                    hide_index=True,
                    use_container_width=True,
                )

            numeric_issues = _numeric_issues(current)

            if numeric_issues:
                st.error(
                    "Non-numeric text was detected in one or "
                    "more business metric columns."
                )

                st.dataframe(
                    pd.DataFrame(numeric_issues),
                    hide_index=True,
                    use_container_width=True,
                )

                st.info(
                    "Review these values before calculating KPIs. "
                    "Do not replace missing financial values with "
                    "text such as 'unknown'."
                )

            missing_numeric = _missing_numeric_check(current)

            if missing_numeric:
                st.warning(
                    "Some numeric business columns contain "
                    "missing values."
                )

                st.dataframe(
                    pd.DataFrame(missing_numeric),
                    hide_index=True,
                    use_container_width=True,
                )

            st.caption(
                "Completeness is descriptive and does not "
                "guarantee that the dataset is valid."
            )

        # =====================================================
        # TAB 2: APPROVE CLEANING
        # =====================================================

        with clean_tab:
            st.warning(
                "Review each transformation before applying it. "
                "Removing rows or duplicates may change totals."
            )

            operations = {
                "Missing values": "missing_values",
                "Remove duplicates": "remove_duplicates",
                "Standardize text": "standardize_text",
                "Map categories": "standardize_categories",
                "Replace blank text": "replace_blank_text",
                "Parse numeric/currency": "standardize_numeric",
                "Parse dates": "standardize_dates",
            }

            with st.form("dc_clean_form"):
                selected = st.selectbox(
                    "Transformation",
                    list(operations),
                    key="dc_operation",
                )

                op = operations[selected]
                kwargs = {}

                column = None
                old = None

                if op == "remove_duplicates":
                    keys = st.multiselect(
                        "Business-key columns "
                        "(leave empty for exact duplicates)",
                        _options(current),
                        key="dc_keys",
                    )

                    keep = st.selectbox(
                        "Keep which record?",
                        ["first", "last"],
                        key="dc_keep",
                    )

                    kwargs = {
                        "key_columns": keys or None,
                        "keep": keep,
                    }

                    st.caption(
                        "Leave business-key columns empty to "
                        "remove only exact duplicate rows."
                    )

                else:
                    column = st.selectbox(
                        "Column",
                        _options(current),
                        key="dc_column",
                    )

                    kwargs["column"] = column

                    if op == "missing_values":
                        strategy = st.selectbox(
                            "Treatment",
                            [
                                "keep",
                                "drop_rows",
                                "median",
                                "mean",
                                "mode",
                                "constant",
                            ],
                            key="dc_strategy",
                        )

                        kwargs["strategy"] = strategy

                        if strategy == "constant":
                            is_numeric = _is_numeric_business_column(
                                current,
                                column,
                            )

                            label = (
                                "Numeric replacement value"
                                if is_numeric
                                else "Replacement value"
                            )

                            kwargs["fill_value"] = st.text_input(
                                label,
                                key="dc_fill",
                            )

                            if is_numeric:
                                st.caption(
                                    "Only numeric values are accepted "
                                    "for this column. Use 'keep' to "
                                    "preserve missing financial values."
                                )

                        elif strategy == "keep":
                            st.info(
                                "Missing values will be preserved. "
                                "No replacement value is invented."
                            )

                    elif op == "replace_blank_text":
                        replacement = st.text_input(
                            "Replace empty/whitespace text with",
                            value="Unknown",
                            key="dc_blank_replacement",
                        )

                        kwargs["replacement"] = replacement

                        count = int(
                            _blank_mask(current[column]).sum()
                        )

                        st.info(
                            f"{count} blank text value(s) "
                            f"found in {column}."
                        )

                    elif op == "standardize_text":
                        kwargs["case"] = st.selectbox(
                            "Letter case",
                            [
                                "title",
                                "lower",
                                "upper",
                                "preserve",
                            ],
                            key="dc_case",
                        )

                    elif op == "standardize_categories":
                        old = st.text_input(
                            "Existing exact category",
                            key="dc_old",
                        )

                        new = st.text_input(
                            "Approved replacement",
                            key="dc_new",
                        )

                        kwargs["mapping"] = {
                            old: new
                        }

                    elif op == "standardize_numeric":
                        kwargs["percentage"] = st.checkbox(
                            "Convert percent strings to decimal rates",
                            key="dc_percentage",
                        )

                        kwargs["strict"] = True

                    elif op == "standardize_dates":
                        kwargs["date_format"] = st.text_input(
                            "Explicit date format",
                            value="%Y-%m-%d",
                            key="dc_date_format",
                        )

                        kwargs["strict"] = True

                approved = st.checkbox(
                    "I reviewed and approve this transformation",
                    key="dc_approve",
                )

                submitted = st.form_submit_button(
                    "Apply approved transformation",
                    type="primary",
                )

            if submitted:
                if not approved:
                    st.error(
                        "Approve the transformation first."
                    )

                elif (
                    op == "missing_values"
                    and kwargs.get("strategy") == "constant"
                    and not str(
                        kwargs.get("fill_value", "")
                    ).strip()
                ):
                    st.error(
                        "Enter a non-empty replacement value."
                    )

                elif (
                    op == "missing_values"
                    and kwargs.get("strategy") == "constant"
                    and _is_numeric_business_column(
                        current,
                        kwargs["column"],
                    )
                    and pd.to_numeric(
                        pd.Series([
                            kwargs["fill_value"]
                        ]),
                        errors="coerce",
                    ).isna().iloc[0]
                ):
                    st.error(
                        "This is a numeric business column. "
                        "Enter a valid number or choose 'keep'. "
                        "Text such as 'unknown' is not allowed."
                    )

                elif (
                    op == "standardize_categories"
                    and not str(old or "").strip()
                ):
                    st.error(
                        "Enter the category to replace."
                    )

                elif (
                    op in {
                        "standardize_categories",
                        "standardize_text",
                        "replace_blank_text",
                    }
                    and _is_numeric_business_column(
                        current,
                        column,
                    )
                ):
                    st.error(
                        "Text and category transformations are "
                        "not permitted on recognized numeric "
                        "business columns. Use numeric parsing "
                        "or missing-value treatment instead."
                    )

                elif op == "replace_blank_text":
                    replacement = kwargs["replacement"].strip()
                    mask = _blank_mask(current[column])

                    if not replacement:
                        st.error(
                            "Replacement must not be empty."
                        )

                    elif not mask.any():
                        st.info(
                            "No blank text values found."
                        )

                    else:
                        try:
                            engine.apply(
                                "standardize_categories",
                                column=column,
                                mapping={
                                    value: replacement
                                    for value in current.loc[
                                        mask, column
                                    ].unique()
                                },
                            )

                            remaining = int(
                                _blank_mask(
                                    engine.get_cleaned_data()[column]
                                ).sum()
                            )

                            if remaining:
                                st.warning(
                                    "Some blank values remain. "
                                    "Review the working dataset."
                                )
                            else:
                                st.success(
                                    "Blank text replacement applied."
                                )

                            st.session_state.dc_confirmed = False
                            st.rerun()

                        except Exception as exc:
                            st.error(
                                f"Blank replacement failed: {exc}"
                            )

                else:
                    try:
                        # Convert numeric constant input into a
                        # real number before passing to the handler.
                        if (
                            op == "missing_values"
                            and kwargs.get("strategy") == "constant"
                            and _is_numeric_business_column(
                                current,
                                kwargs["column"],
                            )
                        ):
                            kwargs["fill_value"] = (
                                _parse_numeric_constant(
                                    kwargs["fill_value"]
                                )
                            )

                        engine.apply(op, **kwargs)

                        st.session_state.dc_confirmed = False

                        st.success(
                            "Transformation applied to working copy."
                        )

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            f"Transformation not applied: {exc}"
                        )

            if st.button(
                "Reset working copy to original",
                key="dc_reset",
            ):
                engine.reset()

                st.session_state.dc_use_cleaned = False
                st.session_state.dc_confirmed = False
                st.session_state.dc_rules = []
                st.session_state.dc_extra_checks = []
                st.session_state.messages = []

                st.rerun()

        # =====================================================
        # TAB 3: VALIDATE & OUTLIERS
        # =====================================================

        with validate_tab:
            current = engine.get_cleaned_data()

            st.markdown(
                "**Statistical outlier review**"
            )

            candidates = _options(
                current,
                numeric=True,
            )

            selected_metrics = st.multiselect(
                "Numeric measures to inspect (exclude IDs)",
                candidates,
                default=[],
                key="dc_outlier_columns",
            )

            method = st.selectbox(
                "Detection method",
                [
                    "iqr",
                    "zscore",
                    "modified_zscore",
                ],
                key="dc_outlier_method",
            )

            if selected_metrics:
                detector = OutlierDetector(current)

                for col in selected_metrics:
                    try:
                        result = detector.preview_outliers(
                            col,
                            method=method,
                        )

                        st.write(
                            f"**{col}**: "
                            f"{result['outlier_count']} flagged; "
                            f"{result['details']}"
                        )

                        if result["outlier_count"]:
                            st.dataframe(
                                result["outlier_rows"],
                                use_container_width=True,
                            )

                    except Exception as exc:
                        st.warning(
                            f"Could not inspect {col}: {exc}"
                        )

            st.markdown(
                "**Configurable business-rule checks**"
            )

            rules = []
            columns = _options(current)

            required = st.multiselect(
                "Required columns",
                columns,
                key="dc_required",
            )

            for col in required:
                rules.append({
                    "method": "validate_required",
                    "params": {
                        "column": col,
                    },
                })

            unique = st.multiselect(
                "Unique identifier columns "
                "(checked separately)",
                columns,
                key="dc_unique",
            )

            for col in unique:
                rules.append({
                    "method": "validate_unique",
                    "params": {
                        "column": col,
                    },
                })

            nonnegative = st.multiselect(
                "Numeric columns requiring values ≥ 0",
                candidates,
                key="dc_nonnegative",
            )

            for col in nonnegative:
                rules.append({
                    "method": "validate_numeric_range",
                    "params": {
                        "column": col,
                        "minimum": 0,
                    },
                })

            st.session_state.dc_rules = rules

            if rules:
                try:
                    validation = engine.validate(rules)

                    st.dataframe(
                        validation["report"],
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.json(
                        validation["summary"]
                    )

                except Exception as exc:
                    st.error(
                        f"Validation failed: {exc}"
                    )

            else:
                st.info(
                    "Choose business rules to validate "
                    "the working data."
                )

            st.divider()

            st.markdown(
                "**Cross-column financial validation**"
            )

            enable_profit_rule = st.checkbox(
                "Flag records where Profit exceeds Sales",
                key="dc_profit_sales_enabled",
            )

            extra_checks = []

            if enable_profit_rule:
                numeric_candidates = _options(
                    current,
                    numeric=True,
                )

                if len(numeric_candidates) < 2:
                    st.warning(
                        "At least two numeric columns "
                        "are required."
                    )

                else:
                    sales_default = (
                        numeric_candidates.index("Sales")
                        if "Sales" in numeric_candidates
                        else 0
                    )

                    profit_default = (
                        numeric_candidates.index("Profit")
                        if "Profit" in numeric_candidates
                        else 0
                    )

                    sales_col = st.selectbox(
                        "Sales / Revenue column",
                        numeric_candidates,
                        index=sales_default,
                        key="dc_sales_rule_col",
                    )

                    profit_col = st.selectbox(
                        "Profit column",
                        numeric_candidates,
                        index=profit_default,
                        key="dc_profit_rule_col",
                    )

                    if sales_col == profit_col:
                        st.warning(
                            "Select two different columns."
                        )

                    else:
                        check = _profit_sales_check(
                            current,
                            sales_col,
                            profit_col,
                        )

                        extra_checks.append(check)

                        if check["issue_count"]:
                            st.warning(
                                f"{check['issue_count']} record(s) "
                                "have Profit greater than Sales."
                            )

                            st.dataframe(
                                check["details"],
                                use_container_width=True,
                                hide_index=True,
                            )

                        else:
                            st.success(
                                "No Profit > Sales violations "
                                "found among valid numeric records."
                            )

            st.session_state.dc_extra_checks = [
                {
                    "rule": check["rule"],
                    "sales_column": check["sales_column"],
                    "profit_column": check["profit_column"],
                    "issue_count": check["issue_count"],
                    "affected_positions": check[
                        "affected_positions"
                    ],
                }
                for check in extra_checks
            ]

            st.caption(
                "Missing numeric values are not treated as zero. "
                "Flagged records require human review."
            )

        # =====================================================
        # TAB 4: COMPARE & EXPORT
        # =====================================================

        with export_tab:
            current = engine.get_cleaned_data()

            comparison = engine.quality_summary()

            st.markdown(
                "**Before vs. after**"
            )

            st.dataframe(
                pd.DataFrame(comparison).T,
                use_container_width=True,
            )

            st.markdown(
                "**Transformation audit log**"
            )

            audit_log = engine.get_audit_log()

            st.dataframe(
                audit_log,
                use_container_width=True,
                hide_index=True,
            )

            if audit_log.empty:
                st.info(
                    "No cleaning transformations have been "
                    "applied yet. The working dataset is "
                    "still the original dataset."
                )

            st.markdown(
                "**Working dataset preview**"
            )

            st.dataframe(
                current.head(100),
                use_container_width=True,
                hide_index=True,
            )

            report = DataQualityReport(engine).generate(
                rules=st.session_state.get(
                    "dc_rules", []
                ),
                outlier_columns=st.session_state.get(
                    "dc_outlier_columns", []
                ),
                outlier_method=st.session_state.get(
                    "dc_outlier_method", "iqr"
                ),
            )

            extra = []

            if st.session_state.get(
                "dc_profit_sales_enabled",
                False,
            ):
                sales_col = st.session_state.get(
                    "dc_sales_rule_col",
                    "Sales",
                )

                profit_col = st.session_state.get(
                    "dc_profit_rule_col",
                    "Profit",
                )

                if (
                    sales_col in current.columns
                    and profit_col in current.columns
                    and sales_col != profit_col
                ):
                    check = _profit_sales_check(
                        current,
                        sales_col,
                        profit_col,
                    )

                    extra.append({
                        "rule": check["rule"],
                        "sales_column": sales_col,
                        "profit_column": profit_col,
                        "issue_count": check["issue_count"],
                        "affected_positions": check[
                            "affected_positions"
                        ],
                    })

            report["additional_business_checks"] = extra

            if extra and any(
                item["issue_count"] > 0
                for item in extra
            ):
                report.setdefault(
                    "warnings", []
                ).append(
                    "Profit exceeds Sales in one or more records."
                )

            numeric_issues = _numeric_issues(current)
            missing_numeric = _missing_numeric_check(current)

            report["numeric_type_issues"] = numeric_issues
            report["missing_numeric_values"] = missing_numeric

            if numeric_issues:
                st.error(
                    "Export warning: invalid text remains in "
                    "numeric business columns."
                )

                report.setdefault(
                    "warnings", []
                ).append(
                    "Invalid numeric text remains in the dataset."
                )

            if missing_numeric:
                st.warning(
                    "Some numeric values are still missing. "
                    "They will remain empty in the CSV export."
                )

            duplicate_count = int(
                current.duplicated().sum()
            )

            if duplicate_count:
                st.warning(
                    f"{duplicate_count} exact duplicate row(s) "
                    "remain in the working dataset."
                )

            if not audit_log.empty:
                st.success(
                    f"{len(audit_log)} approved transformation(s) "
                    "recorded in the audit log."
                )

            st.download_button(
                "Download cleaned CSV",
                data=current.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                key="dc_download_csv",
            )

            st.download_button(
                "Download data-quality JSON report",
                data=json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                file_name="data_quality_report.json",
                mime="application/json",
                key="dc_download_report",
            )

        st.divider()

        st.checkbox(
            "Use approved working dataset for dashboard and agents",
            key="dc_use_cleaned",
        )

        if st.session_state.dc_use_cleaned:
            st.warning(
                "Dashboard and agents use the working dataset. "
                "Unresolved quality issues may affect results."
            )

        else:
            st.info(
                "Dashboard and agents currently use "
                "the original dataset."
            )

    # =========================================================
    # ACTIVE DATASET SELECTION
    # =========================================================

    active_signature = (
        signature,
        bool(st.session_state.dc_use_cleaned),
        len(engine.get_audit_log()),
    )

    if (
        st.session_state.get("dc_active_signature")
        != active_signature
    ):
        st.session_state.dc_active_signature = active_signature
        st.session_state.messages = []

    selected_df = (
        engine.get_cleaned_data()
        if st.session_state.dc_use_cleaned
        else raw_df.copy(deep=True)
    )

    return selected_df


import json
from pathlib import Path

import pandas as pd

from cleaning.cleaning_engine import CleaningEngine
from cleaning.audit_report import DataQualityReport


def main():
    df = pd.DataFrame({
        "OrderID": [101, 102, 102, 103, 104],
        "Product": [
            " laptop ", "PHONE", "PHONE",
            "tablet", None
        ],
        "Sales": [
            50000, 20000, 20000, 15000, None
        ],
        "Profit": [
            10000, 4000, 4000, 20000, 3000
        ],
        "Quantity": [5, 2, 2, -1, 1],
    })

    original = df.copy(deep=True)
    engine = CleaningEngine(df)

    engine.apply(
        "standardize_text",
        column="Product",
    )
    engine.apply(
        "missing_values",
        column="Product",
        strategy="constant",
        fill_value="Unknown",
    )
    engine.apply("remove_duplicates")

    rules = [
        {
            "method": "validate_numeric_range",
            "params": {
                "column": "Quantity",
                "minimum": 1,
            },
        },
        {
            "method": "validate_profit_relationship",
            "params": {
                "sales_col": "Sales",
                "profit_col": "Profit",
            },
        },
    ]

    reporter = DataQualityReport(engine)

    report = reporter.generate(
        rules=rules,
        outlier_columns=["Sales", "Profit"],
    )

    print("\n===== QUALITY COMPARISON =====")
    print(
        json.dumps(
            report["quality_comparison"],
            indent=2,
        )
    )

    print("\n===== BUSINESS VALIDATION =====")
    print(
        json.dumps(
            report["business_validation"],
            indent=2,
        )
    )

    print("\n===== WARNINGS =====")
    for warning in report["warnings"]:
        print("-", warning)

    output = Path("reports/data_quality_report.json")

    reporter.export_json(
        output,
        rules=rules,
        outlier_columns=["Sales", "Profit"],
    )

    with output.open("r", encoding="utf-8") as file:
        exported = json.load(file)

    assert report["quality_comparison"]["before"][
        "completeness_percent"
    ] == 92.0

    assert report["quality_comparison"]["after"][
        "completeness_percent"
    ] == 95.0

    assert len(report["transformations"]) == 3

    assert report["business_validation"]["summary"][
        "total_issue_occurrences"
    ] == 2

    assert len(report["outlier_analysis"]) == 2

    assert exported["report_type"] == (
        "Data Quality and Cleaning Audit"
    )

    pd.testing.assert_frame_equal(df, original)

    print("\nPASS: Quality comparison.")
    print("PASS: Transformation audit.")
    print("PASS: Business validation report.")
    print("PASS: Outlier analysis.")
    print("PASS: JSON export.")
    print("PASS: Original dataset unchanged.")

    print(f"\nReport saved to: {output.resolve()}")


if __name__ == "__main__":
    main()

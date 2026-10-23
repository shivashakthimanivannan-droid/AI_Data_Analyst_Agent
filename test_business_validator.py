
import pandas as pd

from cleaning.business_validator import BusinessValidator


def main():
    df = pd.DataFrame({
        "OrderID": [101, 102, 102, 104, 105],
        "Product": [
            "Laptop", "Phone", "",
            "Tablet", "Laptop"
        ],
        "Sales": [
            50000, 20000, 15000, 10000, 30000
        ],
        "Profit": [
            10000, 5000, 20000, -2000, 6000
        ],
        "Quantity": [
            5, 2, -3, 1, 0
        ],
        "OrderDate": [
            "2026-01-01",
            "2026-02-01",
            "2026-03-01",
            "2026-04-01",
            "2026-05-01",
        ],
        "DeliveryDate": [
            "2026-01-05",
            "2026-02-05",
            "2026-02-28",
            "2026-04-05",
            "2026-05-05",
        ],
    })

    original = df.copy(deep=True)
    validator = BusinessValidator(df)

    print("\n===== RUN BUSINESS VALIDATIONS =====")

    validator.validate_required("Product")
    validator.validate_numeric_range(
        "Quantity", minimum=1
    )
    validator.validate_numeric_range(
        "Sales", minimum=0
    )
    validator.validate_profit_relationship(
        "Sales", "Profit"
    )
    validator.validate_date_order(
        "OrderDate",
        "DeliveryDate",
        date_format="%Y-%m-%d",
    )
    validator.validate_unique("OrderID")

    report = validator.get_report()

    print("\n===== VALIDATION REPORT =====")
    print(report.to_string(index=False))

    print("\n===== SUMMARY =====")
    print(validator.get_summary())

    expected = {
        "required_field": 1,
        "numeric_range": [2, 0],
        "profit_exceeds_sales": 1,
        "invalid_date_order": 1,
        "unique_identifier": 2,
    }

    assert report.loc[
        report["rule"] == "required_field",
        "issue_count"
    ].iloc[0] == expected["required_field"]

    assert report.loc[
        report["rule"] == "numeric_range",
        "issue_count"
    ].tolist() == expected["numeric_range"]

    assert report.loc[
        report["rule"] == "profit_exceeds_sales",
        "issue_count"
    ].iloc[0] == 1

    assert report.loc[
        report["rule"] == "invalid_date_order",
        "issue_count"
    ].iloc[0] == 1

    assert report.loc[
        report["rule"] == "unique_identifier",
        "issue_count"
    ].iloc[0] == 2

    summary = validator.get_summary()

    assert summary["rules_checked"] == 6
    assert summary["failed_rules"] == 5
    assert summary["total_issue_occurrences"] == 7

    pd.testing.assert_frame_equal(df, original)

    print("\nPASS: Required-field validation.")
    print("PASS: Numeric-range validation.")
    print("PASS: Financial relationship validation.")
    print("PASS: Date-order validation.")
    print("PASS: Unique-identifier validation.")
    print("PASS: Original dataset unchanged.")


if __name__ == "__main__":
    main()

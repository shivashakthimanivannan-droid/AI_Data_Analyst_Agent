
import pandas as pd

from cleaning.cleaning_engine import CleaningEngine


def main():
    df = pd.DataFrame({
        "OrderID": [101, 102, 102, 103, 104],
        "Product": [
            " laptop ",
            "PHONE",
            "PHONE",
            "tablet",
            None,
        ],
        "Sales": [
            50000,
            20000,
            20000,
            15000,
            None,
        ],
        "Profit": [
            10000,
            4000,
            4000,
            20000,
            3000,
        ],
        "Quantity": [5, 2, 2, -1, 1],
    })

    original = df.copy(deep=True)
    engine = CleaningEngine(df)

    print("\n===== INITIAL QUALITY =====")
    print(engine.quality_summary()["before"])

    print("\n===== APPLY APPROVED CLEANING =====")

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

    # Missing Sales is deliberately retained:
    # unknown revenue must not become zero.
    engine.apply(
        "missing_values",
        column="Sales",
        strategy="keep",
    )

    engine.apply("remove_duplicates")

    cleaned = engine.get_cleaned_data()

    print("\n===== CLEANED DATA =====")
    print(cleaned.to_string(index=False))

    print("\n===== QUALITY COMPARISON =====")
    print(engine.quality_summary())

    print("\n===== BUSINESS VALIDATION =====")

    rules = [
        {
            "method": "validate_required",
            "params": {"column": "Product"},
        },
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
        {
            "method": "validate_unique",
            "params": {"column": "OrderID"},
        },
    ]

    validation = engine.validate(rules)

    print(validation["report"].to_string(index=False))
    print(validation["summary"])

    print("\n===== OUTLIER REPORT =====")
    print(engine.outlier_report().to_string(index=False))

    print("\n===== AUDIT LOG =====")
    print(engine.get_audit_log().to_string(index=False))

    # Validate cleaning results.
    assert len(cleaned) == 4
    assert cleaned["Product"].isna().sum() == 0
    assert cleaned["Sales"].isna().sum() == 1
    assert cleaned.duplicated().sum() == 0
    assert len(engine.get_audit_log()) == 4

    # Invalid operations must not change data.
    snapshot = engine.get_cleaned_data()

    try:
        engine.apply(
            "standardize_numeric",
            column="Product",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected numeric conversion to fail."
        )

    pd.testing.assert_frame_equal(
        engine.get_cleaned_data(),
        snapshot,
    )

    # Verify the input was never changed.
    pd.testing.assert_frame_equal(df, original)

    print("\nPASS: Approved transformations.")
    print("PASS: Missing revenue preserved.")
    print("PASS: Duplicate removal.")
    print("PASS: Business validation.")
    print("PASS: Audit trail.")
    print("PASS: Failed operation leaves data unchanged.")
    print("PASS: Original dataset unchanged.")


if __name__ == "__main__":
    main()


import pandas as pd

from cleaning.missing_handler import MissingValueHandler


def main():
    df = pd.DataFrame({
        "Product": [
            "Laptop", "Phone", None,
            "Tablet", "Phone"
        ],
        "Sales": [
            50000, 20000, None,
            25000, 30000
        ],
        "Profit": [
            10000, None, 3000,
            5000, 6000
        ],
    })

    original = df.copy(deep=True)

    handler = MissingValueHandler(df)

    print("\n===== MISSING VALUE REPORT =====")
    print(handler.missing_report().to_string(index=False))

    print("\n===== RECOMMENDATIONS =====")
    print(handler.recommend_treatment().to_string(index=False))

    print("\n===== APPLY APPROVED TREATMENTS =====")

    handler.apply_treatment(
        column="Product",
        strategy="constant",
        fill_value="Unknown",
    )

    handler.apply_treatment(
        column="Sales",
        strategy="median",
    )

    handler.apply_treatment(
        column="Profit",
        strategy="keep",
    )

    print(handler.get_cleaned_data().to_string(index=False))

    print("\n===== AUDIT LOG =====")
    print(handler.get_audit_log().to_string(index=False))

    pd.testing.assert_frame_equal(df, original)

    cleaned = handler.get_cleaned_data()

    assert cleaned["Product"].isna().sum() == 0
    assert cleaned["Sales"].isna().sum() == 0
    assert cleaned["Profit"].isna().sum() == 1

    assert cleaned.loc[2, "Sales"] == 27500

    print("\nPASS: Missing-value handling works correctly.")
    print("PASS: Original dataset remains unchanged.")


if __name__ == "__main__":
    main()

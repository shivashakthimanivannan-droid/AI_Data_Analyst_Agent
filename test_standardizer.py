
import pandas as pd

from cleaning.standardizer import DataStandardizer


def main():
    df = pd.DataFrame({
        "Region": [
            " north ", "SOUTH", " east", "WEST "
        ],
        "Product": [
            "LAPTOP", "phone", "tablet", "laptop"
        ],
        "Sales": [
            "₹1,25,000", "₹50,000",
            "₹25,000", "₹10,000"
        ],
        "Discount": [
            "10%", "15%", "5%", "20%"
        ],
        "Date": [
            "2026/01/15", "2026/02/10",
            "2026/03/05", "2026/04/20"
        ],
    })

    original = df.copy(deep=True)
    cleaner = DataStandardizer(df)

    cleaner.standardize_text("Region")
    cleaner.standardize_text("Product")
    cleaner.standardize_numeric("Sales")
    cleaner.standardize_numeric(
        "Discount", percentage=True
    )
    cleaner.standardize_dates(
        "Date", date_format="%Y/%m/%d"
    )

    cleaned = cleaner.get_cleaned_data()

    print("\n===== ORIGINAL DATA =====")
    print(original.to_string(index=False))

    print("\n===== STANDARDIZED DATA =====")
    print(cleaned.to_string(index=False))

    print("\n===== AUDIT LOG =====")
    print(cleaner.get_audit_log().to_string(index=False))

    assert cleaned["Region"].tolist() == [
        "North", "South", "East", "West"
    ]
    assert cleaned["Product"].tolist() == [
        "Laptop", "Phone", "Tablet", "Laptop"
    ]
    assert cleaned["Sales"].tolist() == [
        125000.0, 50000.0, 25000.0, 10000.0
    ]
    assert cleaned["Discount"].tolist() == [
        0.10, 0.15, 0.05, 0.20
    ]
    assert pd.api.types.is_datetime64_any_dtype(
        cleaned["Date"]
    )

    pd.testing.assert_frame_equal(df, original)

    print("\nPASS: Text standardization.")
    print("PASS: Currency conversion.")
    print("PASS: Percentage conversion.")
    print("PASS: Date standardization.")
    print("PASS: Original dataset unchanged.")


if __name__ == "__main__":
    main()


import pandas as pd

from cleaning.duplicate_detector import DuplicateDetector


def main():
    df = pd.DataFrame({
        "OrderID": [
            101, 102, 102, 103, 104, 104, 105
        ],
        "Product": [
            "Laptop", "Phone", "Phone",
            "Tablet", "Laptop", "Laptop", "Phone"
        ],
        "Sales": [
            50000, 20000, 20000,
            15000, 60000, 65000, 25000
        ],
    })

    original = df.copy(deep=True)

    detector = DuplicateDetector(df)

    print("\n===== EXACT DUPLICATES =====")
    exact = detector.exact_duplicate_report()
    print(exact)

    print("\n===== BUSINESS-KEY DUPLICATES =====")
    business = detector.business_key_report(
        key_columns=["OrderID"]
    )
    print(business)

    print("\n===== PREVIEW BUSINESS-KEY DUPLICATES =====")
    print(
        detector.preview_duplicates(
            key_columns=["OrderID"]
        ).to_string(index=False)
    )

    assert exact["duplicate_count"] == 1
    assert business["duplicate_count"] == 2
    assert business["affected_rows"] == 4

    print("\n===== REMOVE EXACT DUPLICATES =====")
    detector.remove_duplicates()
    print(detector.get_cleaned_data().to_string(index=False))

    assert len(detector.get_cleaned_data()) == 6

    print("\n===== AUDIT LOG =====")
    print(detector.get_audit_log().to_string(index=False))

    pd.testing.assert_frame_equal(df, original)

    print("\nPASS: Exact duplicate detection works.")
    print("PASS: Business-key duplicate detection works.")
    print("PASS: Approved removal works.")
    print("PASS: Original dataset remains unchanged.")


if __name__ == "__main__":
    main()

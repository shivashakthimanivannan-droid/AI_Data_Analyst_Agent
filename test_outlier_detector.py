
import pandas as pd

from cleaning.outlier_detector import OutlierDetector


def main():
    df = pd.DataFrame({
        "Product": [
            "Laptop", "Phone", "Tablet",
            "Laptop", "Phone", "Tablet",
            "Laptop", "Phone", "Enterprise"
        ],
        "Sales": [
            10, 12, 11, 13, 12,
            14, 11, 12, 1000
        ],
        "Profit": [
            2, 3, 2, 4, 3,
            4, 2, 3, 200
        ],
    })

    original = df.copy(deep=True)
    detector = OutlierDetector(df)

    print("\n===== IQR DETECTION =====")
    iqr = detector.preview_outliers(
        "Sales", method="iqr"
    )
    print(iqr["details"])
    print(iqr["outlier_rows"].to_string(index=False))

    print("\n===== Z-SCORE DETECTION =====")
    zscore = detector.preview_outliers(
        "Sales", method="zscore", threshold=2.0
    )
    print(zscore["details"])
    print(zscore["outlier_rows"].to_string(index=False))

    print("\n===== MODIFIED Z-SCORE =====")
    modified = detector.preview_outliers(
        "Sales", method="modified_zscore"
    )
    print(modified["details"])
    print(modified["outlier_rows"].to_string(index=False))

    print("\n===== ALL NUMERIC COLUMNS =====")
    print(
        detector.summary(method="iqr").to_string(
            index=False
        )
    )

    assert iqr["outlier_count"] == 1
    assert zscore["outlier_count"] == 1
    assert modified["outlier_count"] == 1

    assert iqr["outlier_rows"]["Sales"].iloc[0] == 1000

    pd.testing.assert_frame_equal(df, original)

    print("\nPASS: IQR detection.")
    print("PASS: Z-score detection.")
    print("PASS: Modified Z-score detection.")
    print("PASS: Original dataset unchanged.")


if __name__ == "__main__":
    main()

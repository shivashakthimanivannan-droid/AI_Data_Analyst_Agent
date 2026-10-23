
import pandas as pd

from cleaning.data_profiler import DataProfiler

def main():
    df = pd.DataFrame({
        "Product": [
            "Laptop", "Phone", "Laptop",
            "Tablet", "Phone", "Phone"
        ],
        "Sales": [
            50000, 20000, 50000,
            None, 25000, 1000000
        ],
        "Profit": [
            10000, 4000, 10000,
            3000, None, 150000
        ],
    })

    original = df.copy(deep=True)

    profiler = DataProfiler(df)
    report = profiler.generate_report()

    print("\n===== DATASET OVERVIEW =====")
    print(report["overview"])

    print("\n===== COLUMN PROFILE =====")
    print(report["columns"].to_string(index=False))

    print("\n===== DUPLICATE REPORT =====")
    print(report["duplicates"])

    print("\n===== NUMERIC PROFILE =====")
    print(report["numeric"])

    print("\n===== OUTLIER REPORT =====")
    print(report["outliers"])

    pd.testing.assert_frame_equal(df, original)
    print("\nPASS: Original dataset was not modified.")


if __name__ == "__main__":
    main()


import ast
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "app" / "app.py"

TESTS = []


def record(name, passed, detail=""):
    TESTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"       {detail}")


def check(name, condition, detail=""):
    record(name, bool(condition), detail)


def load_app_functions():
    """
    Extract data-processing functions and their dependencies
    from app.py without launching Streamlit.
    """
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)

    wanted = {
        "normalize_name",
        "detect_columns",
        "clean_numeric",
        "parse_dates",
        "prepare_dataset",
        "monthly_aggregate",
        "group_aggregate",
        "best_group",
    }

    selected_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in wanted
    ]

    aliases_node = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "ALIASES"
                for target in node.targets
            ):
                aliases_node = node
                break

    if aliases_node is None:
        raise RuntimeError(
            "ALIASES dictionary was not found in app/app.py"
        )

    namespace = {
        "pd": pd,
        "re": __import__("re"),
    }

    module = ast.Module(
        body=[aliases_node] + selected_functions,
        type_ignores=[],
    )

    exec(
        compile(module, str(APP_PATH), "exec"),
        namespace,
    )

    missing = wanted - set(namespace)

    if missing:
        raise RuntimeError(
            f"Missing application functions: {missing}"
        )

    if not namespace.get("ALIASES"):
        raise RuntimeError(
            "The application's ALIASES dictionary is empty."
        )

    return namespace

def main():
    print("=" * 65)
    print("AI DATA ANALYST AGENT - DATASET COMPATIBILITY TEST")
    print("=" * 65)

    funcs = load_app_functions()

    detect = funcs["detect_columns"]
    prepare = funcs["prepare_dataset"]
    monthly = funcs["monthly_aggregate"]
    best = funcs["best_group"]

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)

        # -------------------------------------------------
        # TEST 1: STANDARD SALES DATASET
        # -------------------------------------------------
        sales = pd.DataFrame({
            "Date": ["2026-01-05", "2026-02-05"],
            "Region": ["North", "South"],
            "Product": ["Laptop", "Tablet"],
            "Sales": [10000, 5000],
            "Profit": [2500, 1000],
            "Quantity": [2, 1],
        })

        mapping = detect(sales)
        prepared = prepare(sales, mapping)

        check(
            "Standard sales column detection",
            all(mapping.get(role) for role in (
                "date", "region", "product",
                "sales", "profit", "quantity"
            )),
        )

        check(
            "Standard sales calculations",
            prepared["Sales"].sum() == 15000
            and prepared["Profit"].sum() == 3500,
        )

        # -------------------------------------------------
        # TEST 2: ALTERNATIVE COLUMN NAMES
        # -------------------------------------------------
        alternate = pd.DataFrame({
            "Order Date": ["2026-01-01"],
            "Territory": ["East"],
            "Item": ["Phone"],
            "Revenue": [8000],
            "Net Profit": [1800],
            "Units Sold": [4],
        })

        alt_mapping = detect(alternate)

        check(
            "Alternative business column names",
            all(alt_mapping.get(role) for role in (
                "date", "region", "product",
                "sales", "profit", "quantity"
            )),
        )

        # -------------------------------------------------
        # TEST 3: HR DATASET
        # -------------------------------------------------
        hr = pd.DataFrame({
            "Employee ID": [1, 2, 3],
            "Department": ["HR", "IT", "Finance"],
            "Salary": [40000, 60000, 55000],
            "Age": [25, 29, 32],
        })

        hr_mapping = detect(hr)
        hr_prepared = prepare(hr, hr_mapping)

        check(
            "HR dataset loads without sales columns",
            len(hr_prepared) == 3
            and hr_mapping["sales"] is None,
        )

        # -------------------------------------------------
        # TEST 4: CUSTOMER CHURN DATASET
        # -------------------------------------------------
        churn = pd.DataFrame({
            "CustomerID": [101, 102, 103],
            "Tenure": [12, 24, 6],
            "MonthlyCharges": [450, 700, 300],
            "Churn": ["No", "Yes", "No"],
        })

        churn_mapping = detect(churn)
        churn_prepared = prepare(churn, churn_mapping)

        check(
            "Customer churn dataset preparation",
            len(churn_prepared) == 3
            and "Churn" in churn_prepared.columns,
        )

        # -------------------------------------------------
        # TEST 5: CURRENCY VALUES
        # -------------------------------------------------
        currency = pd.DataFrame({
            "Sales": ["$1,200", "$2,500", "$300"],
            "Profit": ["200", "500", "50"],
        })

        currency_mapping = detect(currency)
        currency_prepared = prepare(
            currency, currency_mapping
        )

        check(
            "Currency and comma cleaning",
            currency_prepared["Sales"].sum() == 4000,
        )

        # -------------------------------------------------
        # TEST 6: MISSING VALUES
        # -------------------------------------------------
        missing = pd.DataFrame({
            "Sales": [1000, None, 2000],
            "Profit": [200, None, 400],
        })

        missing_prepared = prepare(
            missing, detect(missing)
        )

        check(
            "Partial missing numeric values",
            missing_prepared["Sales"].sum() == 3000
            and missing_prepared["Sales"].isna().sum() == 1,
        )

        # -------------------------------------------------
        # TEST 7: ALL MISSING SALES
        # -------------------------------------------------
        all_missing = pd.DataFrame({
            "Sales": [None, None, None],
        })

        all_missing_prepared = prepare(
            all_missing, detect(all_missing)
        )

        total = all_missing_prepared["Sales"].sum()

        check(
            "All-missing sales should not appear as zero",
            pd.isna(total),
            f"Current total: {total!r}",
        )

        # -------------------------------------------------
        # TEST 8: INVALID NUMERIC VALUES
        # -------------------------------------------------
        invalid = pd.DataFrame({
            "Sales": ["1000", "unknown", "2500"],
        })

        invalid_prepared = prepare(
            invalid, detect(invalid)
        )

        check(
            "Invalid numeric values become missing",
            invalid_prepared["Sales"].isna().sum() == 1,
            "A warning is recommended when conversion fails.",
        )

        # -------------------------------------------------
        # TEST 9: MONTHLY AGGREGATION
        # -------------------------------------------------
        result = monthly(
            prepared, "Date", ["Sales", "Profit"]
        )

        check(
            "Monthly aggregation",
            len(result) == 2
            and result["Sales"].sum() == 15000,
        )

        # -------------------------------------------------
        # TEST 10: PRODUCT RANKING
        # -------------------------------------------------
        product, value = best(
            prepared, "Product", "Profit", True
        )

        check(
            "Highest-profit product",
            product == "Laptop" and value == 2500,
        )

        # -------------------------------------------------
        # TEST 11: CSV FILE ROUNDTRIP
        # -------------------------------------------------
        csv_path = folder / "sales.csv"
        sales.to_csv(csv_path, index=False)

        loaded_csv = pd.read_csv(csv_path)

        check(
            "CSV loading",
            loaded_csv.shape == sales.shape,
        )

        # -------------------------------------------------
        # TEST 12: XLSX FILE ROUNDTRIP
        # -------------------------------------------------
        try:
            xlsx_path = folder / "sales.xlsx"
            sales.to_excel(xlsx_path, index=False)
            loaded_excel = pd.read_excel(xlsx_path)

            check(
                "XLSX loading",
                loaded_excel.shape == sales.shape,
            )

        except Exception as exc:
            record("XLSX loading", False, str(exc))

        # -------------------------------------------------
        # TEST 13: EMPTY DATASET
        # -------------------------------------------------
        empty = pd.DataFrame(columns=["Sales", "Profit"])

        check(
            "Empty dataset detectable",
            empty.empty,
        )

        # -------------------------------------------------
        # TEST 14: INVALID DATE VALUES
        # -------------------------------------------------
        bad_dates = pd.DataFrame({
            "Date": [
                "2026-01-05",
                "not-a-date",
                "2026-02-01",
            ],
            "Sales": [100, 200, 300],
        })

        dates_prepared = prepare(
            bad_dates, detect(bad_dates)
        )

        check(
            "Invalid dates become missing",
            dates_prepared["Date"].isna().sum() == 1,
        )

        # -------------------------------------------------
        # TEST 15: AMBIGUOUS DATE FORMAT
        # -------------------------------------------------
        ambiguous = pd.DataFrame({
            "Date": ["01/05/2026"],
            "Sales": [100],
        })

        ambiguous_prepared = prepare(
            ambiguous, detect(ambiguous)
        )

        check(
            "Ambiguous date follows month-first convention",
            ambiguous_prepared["Date"].iloc[0].month == 1,
            "For day-first datasets, the expected month may differ.",
        )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------
    passed = sum(1 for _, success, _ in TESTS if success)
    failed = len(TESTS) - passed

    print("\n" + "=" * 65)
    print("FINAL TEST REPORT")
    print("=" * 65)

    print(f"Total tests : {len(TESTS)}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if failed:
        print("\nCases requiring attention:")
        for name, success, detail in TESTS:
            if not success:
                print(f"- {name}")
                if detail:
                    print(f"  {detail}")

    print("=" * 65)


if __name__ == "__main__":
    main()

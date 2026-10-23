import pandas as pd
from tools.analysis_tools import analyze_column
from tools.analysis_tools import (
    dataset_summary,
    total_sales,
    total_profit,
    total_quantity,
    profit_margin,
    sales_by_region,
    sales_by_product,
    profit_by_region,
    profit_by_product,
    monthly_sales_profit
)


# Load your dataset
df = pd.read_csv("data/sample/sample.csv")


print("\n========== DATASET SUMMARY ==========")
print(dataset_summary(df))

print("\n========== TOTAL SALES ==========")
print(total_sales(df))

print("\n========== TOTAL PROFIT ==========")
print(total_profit(df))

print("\n========== TOTAL QUANTITY ==========")
print(total_quantity(df))

print("\n========== PROFIT MARGIN ==========")
print(profit_margin(df), "%")

print("\n========== SALES BY REGION ==========")
print(sales_by_region(df))

print("\n========== SALES BY PRODUCT ==========")
print(sales_by_product(df))

print("\n========== PROFIT BY REGION ==========")
print(profit_by_region(df))

print("\n========== PROFIT BY PRODUCT ==========")
print(profit_by_product(df))

print("\n========== MONTHLY SALES & PROFIT ==========")
print(monthly_sales_profit(df))

print("\n========== GENERAL ANALYSIS ==========")

print(
    analyze_column(
        df,
        "Sales",
        "mean"
    )
)

print(
    analyze_column(
        df,
        "Profit",
        "max"
    )
)
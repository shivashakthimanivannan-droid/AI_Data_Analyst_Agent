from agents.file_agent import analyze_dataset


file_path = "data/sample/sample.csv"

print("\n========== AI DATA ANALYST ==========\n")

result = analyze_dataset(
    file_path,
    "Show me sales by region as a chart"
)

print(result)
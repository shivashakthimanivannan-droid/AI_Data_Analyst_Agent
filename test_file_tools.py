from tools.file_tools import load_dataset, profile_dataset


file_path = "data/sample/sample.csv"

df = load_dataset(file_path)

profile = profile_dataset(df)

print("\n===== DATASET PROFILE =====")

print("Rows:", profile["rows"])
print("Columns:", profile["columns"])

print("\nColumn Names:")
print(profile["column_names"])

print("\nData Types:")
print(profile["data_types"])

print("\nMissing Values:")
print(profile["missing_values"])

print("\nDuplicate Rows:")
print(profile["duplicate_rows"])

print("\nNumeric Columns:")
print(profile["numeric_columns"])

print("\nCategorical Columns:")
print(profile["categorical_columns"])
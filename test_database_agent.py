
from agents.database_agent import run_database_agent
from tools.mysql_tools import (
    initialize_database,
    get_database_schema,
)


if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE AGENT TEST")
    print("=" * 60)

    try:
        database_path = initialize_database()
        print(f"\nDatabase created/ready: {database_path}")

        print("\nDatabase schema:")
        schema = get_database_schema()
        for table_name, details in schema.items():
            print(f"\nTable: {table_name}")
            print(f"Rows: {details['row_count']}")
            print("Columns:")
            for column in details["columns"]:
                print(f"  - {column['name']} ({column['type']})")

        question = "Which product has the highest profit?"

        print(f"\nQuestion: {question}")
        result = run_database_agent(question)

        print("\nGenerated SQL:")
        print(result.get("sql") or "No SQL generated")

        print("\nAnswer:")
        print(result["answer"])

        print("\nTest finished.")

    except Exception as exc:
        print(f"\nTEST FAILED: {exc}")
        raise

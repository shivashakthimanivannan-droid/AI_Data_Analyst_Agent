
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "database" / "analytics.db"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "sample" / "sample.csv"
DEFAULT_TABLE_NAME = "sales_data"
MAX_RESULT_ROWS = 200


def _quote_identifier(identifier: str) -> str:
    """Safely quote a SQLite table or column identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def initialize_database(
    csv_path=None,
    database_path=None,
    table_name=DEFAULT_TABLE_NAME,
):
    """
    Create the SQLite database and import a CSV if the target table
    does not already exist.
    """
    db_path = Path(database_path or DEFAULT_DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        existing_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()

        if existing_table:
            return str(db_path)

        source_csv = Path(csv_path or DEFAULT_CSV_PATH)

        if not source_csv.exists():
            raise FileNotFoundError(
                f"CSV file not found: {source_csv}"
            )

        dataframe = pd.read_csv(source_csv)

        if dataframe.empty:
            raise ValueError("The CSV file contains no data.")

        if len(dataframe.columns) == 0:
            raise ValueError("The CSV file has no columns.")

        # pandas creates the table and infers suitable SQL data types.
        dataframe.to_sql(
            table_name,
            connection,
            if_exists="fail",
            index=False,
        )

    return str(db_path)


def list_tables(database_path=None):
    """Return user-created tables in the SQLite database."""
    db_path = Path(database_path or DEFAULT_DATABASE_PATH)

    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

    return [row[0] for row in rows]


def get_database_schema(database_path=None):
    """Return table names, column names, types, and row counts."""
    db_path = Path(database_path or DEFAULT_DATABASE_PATH)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. "
            "Initialize it first."
        )

    schema = {}

    with sqlite3.connect(db_path) as connection:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        for (table_name,) in tables:
            quoted_table = _quote_identifier(table_name)

            columns = connection.execute(
                f"PRAGMA table_info({quoted_table})"
            ).fetchall()

            count = connection.execute(
                f"SELECT COUNT(*) FROM {quoted_table}"
            ).fetchone()[0]

            schema[table_name] = {
                "columns": [
                    {
                        "name": column[1],
                        "type": column[2] or "UNKNOWN",
                    }
                    for column in columns
                ],
                "row_count": count,
            }

    return schema


def execute_read_only_query(
    sql,
    database_path=None,
    max_rows=MAX_RESULT_ROWS,
):
    """
    Execute a read-only SQL query.

    Only SELECT or WITH queries are accepted. SQLite's authorizer
    independently blocks writes, schema modifications, ATTACH,
    and other unauthorized operations.
    """
    db_path = Path(database_path or DEFAULT_DATABASE_PATH)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. "
            "Initialize it first."
        )

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query cannot be empty.")

    query = sql.strip()

    # Permit a single statement, optionally ending in one semicolon.
    query = query.removesuffix(";").strip()

    if ";" in query:
        raise ValueError(
            "Only one SQL statement is allowed."
        )

    first_keyword = query.split(None, 1)[0].upper()

    if first_keyword not in {"SELECT", "WITH"}:
        raise ValueError(
            "Only SELECT queries and read-only CTEs are allowed."
        )

    max_rows = max(1, min(int(max_rows), MAX_RESULT_ROWS))

    with sqlite3.connect(db_path, timeout=10) as connection:
        connection.execute("PRAGMA query_only = ON")

        def authorize(action, arg1, arg2, database_name, trigger_name):
            allowed_actions = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }

            if action in allowed_actions:
                return sqlite3.SQLITE_OK

            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)

        cursor = connection.execute(query)
        columns = (
            [item[0] for item in cursor.description]
            if cursor.description
            else []
        )

        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }

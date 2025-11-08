"""dump_db.py

Utility script to read all rows from all tables in the project's SQLite
database and write a JSON dump to `db_dump.json` in the repository root.

Usage:
    python dump_db.py

This uses the project's `DatabaseManager` to ensure the same DB path and
schema are used.
"""
import json
from pathlib import Path
from database.db_manager import DatabaseManager


def dump_all_tables(db_path: str = None, out_path: str = "db_dump.json"):
    db = DatabaseManager(db_path) if db_path else DatabaseManager()
    conn = db.conn
    cursor = conn.cursor()

    # Get user tables (ignore sqlite internal tables)
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    tables = [row[0] for row in cursor.fetchall()]

    result = {}

    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        # sqlite3.Row -> dict conversion
        table_rows = []
        for r in rows:
            try:
                table_rows.append(dict(r))
            except Exception:
                # Fallback: convert by index + description
                cols = [c[0] for c in cursor.description]
                table_rows.append({cols[i]: r[i] for i in range(len(cols))})

        result[table] = table_rows

    # Write JSON file
    out_file = Path(out_path)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Print summary to stdout
    for table, rows in result.items():
        print(f"Table: {table} — rows: {len(rows)}")

    print(f"\nFull JSON dump written to: {out_file.resolve()}")


if __name__ == "__main__":
    dump_all_tables()

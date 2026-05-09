import sqlite3


def _table_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _add_missing_columns(cursor, table_name, column_specs):
    existing = _table_columns(cursor, table_name)
    added = []

    for column_name, column_sql in column_specs:
        if column_name not in existing:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")
            added.append(f"{table_name}.{column_name}")

    return added


def migrate_database(db_file="calls.db"):
    """
    Create/migrate the SQLite schema used by the current app.

    This is intentionally non-destructive:
    - creates missing tables
    - adds missing nullable/default-safe columns
    - never drops, rewrites, or deletes existing call data
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    added = []

    c.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_name TEXT,
        transcript TEXT,
        report TEXT,
        score INTEGER,
        risk TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        auto_disposition TEXT,
        manual_disposition TEXT,
        final_disposition TEXT,
        disposition_reason TEXT,
        duration_seconds INTEGER
    )
    """)

    added.extend(_add_missing_columns(c, "calls", [
        ("auto_disposition", "TEXT"),
        ("manual_disposition", "TEXT"),
        ("final_disposition", "TEXT"),
        ("disposition_reason", "TEXT"),
        ("duration_seconds", "INTEGER"),
    ]))

    c.execute("""
    CREATE TABLE IF NOT EXISTS processing_state (
        call_name TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0,
        message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    added.extend(_add_missing_columns(c, "processing_state", [
        ("progress", "INTEGER NOT NULL DEFAULT 0"),
        ("message", "TEXT"),
        ("attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("error", "TEXT"),
    ]))

    c.execute("""
    CREATE TABLE IF NOT EXISTS upload_times (
        filename TEXT PRIMARY KEY,
        uploaded_time INTEGER NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    return added

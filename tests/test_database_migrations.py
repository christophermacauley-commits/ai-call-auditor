import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_migrations import migrate_database


EXPECTED_CALLS_COLUMNS = {
    "id",
    "call_name",
    "transcript",
    "report",
    "score",
    "risk",
    "timestamp",
    "auto_disposition",
    "manual_disposition",
    "final_disposition",
    "disposition_reason",
    "duration_seconds",
}

EXPECTED_PROCESSING_STATE_COLUMNS = {
    "call_name",
    "filename",
    "status",
    "progress",
    "message",
    "attempts",
    "error",
    "updated_at",
}

EXPECTED_UPLOAD_TIMES_COLUMNS = {
    "filename",
    "uploaded_time",
}


def check(name, condition):
    if not condition:
        raise AssertionError(f"{name} failed")


def table_columns(conn, table_name):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def table_names(conn):
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def test_migration_adds_missing_calls_columns_without_rewriting_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "old_calls.db"

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_name TEXT,
            transcript TEXT,
            report TEXT,
            score INTEGER,
            risk TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        c.execute(
            """
            INSERT INTO calls (call_name, transcript, report, score, risk)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("old_schema_fixture", "original transcript", "original report", 90, "LOW"),
        )
        conn.commit()
        conn.close()

        added = migrate_database(str(db_path))

        conn = sqlite3.connect(db_path)
        columns = table_columns(conn, "calls")
        row = conn.execute(
            "SELECT call_name, transcript, report, score, risk FROM calls"
        ).fetchone()
        conn.close()

        check("calls columns added", EXPECTED_CALLS_COLUMNS.issubset(columns))
        check(
            "expected calls migrations reported",
            {
                "calls.auto_disposition",
                "calls.manual_disposition",
                "calls.final_disposition",
                "calls.disposition_reason",
                "calls.duration_seconds",
            }.issubset(set(added)),
        )
        check(
            "old calls data preserved",
            row == (
                "old_schema_fixture",
                "original transcript",
                "original report",
                90,
                "LOW",
            ),
        )


def test_migration_creates_missing_helper_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "empty.db"

        migrate_database(str(db_path))

        conn = sqlite3.connect(db_path)
        names = table_names(conn)
        calls_columns = table_columns(conn, "calls")
        processing_columns = table_columns(conn, "processing_state")
        upload_columns = table_columns(conn, "upload_times")
        conn.close()

        check("calls table created", "calls" in names)
        check("processing_state table created", "processing_state" in names)
        check("upload_times table created", "upload_times" in names)
        check("calls table has current columns", EXPECTED_CALLS_COLUMNS.issubset(calls_columns))
        check(
            "processing_state has current columns",
            EXPECTED_PROCESSING_STATE_COLUMNS.issubset(processing_columns),
        )
        check(
            "upload_times has current columns",
            EXPECTED_UPLOAD_TIMES_COLUMNS.issubset(upload_columns),
        )


def test_migration_is_idempotent():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "idempotent.db"

        first_added = migrate_database(str(db_path))
        second_added = migrate_database(str(db_path))

        conn = sqlite3.connect(db_path)
        columns = table_columns(conn, "calls")
        conn.close()

        check("first migration creates schema", EXPECTED_CALLS_COLUMNS.issubset(columns))
        check("second migration has no new columns", second_added == [])


test_migration_adds_missing_calls_columns_without_rewriting_data()
test_migration_creates_missing_helper_tables()
test_migration_is_idempotent()

print("Database migration guardrail tests passed.")

#!/usr/bin/env python3
"""
Check generated reports against human-approved golden expectations.

This does not re-run audits. It reads files in reports/ and verifies key truths:
stage, sold status, risk, autofail, and major contradictions.

Usage:
  python3 tests/check_golden_reports.py
"""

from __future__ import annotations

import json
import sqlite3
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
GOLDEN_FILE = ROOT / "golden_cases" / "golden18.json"


def discover_db_file() -> Path | None:
    """
    Find the SQLite DB that has the calls table.

    The app has used different DB_FILE names across patches/environments, so
    the golden checker should discover it instead of hard-coding audits.db.
    """
    candidates = []
    for name in ["audit.db", "audits.db", "calls.db", "call_audits.db"]:
        candidates.append(ROOT / name)
    candidates.extend(ROOT.glob("*.db"))
    candidates.extend(ROOT.glob("*.sqlite"))
    candidates.extend(ROOT.glob("*.sqlite3"))

    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            conn = sqlite3.connect(candidate)
            c = conn.cursor()
            row = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='calls'"
            ).fetchone()
            conn.close()
            if row:
                return candidate
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            continue

    return None


DB_FILE = discover_db_file()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def read_report(path: Path) -> str:
    return path.read_text(errors="ignore")


def find_report(match: str) -> Path | None:
    if not match:
        return None

    candidates = sorted(REPORTS_DIR.glob("*_report.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    needle = norm(match)

    exactish = []
    loose = []
    for path in candidates:
        stem = path.name.removesuffix("_report.txt")
        nstem = norm(stem)
        if needle == nstem:
            exactish.append(path)
        elif needle in nstem:
            loose.append(path)

    if exactish:
        return exactish[0]
    if loose:
        return loose[0]
    return None


def extract_line_value(report: str, label_regex: str) -> str | None:
    m = re.search(label_regex, report, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def report_contains(report: str, phrase: str) -> bool:
    return norm(phrase) in norm(report)


def status_value_ok(actual: str | None, expected) -> bool:
    if actual is None:
        return False
    actual_n = norm(actual)
    if isinstance(expected, list):
        return any(actual_n == norm(x) for x in expected)
    return actual_n == norm(expected)



def get_db_disposition(call_name: str) -> str | None:
    """Read final/auto/manual disposition from the local calls table when available."""
    if not DB_FILE or not DB_FILE.exists() or not call_name:
        return None

    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        row = c.execute(
            """
            SELECT final_disposition, manual_disposition, auto_disposition
            FROM calls
            WHERE call_name=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (call_name,),
        ).fetchone()
        conn.close()
    except Exception:
        return None

    if not row:
        return None

    final_disposition, manual_disposition, auto_disposition = row
    return final_disposition or manual_disposition or auto_disposition

def check_case(case: dict) -> tuple[str, list[str]]:
    case_id = case.get("id", "(missing id)")
    match = case.get("match", case_id)
    failures: list[str] = []

    report_path = find_report(match)
    if not report_path:
        return case_id, [f"Missing report matching: {match!r}"]

    report = read_report(report_path)

    expected_stage = case.get("expected_stage")
    if expected_stage:
        actual = extract_line_value(report, r"^CALL STAGE REACHED:\s*(.+?)\s*$")
        if not status_value_ok(actual, expected_stage):
            failures.append(f"Stage expected {expected_stage!r}, got {actual!r} in {report_path.name}")

    expected_risk = case.get("expected_risk")
    if expected_risk:
        actual = extract_line_value(report, r"^RISK:\s*(.+?)\s*$")
        if not status_value_ok(actual, expected_risk):
            failures.append(f"Risk expected {expected_risk!r}, got {actual!r} in {report_path.name}")

    expected_policy_sold = case.get("expected_policy_sold")
    if expected_policy_sold:
        actual = extract_line_value(report, r"^- Policy sold:\s*(YES|NO)\s*$")
        if not actual:
            actual = extract_line_value(report, r"^- Was the policy sold\?\s*(YES|NO)\s*$")
        if not status_value_ok(actual, expected_policy_sold):
            failures.append(f"Policy sold expected {expected_policy_sold!r}, got {actual!r} in {report_path.name}")

    expected_autofail = case.get("expected_autofail")
    if expected_autofail:
        actual = extract_line_value(report, r"^- Automatic fail triggered:\s*(YES|NO)\s*$")
        if not status_value_ok(actual, expected_autofail):
            failures.append(f"Autofail expected {expected_autofail!r}, got {actual!r} in {report_path.name}")

    expected_final_stage = case.get("expected_final_supporting_stage")
    if expected_final_stage:
        actual = extract_line_value(report, r"^- Final stage supporting sale:\s*(.+?)\s*$")
        if not status_value_ok(actual, expected_final_stage):
            failures.append(f"Final supporting stage expected {expected_final_stage!r}, got {actual!r} in {report_path.name}")

    expected_disposition = case.get("expected_disposition")
    if expected_disposition:
        call_name = report_path.name.removesuffix("_report.txt")
        actual = get_db_disposition(call_name)
        if not actual:
            actual = extract_line_value(report, r"^DISPOSITION:\s*(.+?)\s*$")
        if not status_value_ok(actual, expected_disposition):
            failures.append(f"Disposition expected {expected_disposition!r}, got {actual!r} for {report_path.name}")

    for phrase in case.get("must_contain", []):
        if not report_contains(report, phrase):
            failures.append(f"Missing required phrase in {report_path.name}: {phrase!r}")

    for options in [case.get("must_contain_any", [])]:
        if options and not any(report_contains(report, phrase) for phrase in options):
            failures.append(f"Missing at least one acceptable phrase in {report_path.name}: {options!r}")

    for phrase in case.get("must_not_contain", []):
        if report_contains(report, phrase):
            failures.append(f"Forbidden phrase present in {report_path.name}: {phrase!r}")

    return case_id, failures


def main() -> int:
    if not GOLDEN_FILE.exists():
        print(f"Missing golden file: {GOLDEN_FILE}")
        return 2

    data = json.loads(GOLDEN_FILE.read_text())
    global_forbidden = data.get("global_must_not_contain", [])
    cases = data.get("cases", [])

    all_failures: list[tuple[str, str]] = []

    for case in cases:
        case_id, failures = check_case(case)
        for failure in failures:
            all_failures.append((case_id, failure))

    # Global checks apply to all matched reports.
    for case in cases:
        report_path = find_report(case.get("match", case.get("id", "")))
        if not report_path:
            continue
        report = read_report(report_path)
        for phrase in global_forbidden:
            if report_contains(report, phrase):
                all_failures.append((case.get("id", "(missing id)"), f"Global forbidden phrase present in {report_path.name}: {phrase!r}"))

    if all_failures:
        print("Golden report checks FAILED:")
        current = None
        for case_id, failure in all_failures:
            if case_id != current:
                print(f"\n[{case_id}]")
                current = case_id
            print(f"- {failure}")
        print(f"\nTotal failures: {len(all_failures)}")
        return 1

    print(f"Golden report checks passed for {len(cases)} cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

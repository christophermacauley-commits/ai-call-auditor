import argparse
import os
import sqlite3

def save_vicidial_call(
    db_path,
    filename,
    row,
    audited,
    skipped_reason,
    agent_name=None,
):
    conn = sqlite3.connect(db_path)

    try:
        conn.execute(
        """
        INSERT OR REPLACE INTO vicidial_calls (
            recording_filename,
            recording_url,
            agent_user,
            agent_name,
            status,
            phone_number,
            recording_start,
            recording_date,
            duration_seconds,
            audited,
            skipped_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            row.get("recording_url"),
            row.get("agent_user") or row.get("user"),
            agent_name,
            row.get("status"),
            row.get("phone_number") or filename.replace(".mp3", "").split("_")[-1].replace("-all", ""),
            row.get("start_time") or row.get("call_datetime"),
            row.get("recording_date") or (row.get("call_datetime", "")[:10]),
            row.get("duration_seconds"),
            1 if audited else 0,
            skipped_reason,
        ),
    )

        conn.commit()

    except Exception as e:
        print("VICIDIAL DB SAVE ERROR:", repr(e))

    finally:
        conn.close()


from collections import Counter
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "calls.db"
CALLS_DIR = ROOT / "calls"
INCOMING_DIR = ROOT / "incoming_calls"


def api_call(base, user, password, params):
    full_params = {
        "source": "AIAUDITOR",
        "user": user,
        "pass": password,
        **params,
    }
    url = base + "?" + urllib.parse.urlencode(full_params)

    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace").strip()


def parse_recording_lookup(text):
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("ERROR:"):
            continue

        parts = line.split("|")
        if len(parts) < 5:
            continue

        call_datetime = parts[0]
        user = parts[1]
        recid = parts[2]
        lead_id = parts[3]
        duration_seconds = None

        if len(parts) >= 6 and parts[4].isdigit() and parts[5].startswith(("http://", "https://")):
            duration_seconds = int(parts[4])
            recording_url = parts[5]
        else:
            recording_url = parts[4]

        filename = Path(urllib.parse.urlparse(recording_url).path).name

        rows.append({
            "call_datetime": call_datetime,
            "user": user,
            "recid": recid,
            "lead_id": lead_id,
            "duration_seconds": duration_seconds,
            "recording_url": recording_url,
            "filename": filename,
            "call_name": Path(filename).stem,
        })

    return rows


def lead_status(base, user, password, lead_id):
    text = api_call(base, user, password, {
        "function": "lead_all_info",
        "lead_id": lead_id,
    })

    if text.startswith("ERROR:"):
        return "", text

    parts = text.split("|")
    status = parts[0].strip().upper() if parts else ""

    return status, text


def already_done(call_name, filename):
    report = ROOT / "reports" / f"{call_name}_report.txt"
    processed = ROOT / "processed_calls" / filename
    queued = CALLS_DIR / filename

    if report.exists() or processed.exists() or queued.exists():
        return True

    if DB_FILE.exists():
        conn = sqlite3.connect(DB_FILE)
        try:
            row = conn.execute(
                "SELECT id FROM calls WHERE call_name=? LIMIT 1",
                (call_name,),
            ).fetchone()
            if row:
                return True
        finally:
            conn.close()

    return False


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    dest.write_bytes(data)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Pull Vicidial recordings into AI Auditor queue.")
    parser.add_argument("--lead-id", help="Vicidial lead_id to pull recordings for")
    parser.add_argument("--agent-user", help="Vicidial agent user to pull recordings for")
    parser.add_argument("--date", required=True, help="Date folder for incoming_calls, e.g. 2026-05-12")
    parser.add_argument("--batch-limit", type=int, default=10)
    parser.add_argument("--min-seconds", type=int, default=0)
    parser.add_argument("--status", action="append", default=[], help="Allowed Vicidial status; repeatable")
    parser.add_argument("--exclude-status", action="append", default=[], help="Excluded Vicidial status; repeatable")
    parser.add_argument("--filename-prefix", default="")
    parser.add_argument("--queue", action="store_true", help="Also copy downloaded files into calls/")
    parser.add_argument("--summary-only", action="store_true", help="Only print status counts")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    base = os.getenv("VICIDIAL_API_URL")
    user = os.getenv("VICIDIAL_API_USER")
    password = os.getenv("VICIDIAL_API_PASS")

    if not base or not user or not password:
        raise SystemExit("Missing VICIDIAL_API_URL, VICIDIAL_API_USER, or VICIDIAL_API_PASS in .env")

    allowed_statuses = {s.strip().upper() for s in args.status if s.strip()}
    excluded_statuses = {
        "BOOTC",
        "TEST",
        "TEST1",
        "VM",
        "WN",
        "UI",
        "NIS",
        "HU",
        "DROP",
    }

    excluded_statuses.update(
        {s.strip().upper() for s in args.exclude_status if s.strip()}
    )

    if not args.lead_id and not args.agent_user:
        raise SystemExit("Provide either --lead-id or --agent-user")

    lookup_params = {
        "function": "recording_lookup",
    }

    if args.lead_id:
        lookup_params["lead_id"] = args.lead_id

    if args.agent_user:
        lookup_params["agent_user"] = args.agent_user
        lookup_params["date"] = args.date
        lookup_params["duration"] = "Y"

    lookup = api_call(base, user, password, lookup_params)

    rows = parse_recording_lookup(lookup)

    if not rows:
        print("No recordings found.")
        print(lookup)
        return

    queued = 0
    downloaded = 0
    skipped = 0
    status_counter = Counter()

    incoming_date_dir = INCOMING_DIR / args.date
    incoming_date_dir.mkdir(parents=True, exist_ok=True)
    CALLS_DIR.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if queued >= args.batch_limit:
            break

        filename = row["filename"]
        call_name = row["call_name"]
        row_date = row["call_datetime"][:10]

        if row_date != args.date:
            print(f"SKIP wrong date={row_date} {filename}")
            skipped += 1
            continue

        if args.filename_prefix and not filename.startswith(args.filename_prefix):
            skipped += 1
            continue

        duration_seconds = row.get("duration_seconds")
        if args.min_seconds > 0 and duration_seconds is not None and duration_seconds < args.min_seconds:
            print(f"SKIP short duration={duration_seconds}s {filename}")
            skipped += 1
            continue

        status, _raw = lead_status(base, user, password, row["lead_id"])
        row["status"] = status
        status_counter[status] += 1

        if allowed_statuses and status not in allowed_statuses:
            print(f"SKIP status={status} {filename}")
            skipped += 1
            continue

        if status == "ACT":
            duration_seconds = row.get("duration_seconds") or 0

            if duration_seconds < 60:
                print(f"SKIP short ACT duration={duration_seconds}s {filename}")

                save_vicidial_call(
                    DB_FILE,
                    filename,
                    row,
                    audited=False,
                    skipped_reason=f"short_act_{duration_seconds}s",
                )

                skipped += 1
                continue

        if excluded_statuses and status in excluded_statuses:
            print(f"SKIP excluded status={status} {filename}")

            save_vicidial_call(
                DB_FILE,
                filename,
                row,
                audited=False,
                skipped_reason=f"excluded_status_{status}",
            )

            skipped += 1
            continue

        if already_done(call_name, filename):
            print(f"SKIP already exists/done {filename}")

            save_vicidial_call(
                DB_FILE,
                filename,
                row,
                audited=True,
                skipped_reason="already_exists",
            )

            skipped += 1
            continue

        incoming_path = incoming_date_dir / filename

        if not incoming_path.exists():
            size = download(row["recording_url"], incoming_path)
            downloaded += 1
            print(f"DOWNLOADED status={status} size={size} {filename}")
        else:
            print(f"FOUND existing incoming status={status} {filename}")

            save_vicidial_call(
                DB_FILE,
                filename,
                row,
                audited=True,
                skipped_reason=None,
            )

        if args.queue:
            queue_path = CALLS_DIR / filename
            if not queue_path.exists():
                queue_path.write_bytes(incoming_path.read_bytes())
                queued += 1
                print(f"QUEUED status={status} {filename}")

    if args.summary_only:
        print("\nSTATUS SUMMARY")
        for status, count in sorted(status_counter.items()):
            print(f"{status}: {count}")

    print(f"DONE lead_id={args.lead_id} agent_user={args.agent_user} downloaded={downloaded} queued={queued} skipped={skipped}")


if __name__ == "__main__":
    main()

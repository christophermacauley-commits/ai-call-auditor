import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIRS = [
    ROOT / "calls",
    ROOT / "processed_calls",
    ROOT / "incoming_calls",
]


def main():
    parser = argparse.ArgumentParser(description="Delete local MP3 recordings older than retention days.")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "backup_golden_fixtures.py"), "--label", "before_audio_cleanup"],
            check=True,
        )

    cutoff = datetime.now() - timedelta(days=args.days)

    deleted = 0
    skipped = 0

    for folder in AUDIO_DIRS:
        if not folder.exists():
            continue

        for mp3 in folder.rglob("*.mp3"):
            modified = datetime.fromtimestamp(mp3.stat().st_mtime)

            if modified >= cutoff:
                skipped += 1
                continue

            print(f"{'WOULD DELETE' if args.dry_run else 'DELETE'} {mp3}")

            if not args.dry_run:
                mp3.unlink()

            deleted += 1

    print(f"DONE days={args.days} dry_run={args.dry_run} deleted={deleted} kept_recent={skipped}")


if __name__ == "__main__":
    main()

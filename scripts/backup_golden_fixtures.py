import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = ROOT / "golden_cases" / "golden18.json"

SOURCE_DIRS = [
    ROOT / "transcripts",
    ROOT / "transcripts_role_labeled",
    ROOT / "processed_transcripts",
    ROOT / "transcript_uploads",
    ROOT / "reports",
    ROOT / "golden_cases",
]

BACKUP_ROOT = ROOT / "golden_backups"


def load_golden_names():
    names = set()

    if GOLDEN_FILE.exists():
        data = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            for key in ("id", "match"):
                value = str(case.get(key, "") or "").strip()
                if value:
                    names.add(value)

    # Include real-call regression fixtures referenced directly by tests.
    test_file = ROOT / "tests" / "test_audit_guardrails.py"
    if test_file.exists():
        text = test_file.read_text(encoding="utf-8", errors="replace")
        import re
        names.update(re.findall(r'name\s*=\s*"([^"]+)"', text))
        names.update(re.findall(r'run_disposition_case\("([^"]+)"', text))
        names.update(re.findall(r'Path\("transcripts",\s*f"\{name\}\.txt"\)', text))

    return sorted(n for n in names if n)


def is_match(path, names):
    lower = path.name.lower()
    for name in names:
        n = name.lower()
        if lower == f"{n}.txt" or lower == f"{n}_report.txt":
            return True
        if lower.startswith(n) and (
            lower.endswith(".txt")
            or lower.endswith("_report.txt")
            or lower.endswith(".json")
            or lower.endswith(".mp3")
            or lower.endswith(".wav")
            or lower.endswith(".m4a")
        ):
            return True
    return path == GOLDEN_FILE


def main():
    parser = argparse.ArgumentParser(description="Back up golden/test fixture transcripts, reports, and expectations.")
    parser.add_argument("--label", default="", help="Optional label appended to backup folder name.")
    args = parser.parse_args()

    names = load_golden_names()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.label.strip()}" if args.label.strip() else ""
    backup_dir = BACKUP_ROOT / f"golden_backup_{stamp}{suffix}"

    copied = 0

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        for item in source_dir.rglob("*"):
            if not item.is_file():
                continue
            if not is_match(item, names):
                continue

            rel = item.relative_to(ROOT)
            dest = backup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied += 1

    manifest = backup_dir / "MANIFEST.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"Golden backup created: {stamp}\n"
        f"Golden names tracked: {len(names)}\n"
        f"Files copied: {copied}\n\n"
        + "\n".join(names)
        + "\n",
        encoding="utf-8",
    )

    print(f"Created {backup_dir}")
    print(f"Files copied: {copied}")


if __name__ == "__main__":
    main()

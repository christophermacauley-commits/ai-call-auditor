import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_MAP = ROOT / "training" / "agent_map.txt"

DATE = "2026-05-12"

overall = Counter()

for line in AGENT_MAP.read_text(encoding="utf-8").splitlines():
    line = line.strip()

    if not line or "=" not in line:
        continue

    agent_user, agent_name = line.split("=", 1)

    cmd = [
        sys.executable,
        "scripts/pull_vicidial_api.py",
        "--agent-user",
        agent_user.strip(),
        "--date",
        DATE,
        "--summary-only",
    ]

    print(f"\n=== {agent_user.strip()} | {agent_name.strip()} ===")

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    for output_line in result.stdout.splitlines():
        output_line = output_line.strip()

        if ":" not in output_line:
            continue

        if output_line.startswith("DONE"):
            continue

        try:
            status, count = output_line.split(":", 1)
            overall[status.strip()] += int(count.strip())
        except Exception:
            pass

print("\n==============================")
print("OVERALL STATUS TOTALS")
print("==============================")

for status, count in sorted(overall.items()):
    print(f"{status}: {count}")

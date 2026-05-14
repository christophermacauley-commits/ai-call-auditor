import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_MAP = ROOT / "training" / "agent_map.txt"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pull eligible Vicidial closer calls for all mapped agents.")
    parser.add_argument("--date", required=True, help="Date to pull, e.g. 2026-05-12")
    parser.add_argument("--batch-limit", type=int, default=10)
    parser.add_argument("--queue", action="store_true")
    args = parser.parse_args()

    total_agents = 0

    for line in AGENT_MAP.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or "=" not in line:
            continue

        agent_user, agent_name = line.split("=", 1)
        agent_user = agent_user.strip()
        agent_name = agent_name.strip()
        total_agents += 1

        print(f"\n=== {agent_user} | {agent_name} ===")

        cmd = [
            sys.executable,
            "scripts/pull_vicidial_api.py",
            "--agent-user",
            agent_user,
            "--date",
            args.date,
            "--batch-limit",
            str(args.batch_limit),
        ]

        if args.queue:
            cmd.append("--queue")

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

    print(f"\nDONE team pull date={args.date} agents_checked={total_agents} queue={args.queue}")


if __name__ == "__main__":
    main()

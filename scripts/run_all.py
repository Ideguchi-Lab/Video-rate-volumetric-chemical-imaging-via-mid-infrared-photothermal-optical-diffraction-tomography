"""Run every figures/*/run.py in order and report failures.

Usage:
    uv run scripts/run_all.py            # everything
    uv run scripts/run_all.py --plot-only
    uv run scripts/run_all.py fig1 fig3  # subset by prefix
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    extra = [a for a in sys.argv[1:] if a.startswith("--")]
    prefixes = [a for a in sys.argv[1:] if not a.startswith("--")]

    runs = sorted(
        p for p in (ROOT / "figures").glob("*/run.py")
        if not p.parent.name.startswith("_")
        and (not prefixes or any(p.parent.name.startswith(pre) for pre in prefixes))
    )
    failed = []
    for run in runs:
        print(f"=== {run.parent.name} ===", flush=True)
        rc = subprocess.call([sys.executable, str(run), *extra], cwd=ROOT)
        if rc != 0:
            failed.append(run.parent.name)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("all figures reproduced")


if __name__ == "__main__":
    main()

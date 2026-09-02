"""Create a new figure directory from figures/_template.

Usage:
    uv run scripts/new_figure.py fig4_spectra --label "Fig. 4" --title "Hyperspectral imaging"
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "figures" / "_template"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", help="directory name, e.g. fig4_spectra or figS2_drift")
    p.add_argument("--label", default=None, help='caption label, e.g. "Fig. 4"')
    p.add_argument("--title", default="<short title>")
    args = p.parse_args()

    m = re.match(r"(fig[S]?\d+)_", args.name)
    if not m:
        raise SystemExit("name must look like fig<N>_<slug> or figS<N>_<slug>")
    fig_id = m.group(1)
    label = args.label or fig_id.replace("figS", "Supplementary Fig. ").replace("fig", "Fig. ")

    dest = ROOT / "figures" / args.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    shutil.copytree(TEMPLATE, dest)

    for f in ("README.md", "config.yaml", "run.py"):
        path = dest / f
        text = path.read_text()
        text = text.replace("Fig. X — <short title>", f"{label} — {args.title}")
        text = text.replace("Reproduce Fig. X.", f"Reproduce {label}: {args.title}.")
        text = text.replace("figX_<name>", args.name)
        text = text.replace("figure_id: figX", f"figure_id: {fig_id}")
        text = text.replace("figX/raw", f"{fig_id}/raw")
        path.write_text(text)
    print(f"created {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

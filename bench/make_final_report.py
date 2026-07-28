"""Generate helios_report.pdf -- the complete project record.

Reuses the artifact loading, chart rendering and layout helpers from
make_report.py; the content lives in final_report_content.py.

    python bench/make_final_report.py --out docs/helios_report.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_report import (  # noqa: E402
    ACCENT,
    MUTED,
    RULE,
    Data,
    _Helpers,
    make_charts,
    provenance,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "helios_report.pdf"))
    args = ap.parse_args()

    from final_report_content import build_story, render

    d = Data()
    prov = provenance(d)
    print(f"commit {prov['commit']}, {prov['tests']} tests collected")
    if d.missing:
        print(f"  missing artifacts (noted in the report): {sorted(set(d.missing))}")

    figs = make_charts(d, ROOT / "artifacts" / "figures")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(build_story(d, figs, prov, _Helpers), out, prov, (MUTED, RULE, ACCENT))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

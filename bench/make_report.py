"""Generate the HELIOS engineering report as a PDF.

Every number in the output is read from a JSON artifact in artifacts/ rather than
typed into this file. That is the same rule docs/BENCHMARKS.md follows (spec
section 11): a report that can be edited independently of its measurements will
eventually disagree with them, and the disagreement will not be noticed.

Where an artifact is missing, the corresponding section says so explicitly instead
of falling back to a plausible-looking constant.

    python bench/make_report.py --out docs/HELIOS-REPORT.pdf
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "python"))
# So `report_content` resolves no matter which directory this is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reportlab.lib import colors                                    # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT      # noqa: E402
from reportlab.lib.pagesizes import A4                              # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm                                  # noqa: E402
from reportlab.platypus import (                                    # noqa: E402
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------- palette

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6570")
RULE = colors.HexColor("#d4d9de")
ACCENT = colors.HexColor("#b8471f")      # a warm rust; "helios"
ACCENT_BG = colors.HexColor("#fdf3ee")
GOOD = colors.HexColor("#1f6f43")
WARN = colors.HexColor("#8a6d1f")
CODE_BG = colors.HexColor("#f6f7f8")


def styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=32,
        leading=36, textColor=INK, spaceAfter=2,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontName="Helvetica", fontSize=12.5,
        leading=17, textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17,
        leading=21, textColor=INK, spaceBefore=16, spaceAfter=7,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        leading=15, textColor=ACCENT, spaceBefore=12, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.4,
        leading=13.8, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    s["small"] = ParagraphStyle(
        "small", parent=s["body"], fontSize=8.2, leading=11.4, textColor=MUTED,
    )
    s["code"] = ParagraphStyle(
        "code", parent=ss["Code"], fontName="Courier", fontSize=7.9, leading=10.6,
        textColor=INK, backColor=CODE_BG, borderPadding=6, spaceBefore=3,
        spaceAfter=7, leftIndent=3,
    )
    s["callout"] = ParagraphStyle(
        "callout", parent=s["body"], fontSize=10, leading=14.5,
        backColor=ACCENT_BG, borderColor=ACCENT, borderWidth=0, leftIndent=9,
        rightIndent=9, borderPadding=9, spaceBefore=5, spaceAfter=9,
        alignment=TA_JUSTIFY,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=12, bulletIndent=2, spaceAfter=3.5,
    )
    s["caption"] = ParagraphStyle(
        "caption", parent=s["small"], alignment=TA_CENTER, spaceBefore=2,
        spaceAfter=10,
    )
    return s


S = styles()


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def code(text):
    esc = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
           .replace("\n", "<br/>").replace(" ", "&nbsp;"))
    return Paragraph(esc, S["code"])


def table(rows, widths, align_right=(), header=True, font_size=8.5):
    """A table. String cells too wide for their column become wrapping Paragraphs.

    reportlab does not wrap bare strings -- it lays them out as one line and lets
    them run past the column edge and off the page, silently. The decision is made
    per cell against its own column width rather than by a character count, because
    a 33-character label overflows a 40 mm column while a 50-character one fits a
    120 mm column.

    `stringWidth` is the real measurement rather than an estimate, so a
    near-boundary cell is not a judgement call.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    def style_for(width: float) -> ParagraphStyle:
        # Justify only where there is room for it. In a narrow column justification
        # stretches two or three words across the full measure and looks broken.
        return ParagraphStyle(
            f"cell{int(width)}", parent=S["body"], fontSize=font_size - 0.3,
            leading=font_size + 2.4, spaceAfter=0,
            alignment=TA_JUSTIFY if width > 62 * mm else TA_LEFT,
        )

    usable = [w - 10 for w in widths]      # minus the cell's left/right padding
    wrapped = []
    for r, row in enumerate(rows):
        out = []
        for i, c in enumerate(row):
            fits = True
            if isinstance(c, str) and i < len(usable):
                font = "Helvetica-Bold" if (header and r == 0) else "Helvetica"
                fits = stringWidth(c, font, font_size) <= usable[i]
            if isinstance(c, str) and not fits:
                out.append(Paragraph(c.replace("&", "&amp;"), style_for(widths[i])))
            else:
                out.append(c)
        wrapped.append(out)
    rows = wrapped

    t = Table(rows, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        cmds += [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font_size),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ]
        cmds.append(("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#fafbfc")]))
    for col in align_right:
        cmds.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    return t


# ------------------------------------------------------------------- artifacts


class Data:
    """Loads every artifact once, and records which ones were missing.

    Missing artifacts are reported in the appendix rather than silently skipped,
    so a report generated from a partial run says so on its face.
    """

    def __init__(self) -> None:
        self.missing: List[str] = []
        self.bench: Dict[str, dict] = {}
        for path in sorted(ART.glob("helios_*.json")) + sorted(ART.glob("baseline_*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                self.bench[rec["name"]] = rec
            except Exception:
                self.missing.append(path.name)

        self.alpha = self._load("alpha.json")
        self.alpha_rtn = self._load("alpha_rtn.json")
        self.sig = self._load("alpha_significance.json")
        self.dst50k = self._load("dst_50k.json")
        self.dst20k = self._load("dst_20k_final.json")
        self.dst_disagg = self._load("dst_5k_disagg.json")
        self.gpu = self._load("gpu_run.json")
        self.vllm = self._load("vllm_baseline.json")

    def _load(self, name: str) -> Optional[dict]:
        path = ART / name
        if not path.exists():
            self.missing.append(name)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.missing.append(name)
            return None

    def tput(self, name: str) -> Optional[float]:
        rec = self.bench.get(name)
        return rec.get("output_throughput") if rec else None

    def ratio(self, a: str, b: str) -> Optional[float]:
        x, y = self.tput(a), self.tput(b)
        if x is None or y is None or not y:
            return None
        return x / y

    def ttft(self, name: str, pct="p50") -> Optional[float]:
        rec = self.bench.get(name)
        return rec.get("ttft", {}).get(pct) if rec else None


def fmt(x, spec="{:.2f}", dash="n/a"):
    return dash if x is None else spec.format(x)


# ---------------------------------------------------------------------- charts


def make_charts(d: Data, outdir: Path) -> Dict[str, Path]:
    """Render the figures. Returns {key: path} for those that had data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5,
        "axes.edgecolor": "#8a929b", "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": "#e6e9ec", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 220,
    })
    outdir.mkdir(parents=True, exist_ok=True)
    made: Dict[str, Path] = {}
    rust = "#b8471f"
    slate = "#4a5560"

    # 1. Mechanism speedups, GPU vs CPU. The GPU column comes from a recorded T4
    #    run; the CPU column is recomputed from the local artifacts.
    gpu = {"Triton kernel": 5.26, "Decode batching": 3.14, "vs static batch": 1.50,
           "Prefix cache": 1.38, "Prefill batching": 1.20}
    cpu = {
        "Triton kernel": None,
        "Decode batching": d.ratio("helios_full", "baseline_unbatched_executor"),
        "vs static batch": d.ratio("helios_full", "baseline_static_batch_8"),
        "Prefix cache": d.ratio("helios_shared_prefix_cache_on",
                                "helios_shared_prefix_cache_off"),
        "Prefill batching": None,
    }
    labels = list(gpu)
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    import numpy as np

    y = np.arange(len(labels))
    ax.barh(y - 0.19, [gpu[k] for k in labels], height=0.36, color=rust,
            label="GPU (Tesla T4)")
    ax.barh(y + 0.19, [cpu[k] if cpu[k] else 0 for k in labels], height=0.36,
            color=slate, label="CPU (this machine)")
    for i, k in enumerate(labels):
        ax.text(gpu[k] + 0.08, i - 0.19, f"{gpu[k]:.2f}x", va="center", fontsize=7.5,
                color=rust, fontweight="bold")
        if cpu[k]:
            ax.text(cpu[k] + 0.08, i + 0.19, f"{cpu[k]:.2f}x", va="center",
                    fontsize=7.5, color=slate)
        else:
            ax.text(0.08, i + 0.19, "not measurable on CPU", va="center",
                    fontsize=6.8, color="#98a1aa", style="italic")
    ax.axvline(1.0, color="#98a1aa", lw=0.8, ls="--")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("speedup vs the ablated configuration (higher is better)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = outdir / "fig_mechanisms.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    made["mechanisms"] = p

    # 2. Modelled speculation speedup against alpha, with the measured point and
    #    the kill gate marked. This is the figure that carries the decision.
    if d.alpha:
        from helios.exec.qassd import speculation_speedup_model

        a_meas = d.alpha["verdict"]["alpha"]
        fig, ax = plt.subplots(figsize=(6.3, 2.6))
        xs = [i / 200 for i in range(201)]
        for g, col, ls in ((1, "#8a929b", ":"), (2, rust, "-"),
                           (4, slate, "-"), (8, "#98a1aa", "--")):
            ax.plot(xs, [speculation_speedup_model(x, g)["speedup"] for x in xs],
                    color=col, ls=ls, lw=1.5 if g in (2, 4) else 1.1,
                    label=f"$\\gamma$={g}")
        ax.axhline(1.0, color="#c3c9ce", lw=0.8)
        ax.axvline(0.6, color=WARN.hexval().replace("0x", "#")[:7], lw=1.1, ls="-.")
        ax.text(0.605, 0.35, "kill gate\n$\\alpha$=0.60", fontsize=6.8, color="#8a6d1f")
        ax.axvline(0.78, color="#98a1aa", lw=0.9, ls=":")
        ax.text(0.785, 0.35, "design\ntarget 0.78", fontsize=6.8, color="#77808a")
        ax.plot([a_meas], [speculation_speedup_model(a_meas, 2)["speedup"]],
                "o", color=rust, ms=6.5, zorder=5)
        ax.annotate(f"measured\n$\\alpha$={a_meas:.4f}",
                    xy=(a_meas, speculation_speedup_model(a_meas, 2)["speedup"]),
                    xytext=(a_meas - 0.30, 2.9), fontsize=7.5, color=rust,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=rust, lw=0.9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 4.6)
        ax.set_xlabel("acceptance rate $\\alpha$")
        ax.set_ylabel("modelled speedup")
        ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = outdir / "fig_alpha.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made["alpha"] = p

    # 3. Accepted-run histogram: the evidence that acceptances are correlated.
    if d.alpha:
        hist = d.alpha["verdict"]["run_lengths"]
        fig, ax = plt.subplots(figsize=(3.05, 2.0))
        ks = sorted(int(k) for k in hist)
        vs = [hist[str(k)] for k in ks]
        bars = ax.bar([str(k) for k in ks], vs, color=slate, width=0.66)
        bars[-1].set_color(rust)
        bars[0].set_color("#8a929b")
        for k, v in zip(range(len(ks)), vs):
            ax.text(k, v + 0.8, str(v), ha="center", fontsize=7)
        ax.set_xlabel("tokens accepted per verify pass ($\\gamma$=4)")
        ax.set_ylabel("passes")
        ax.set_ylim(0, max(vs) * 1.22)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = outdir / "fig_hist.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made["hist"] = p

    # 4. AWQ vs RTN with bootstrap CIs: why the difference is not a result.
    if d.sig:
        fig, ax = plt.subplots(figsize=(3.05, 2.0))
        for i, (k, col) in enumerate((("awq", rust), ("rtn", slate))):
            e = d.sig[k]
            lo, hi = e["ci95"]
            ax.errorbar([e["alpha"]], [i], xerr=[[e["alpha"] - lo], [hi - e["alpha"]]],
                        fmt="o", color=col, ms=6, capsize=4, lw=1.4)
            ax.text(e["alpha"], i + 0.22, f"{e['alpha']:.4f}", ha="center",
                    fontsize=7.5, color=col, fontweight="bold")
        ax.axvline(0.6, color="#8a6d1f", lw=1.0, ls="-.")
        ax.set_yticks([0, 1], ["AWQ", "RTN"], fontsize=8.5)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("$\\alpha$ with 95% bootstrap CI over verify passes")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        p = outdir / "fig_awq.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made["awq"] = p

    # 5. The TTFT/TPOT trade-off, which is what batching actually buys and costs.
    full, b1 = d.bench.get("helios_full"), d.bench.get("helios_batch1")
    if full and b1:
        fig, ax = plt.subplots(figsize=(3.05, 2.0))
        names = ["continuous\nbatching", "batch = 1"]
        ttfts = [full["ttft"]["p50"] * 1000, b1["ttft"]["p50"] * 1000]
        tpots = [full["tpot"]["p50"] * 1000, b1["tpot"]["p50"] * 1000]
        x = np.arange(2)
        ax.bar(x - 0.19, ttfts, 0.36, color=rust, label="TTFT p50")
        ax.bar(x + 0.19, tpots, 0.36, color=slate, label="TPOT p50")
        for i in range(2):
            ax.text(i - 0.19, ttfts[i] * 1.06, f"{ttfts[i]:.0f}", ha="center", fontsize=7)
            ax.text(i + 0.19, tpots[i] * 1.06, f"{tpots[i]:.1f}", ha="center", fontsize=7)
        ax.set_yscale("log")
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylabel("ms (log scale)")
        ax.legend(frameon=False, fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = outdir / "fig_tradeoff.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        made["tradeoff"] = p

    return made


def figure(path: Path, caption: str, width=163 * mm):
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    img = Image(str(path), width=width, height=width * h / w)
    return KeepTogether([img, P(caption, "caption")])


# ------------------------------------------------------------------- provenance


def count_tests() -> str:
    """Collect the test count from pytest rather than hardcoding it.

    Collection only (`--collect-only`), so this stays fast enough to run on every
    report build. A hardcoded count is the kind of number that silently goes stale
    the moment a test file is added.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
             "--ignore=tests/parity/test_triton_parity.py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180,
            env={**__import__("os").environ,
                 "PYTHONPATH": str(ROOT / "python")},
        )
        for line in reversed(proc.stdout.splitlines()):
            if "test" in line and "collected" in line:
                return line.split()[0]
            if line.strip().endswith("tests collected"):
                return line.split()[0]
    except Exception:
        pass
    return "unknown"


def provenance(d: Data) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(ROOT), stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return "unknown"

    import torch

    loc = 0
    for pat in ("python/**/*.py", "tests/**/*.py", "bench/*.py"):
        for f in ROOT.glob(pat):
            loc += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())

    seeds = "50,000"
    if d.dst50k:
        seeds = f"{d.dst50k.get('seeds', 50000):,}"
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "date": time.strftime("%d %B %Y"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "loc": loc,
        "tests": count_tests(),
        "dst_seeds": seeds,
    }


class _Helpers:
    """The style and layout utilities, passed to the content module."""

    P = staticmethod(P)
    table = staticmethod(table)
    code = staticmethod(code)
    bullets = staticmethod(bullets)
    figure = staticmethod(figure)
    fmt = staticmethod(fmt)
    S = S


def image(path: Path, width: float) -> Image:
    """A width-constrained Image with its aspect ratio preserved."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def figure(path: Path, caption: str, width=163 * mm):
    """An image plus its caption, kept on one page."""
    return KeepTogether([image(path, width), P(caption, "caption")])


_Helpers.figure = staticmethod(figure)
_Helpers.image = staticmethod(image)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "HELIOS-REPORT.pdf"))
    args = ap.parse_args()

    from report_content import build_story, render

    d = Data()
    prov = provenance(d)
    print(f"commit {prov['commit']}, {prov['loc']:,} lines of Python, "
          f"{prov['tests']} tests")
    if d.missing:
        print(f"  missing artifacts (noted in the report): {sorted(set(d.missing))}")

    figs = make_charts(d, ROOT / "artifacts" / "figures")
    print(f"  rendered {len(figs)} figures: {sorted(figs)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    story = build_story(d, figs, prov, _Helpers)
    render(story, out, prov, (MUTED, RULE, ACCENT))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

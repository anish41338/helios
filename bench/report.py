"""Generate docs/BENCHMARKS.md from raw JSON artifacts.

Spec section 11: "Every number in docs/BENCHMARKS.md is generated from raw JSON
artifacts by bench/report.py. Hand-editing that file is forbidden. Include the
exact command, commit SHA, driver/CUDA version, and clock/power state."

Accordingly this script emits the provenance block first and refuses to invent
comparisons the artifacts do not contain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

HEADER = """# HELIOS benchmarks

> **GENERATED FILE -- DO NOT EDIT.**
> Produced by `bench/report.py` from the raw JSON artifacts in `artifacts/`.
> Regenerate with the command shown under Provenance.

## What these numbers are, and are not

These are measurements from **this** repository on **this** machine, using a
small randomly-initialised model. They characterise the *engine's scheduling
behaviour* -- the relative cost of paging, batching, chunked prefill, and the
prefix cache -- and nothing else.

They are **not**:

- a comparison against vLLM, TensorRT-LLM, or any other engine (no such
  baseline was run; see `docs/SCOPE.md`)
- evidence about GPU performance (this build ran on CPU; there are no CUDA or
  Triton kernels)
- evidence about quantized inference, QASSD acceptance rates, or
  prefill/decode disaggregation (not implemented -- see `docs/SCOPE.md`)
- meaningful as absolute tokens/second, since the model is a toy with random
  weights

Per spec section 19.6, no figure here is taken from a paper, and per section
19.3 every figure traces to a JSON artifact listed below.
"""


def load_results(artifact_dir: Path) -> List[Dict]:
    results = []
    for path in sorted(artifact_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
        if "name" in data and "output_throughput" in data:
            data["_path"] = path.name
            results.append(data)
    return results


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(d: Dict, key: str, digits: int = 4) -> str:
    return _fmt(d.get(key), digits)


def render(results: List[Dict], command: str) -> str:
    if not results:
        return HEADER + "\nNo artifacts found. Run `bench/loadgen.py` first.\n"

    out = [HEADER]

    env = results[0].get("env", {})
    out.append("## Provenance\n")
    out.append(f"- commit: `{env.get('commit', 'unknown')}`")
    out.append(f"- timestamp: {env.get('timestamp', 'unknown')}")
    out.append(f"- platform: {env.get('platform', 'unknown')}")
    out.append(f"- processor: `{env.get('processor', 'unknown')}`")
    out.append(f"- python: {env.get('python', 'unknown')}")
    out.append(f"- torch: {env.get('torch', 'unknown')}")
    out.append(
        f"- device: **{env.get('device', 'unknown')}** "
        f"(cuda_available={env.get('cuda_available')})"
    )
    out.append(f"- regenerate: `{command}`")
    out.append("")
    out.append(
        "> Clock/power state is not pinned on this machine, so run-to-run "
        "variation of a few percent is expected. Differences smaller than that "
        "should not be interpreted."
    )
    out.append("")

    ablations = [r for r in results if r["name"].startswith("helios_")]
    baselines = [r for r in results if r["name"].startswith("baseline_")]

    if ablations:
        out.append("## Ablations\n")
        out.append(
            "Each row differs from `full` by one mechanism, which is what makes "
            "the delta attributable (spec section 11).\n"
        )
        out.append(
            "| configuration | reqs | out tok/s | TTFT p50 | TTFT p95 | "
            "TPOT p50 | TPOT p95 | e2e p50 | goodput |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(ablations, key=lambda x: x["name"]):
            name = r["name"].replace("helios_", "")
            out.append(
                f"| `{name}` | {r['completed']}/{r['num_requests']} "
                f"| {r['output_throughput']:.1f} "
                f"| {_pct(r['ttft'], 'p50')} | {_pct(r['ttft'], 'p95')} "
                f"| {_pct(r['tpot'], 'p50')} | {_pct(r['tpot'], 'p95')} "
                f"| {_pct(r['e2e'], 'p50')} "
                f"| {r.get('goodput_ratio', 0.0):.2f} |"
            )
        out.append("")

    if baselines:
        out.append("## Baselines\n")
        out.append(
            "`baseline_hf_loop` re-runs the whole sequence per token with no "
            "cache reuse -- the 'no engine at all' reference. "
            "`baseline_static_batch_N` drains each batch fully before admitting "
            "the next, which is the head-of-line waste continuous batching "
            "removes.\n"
        )
        out.append(
            "> The naive loop runs a deliberately smaller workload (it is "
            "quadratic in sequence length), so its throughput is **not** "
            "directly comparable to the rows above. Compare mechanisms, not "
            "absolute numbers across different workloads."
        )
        out.append("")
        out.append(
            "| baseline | reqs | out tok/s | TTFT p50 | TPOT p50 | e2e p50 | workload |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for r in sorted(baselines, key=lambda x: x["name"]):
            w = r.get("workload", {})
            desc = (
                f"n={w.get('num_requests')} plen~{w.get('prompt_len_mean')} "
                f"olen~{w.get('output_len_mean')}"
            )
            out.append(
                f"| `{r['name']}` | {r['completed']}/{r['num_requests']} "
                f"| {r['output_throughput']:.1f} "
                f"| {_pct(r['ttft'], 'p50')} | {_pct(r['tpot'], 'p50')} "
                f"| {_pct(r['e2e'], 'p50')} | {desc} |"
            )
        out.append("")

    # Scheduler counters give the mechanistic explanation for the timings.
    full = next((r for r in ablations if r["name"] == "helios_full"), None)
    if full and full.get("scheduler_stats"):
        out.append("## Scheduler counters (`full` configuration)\n")
        out.append("| metric | value |")
        out.append("|---|---|")
        for key, value in sorted(full["scheduler_stats"].items()):
            if isinstance(value, float):
                value = f"{value:.4f}"
            out.append(f"| `{key}` | {value} |")
        out.append("")

    out.append("## Workload\n")
    w = (ablations or baselines)[0].get("workload", {})
    out.append("| parameter | value |")
    out.append("|---|---|")
    for key in sorted(w):
        out.append(f"| `{key}` | {w[key]} |")
    out.append("")

    out.append("## Source artifacts\n")
    for r in sorted(results, key=lambda x: x["_path"]):
        out.append(f"- `artifacts/{r['_path']}` -> `{r['name']}`")
    out.append("")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="docs/BENCHMARKS.md")
    args = ap.parse_args()

    artifact_dir = Path(args.artifacts)
    results = load_results(artifact_dir)
    command = f"python bench/report.py --artifacts {args.artifacts} --out {args.out}"
    text = render(results, command)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} from {len(results)} artifact(s)")


if __name__ == "__main__":
    main()

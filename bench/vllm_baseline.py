"""vLLM baseline on identical hardware, model, and workload (spec section 11).

STATUS: written, NOT RUN. vLLM requires CUDA and this was built on a CPU-only
machine. Nothing in this repository claims a comparison against vLLM, and nothing
should until this script has been run and its JSON committed. Spec section 19.3.

The point of a baseline
----------------------
Every number in docs/BENCHMARKS.md is a HELIOS-versus-HELIOS ablation: a
mechanism switched off, nothing else changed. Those are the right measurements for
attributing a speedup to a mechanism, and they are worth more than a
cross-engine comparison for that purpose -- but they cannot answer "is this
engine any good", because a slow engine's ablations are just as clean as a fast
one's.

This script answers that question, and it is designed to be able to answer it
unfavourably. That matters: an honest baseline is one where losing is a possible
outcome you have pre-committed to reporting.

What a fair comparison requires
-------------------------------
Getting this wrong is the usual way baselines mislead, so the controls are
explicit:

  * **same model directory** -- not "an equivalent model";
  * **same dtype**;
  * **same KV budget.** vLLM sizes its cache as a fraction of total GPU memory
    (`gpu_memory_utilization`); HELIOS takes an absolute byte budget. Left
    unmatched, one engine gets more blocks and therefore more concurrency, and
    the comparison measures the memory split rather than the engine. This script
    derives vLLM's fraction from HELIOS's byte budget;
  * **same arrival process** -- the same Poisson trace with the same seed, replayed
    against both, not two independently generated traces;
  * **same sampling parameters**, greedy, so neither engine benefits from a
    different temperature path;
  * **warm-up discarded** for both. A first-run lazy-import cost already produced
    one wrong conclusion in this project (docs/BENCHMARKS.md records it), and
    vLLM's CUDA graph capture makes its first request dramatically slower still;
  * **no speculative decoding on either side**, since the two implementations
    differ in ways that would not be attributable.

What this cannot control for, and must be said alongside any result: vLLM ships
hand-written CUDA and Triton kernels, CUDA graphs, and a C++/Rust-adjacent hot
path. HELIOS is Python with one Triton kernel. If HELIOS loses on absolute
throughput that is the expected outcome and not a defect in the design -- the
interesting quantities are the *scaling* behaviour (how throughput responds to
concurrency) and the goodput under an SLO, where scheduling policy shows up.

Usage (on a CUDA machine, with vLLM installed):

    pip install vllm
    python bench/vllm_baseline.py --model artifacts/qwen05b --requests 64
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def preflight() -> tuple[bool, str]:
    """Refuse to produce numbers that would not mean anything.

    Fails loudly rather than degrading, because a baseline that quietly ran on
    CPU, or against a differently-sized cache, would produce a plausible number
    that is not a comparison.
    """
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, (
            "no CUDA device. vLLM requires one, and a CPU-only run would compare "
            "HELIOS-on-CPU against nothing. Use a GPU session (see docs/GPU.md)."
        )
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False, "vLLM is not installed. Run: pip install vllm"
    return True, "ok"


def build_trace(n: int, seed: int, rate: float, prompt_lo: int, prompt_hi: int,
                max_tokens: int, vocab_size: int):
    """One arrival trace, replayed against both engines.

    Generated from a seed and returned as data rather than driven directly, so
    both engines see byte-identical prompts arriving at identical offsets. Two
    separately generated traces with the same distribution would differ enough to
    swamp the effect being measured at these sample sizes.
    """
    rng = random.Random(seed)
    trace = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        prompt_len = rng.randint(prompt_lo, prompt_hi)
        # Token ids only, so the comparison does not depend on either engine's
        # tokenizer handling. Kept clear of the low special-token range.
        ids = [10 + rng.randrange(max(1, vocab_size - 20)) for _ in range(prompt_len)]
        trace.append({"index": i, "arrival_s": t, "token_ids": ids,
                      "max_tokens": max_tokens})
    return trace


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p / 100.0))]


def run_helios(model_dir: str, trace, kv_bytes: int, max_num_seqs: int):
    from helios.core.types import SamplingParams
    from helios.engine import EngineConfig, LLMEngine

    engine = LLMEngine(
        EngineConfig(
            model_dir=model_dir, device="cuda", kv_cache_bytes=kv_bytes,
            max_num_seqs=max_num_seqs, block_size=16,
            enable_prefix_cache=True, enable_chunked_prefill=True,
        )
    )
    params_for = lambda n: SamplingParams(max_tokens=n, temperature=0.0)

    # Warm-up, discarded.
    engine.add_request("warmup", trace[0]["token_ids"], params_for(4))
    engine.run_until_complete(max_steps=1000)

    start = time.perf_counter()
    pending = list(trace)
    submitted = 0
    first_seen: dict[str, float] = {}
    done: dict[str, float] = {}

    while pending or engine.scheduler.has_work:
        now = time.perf_counter() - start
        while pending and pending[0]["arrival_s"] <= now:
            item = pending.pop(0)
            rid = f"req-{item['index']}"
            engine.add_request(rid, item["token_ids"], params_for(item["max_tokens"]))
            submitted += 1
        for out in engine.step():
            done[out.request_id] = time.perf_counter() - start
        if not engine.scheduler.has_work and pending:
            time.sleep(min(0.002, max(0.0, pending[0]["arrival_s"] - now)))
    elapsed = time.perf_counter() - start

    metrics = {m.request_id: m for m in engine.metrics()}
    ttfts = [m.ttft for m in metrics.values() if m.ttft is not None]
    tpots = [m.tpot for m in metrics.values() if m.tpot is not None]
    out_tokens = sum(m.output_tokens for m in metrics.values())
    return {
        "engine": "helios",
        "requests": submitted,
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "output_tok_per_s": out_tokens / elapsed if elapsed else 0.0,
        "ttft_p50": percentile(ttfts, 50), "ttft_p95": percentile(ttfts, 95),
        "tpot_p50": percentile(tpots, 50), "tpot_p95": percentile(tpots, 95),
        "kv_blocks": engine.num_blocks,
    }


def run_vllm(model_dir: str, trace, kv_bytes: int, max_num_seqs: int, dtype: str):
    """vLLM's offline LLM API, driven with the same trace.

    `gpu_memory_utilization` is derived so vLLM's KV cache matches HELIOS's byte
    budget as closely as its interface allows. This is the single most important
    control and the easiest to get wrong: vLLM defaults to 0.90, which on a 16 GB
    card is many times a 2 GB HELIOS budget, and the resulting concurrency
    difference would dominate every latency number.
    """
    import torch
    from vllm import LLM, SamplingParams as VSamplingParams

    total = torch.cuda.get_device_properties(0).total_memory
    # Weights + activations still have to fit, so the fraction is (weights + KV)
    # over total. Estimated from the checkpoint size on disk.
    weight_bytes = sum(
        p.stat().st_size for p in Path(model_dir).glob("*.safetensors")
    )
    util = min(0.92, (weight_bytes * 1.25 + kv_bytes) / total)

    llm = LLM(
        model=model_dir, dtype=dtype, gpu_memory_utilization=util,
        max_num_seqs=max_num_seqs, enable_prefix_caching=True,
        enforce_eager=False, disable_log_stats=True,
    )

    # Warm-up, discarded, and it matters more here: CUDA graph capture and
    # autotuning make vLLM's first request an outlier by a wide margin.
    llm.generate(
        prompt_token_ids=[trace[0]["token_ids"]],
        sampling_params=VSamplingParams(max_tokens=4, temperature=0.0),
    )

    # vLLM's offline API has no arrival-time hook, so the whole trace is
    # submitted at once. Stated as a limitation rather than papered over: it
    # gives vLLM a *scheduling advantage* (it sees the full queue immediately and
    # can pack batches optimally), so a HELIOS win here would be conservative and
    # a HELIOS loss is partly attributable to this. The AsyncLLMEngine path would
    # allow true Poisson arrivals and is the right follow-up.
    start = time.perf_counter()
    outputs = llm.generate(
        prompt_token_ids=[t["token_ids"] for t in trace],
        sampling_params=[
            VSamplingParams(max_tokens=t["max_tokens"], temperature=0.0)
            for t in trace
        ],
    )
    elapsed = time.perf_counter() - start

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "engine": "vllm",
        "requests": len(outputs),
        "elapsed_s": elapsed,
        "output_tokens": out_tokens,
        "output_tok_per_s": out_tokens / elapsed if elapsed else 0.0,
        "gpu_memory_utilization": util,
        "arrival_model": "all-at-once (offline API); see the note in run_vllm",
        "ttft_p50": None, "ttft_p95": None, "tpot_p50": None, "tpot_p95": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--rate", type=float, default=8.0, help="arrivals per second")
    ap.add_argument("--prompt-lo", type=int, default=64)
    ap.add_argument("--prompt-hi", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--kv-mb", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--skip-vllm", action="store_true",
        help="run only the HELIOS side (useful for checking the harness itself)",
    )
    args = ap.parse_args()

    ok, why = preflight()
    if not ok and not args.skip_vllm:
        print(f"PREFLIGHT FAILED: {why}")
        print(
            "\nNo baseline number is produced. docs/SCOPE.md lists the vLLM\n"
            "comparison as NOT built precisely so this stays true until the\n"
            "script has actually run somewhere it can."
        )
        return 1

    import json as _json

    with open(Path(args.model) / "config.json", encoding="utf-8") as f:
        vocab_size = _json.load(f)["vocab_size"]

    kv_bytes = args.kv_mb * 1024 * 1024
    trace = build_trace(
        args.requests, args.seed, args.rate, args.prompt_lo, args.prompt_hi,
        args.max_tokens, vocab_size,
    )
    total_prompt = sum(len(t["token_ids"]) for t in trace)
    print(
        f"trace: {len(trace)} requests, {total_prompt} prompt tokens, "
        f"seed={args.seed}, rate={args.rate}/s\n"
        f"controls: same model dir, dtype={args.dtype}, "
        f"kv_budget={args.kv_mb} MiB, max_num_seqs={args.max_num_seqs}, greedy\n"
    )

    results = []
    print("running HELIOS ...", flush=True)
    h = run_helios(args.model, trace, kv_bytes, args.max_num_seqs)
    results.append(h)
    print(f"  {h['output_tok_per_s']:.1f} out tok/s, "
          f"ttft p50 {h['ttft_p50'] * 1000:.1f} ms\n")

    if not args.skip_vllm:
        print("running vLLM ...", flush=True)
        v = run_vllm(args.model, trace, kv_bytes, args.max_num_seqs, args.dtype)
        results.append(v)
        print(f"  {v['output_tok_per_s']:.1f} out tok/s\n")

        ratio = h["output_tok_per_s"] / max(1e-9, v["output_tok_per_s"])
        print(f"HELIOS / vLLM throughput ratio: {ratio:.2f}x")
        if ratio < 1.0:
            print(
                f"  vLLM is {1 / ratio:.2f}x faster. Report this as measured.\n"
                "  It is the expected direction: vLLM ships hand-written CUDA\n"
                "  kernels and CUDA graphs against this build's Python hot path.\n"
                "  The defensible HELIOS claims are its ablations and its\n"
                "  scheduler verification, not absolute throughput."
            )
        else:
            print(
                "  HELIOS is ahead on this workload. Before claiming that, check\n"
                "  the KV budgets and max_num_seqs in the JSON actually matched,\n"
                "  and that vLLM was not memory-starved by the derived\n"
                "  gpu_memory_utilization -- that is the most likely way this\n"
                "  comparison flatters HELIOS."
            )

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "controls": {
            "dtype": args.dtype, "kv_mib": args.kv_mb,
            "max_num_seqs": args.max_num_seqs, "seed": args.seed,
            "rate_per_s": args.rate, "greedy": True,
            "speculation": "disabled on both",
        },
        "trace": {"requests": len(trace), "prompt_tokens": total_prompt},
        "results": results,
    }
    out = Path(args.out) if args.out else ROOT / "artifacts" / "vllm_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

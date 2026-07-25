"""Load generator and benchmark harness.

Spec section 11 methodology: Poisson arrivals at a sweep of lambda, fixed seed
per run, TTFT/TPOT/e2e percentiles, throughput, and goodput. Emits raw JSON
artifacts; bench/report.py turns those into docs/BENCHMARKS.md.

Per spec section 11, every number produced here is a measurement on the machine
that ran it, recorded alongside the command, commit, and hardware that produced
it. Nothing in this file compares HELIOS to vLLM -- see docs/SCOPE.md for why
that baseline is absent from this build.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from helios.core.types import SamplingParams, SloClass  # noqa: E402
from helios.engine import EngineConfig, LLMEngine  # noqa: E402


@dataclass
class WorkloadSpec:
    """A reproducible synthetic workload.

    Length distributions are lognormal-ish by default, which is closer to real
    chat traffic than a uniform draw: many short prompts, a long tail of large
    ones. ShareGPT traces would be better still but are not bundled -- see
    `--trace` to supply one.
    """

    num_requests: int = 64
    arrival_rate: float = 0.0          # requests/second; 0 == submit all at once
    prompt_len_mean: int = 128
    prompt_len_sigma: float = 0.6
    output_len_mean: int = 32
    output_len_sigma: float = 0.4
    max_prompt_len: int = 512
    seed: int = 0
    slo_mix: Tuple[float, float, float] = (0.2, 0.6, 0.2)   # A, B, C
    shared_prefix_len: int = 0         # >0 gives every request a common prefix
    vocab_size: int = 256

    def generate(self) -> List[Dict]:
        rng = random.Random(self.seed)
        prefix = [10 + i for i in range(self.shared_prefix_len)]
        requests = []
        t = 0.0

        for i in range(self.num_requests):
            plen = int(rng.lognormvariate(0.0, self.prompt_len_sigma) * self.prompt_len_mean)
            plen = max(1, min(plen, self.max_prompt_len))
            olen = int(rng.lognormvariate(0.0, self.output_len_sigma) * self.output_len_mean)
            olen = max(1, olen)

            body_len = max(1, plen - len(prefix))
            tokens = prefix + [
                rng.randrange(3, self.vocab_size) for _ in range(body_len)
            ]

            r = rng.random()
            if r < self.slo_mix[0]:
                cls = SloClass.A
            elif r < self.slo_mix[0] + self.slo_mix[1]:
                cls = SloClass.B
            else:
                cls = SloClass.C

            if self.arrival_rate > 0:
                # Poisson process: exponential inter-arrival times.
                t += rng.expovariate(self.arrival_rate)

            requests.append(
                {
                    "request_id": f"req-{i}",
                    "prompt_token_ids": tokens,
                    "max_tokens": olen,
                    "slo_class": cls,
                    "arrival_time": t,
                }
            )
        return requests


@dataclass
class RunResult:
    """One benchmark run's raw record."""

    name: str
    workload: Dict
    config: Dict
    num_requests: int = 0
    completed: int = 0
    duration_s: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft: Dict[str, float] = field(default_factory=dict)
    tpot: Dict[str, float] = field(default_factory=dict)
    e2e: Dict[str, float] = field(default_factory=dict)
    output_throughput: float = 0.0
    total_throughput: float = 0.0
    goodput_ratio: float = 0.0
    # Mean/max resident decoding sequences per step -- the operating point that
    # determines whether batched decode had anything to batch.
    mean_decode_batch: float = 0.0
    max_decode_batch: int = 0
    scheduler_stats: Dict = field(default_factory=dict)
    env: Dict = field(default_factory=dict)


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(len(ordered) * p / 100.0))
        return ordered[idx]

    return {
        "mean": statistics.fmean(ordered),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _env_info() -> Dict:
    """Record what produced the numbers (spec section 11)."""
    import torch

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    return {
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------- HELIOS run


def run_helios(
    model_dir: str,
    spec: WorkloadSpec,
    engine_overrides: Optional[Dict] = None,
    name: str = "helios",
) -> RunResult:
    """Drive the real engine through a workload, measuring wall-clock latency."""
    overrides = engine_overrides or {}
    cfg_kwargs = dict(
        model_dir=model_dir,
        kv_cache_bytes=64 * 1024 * 1024,
        block_size=16,
        max_num_seqs=64,
        max_num_batched_tokens=2048,
        max_model_len=1024,
    )
    cfg_kwargs.update(overrides)
    engine = LLMEngine(EngineConfig(**cfg_kwargs))
    result = _drive(engine, spec, name)
    result.config = {k: str(v) for k, v in cfg_kwargs.items()}
    return result


def _drive(engine: LLMEngine, spec: WorkloadSpec, name: str) -> RunResult:
    """Submit a workload honouring its arrival times and measure the result.

    Shared by every engine-backed configuration so that an ablation differs only
    in the engine it is handed -- if each variant had its own driver loop, a
    difference in the loop would be indistinguishable from a difference in the
    mechanism.
    """
    requests = spec.generate()
    # Clamp to what this engine can actually serve, so the harness measures
    # scheduling rather than rejection.
    servable = [
        r for r in requests if engine.scheduler.can_ever_serve(len(r["prompt_token_ids"]))
    ]

    start = time.perf_counter()
    pending = list(servable)
    submitted = 0
    decode_batch_sizes: List[int] = []

    while pending or engine.scheduler.has_work:
        now = time.perf_counter() - start
        while pending and pending[0]["arrival_time"] <= now:
            r = pending.pop(0)
            try:
                engine.add_request(
                    r["request_id"],
                    r["prompt_token_ids"],
                    SamplingParams(max_tokens=r["max_tokens"], temperature=0.0),
                    r["slo_class"],
                )
                submitted += 1
            except ValueError:
                pass

        if engine.scheduler.has_work:
            decode_batch_sizes.append(
                sum(1 for s in engine.scheduler.running if s.is_prefill_done)
            )
            engine.step()
        elif pending:
            # Idle until the next arrival rather than spinning.
            time.sleep(min(0.001, max(0.0, pending[0]["arrival_time"] - now)))

    duration = time.perf_counter() - start

    metrics = [m for m in engine.metrics() if m.finish_time is not None]
    result = RunResult(
        name=name,
        workload=_workload_dict(spec),
        config={},
        num_requests=submitted,
        completed=len(metrics),
        duration_s=duration,
        prompt_tokens=sum(m.prompt_tokens for m in metrics),
        output_tokens=sum(m.output_tokens for m in metrics),
        ttft=_percentiles([m.ttft for m in metrics if m.ttft is not None]),
        tpot=_percentiles([m.tpot for m in metrics if m.tpot is not None]),
        e2e=_percentiles([m.e2e for m in metrics if m.e2e is not None]),
        scheduler_stats={
            k: v for k, v in engine.stats_snapshot().items() if isinstance(v, (int, float))
        },
        env=_env_info(),
    )
    # Mean resident decode batch: the mechanism's actual operating point, and
    # the number that explains whether batching had anything to work with.
    nonzero = [n for n in decode_batch_sizes if n > 0]
    result.mean_decode_batch = (
        statistics.fmean(nonzero) if nonzero else 0.0
    )
    result.max_decode_batch = max(decode_batch_sizes) if decode_batch_sizes else 0
    if duration > 0:
        result.output_throughput = result.output_tokens / duration
        result.total_throughput = (result.prompt_tokens + result.output_tokens) / duration
    if metrics:
        result.goodput_ratio = sum(1 for m in metrics if m.meets_slo()) / len(metrics)
    return result


def _workload_dict(spec: WorkloadSpec) -> Dict:
    d = asdict(spec)
    d["slo_mix"] = list(spec.slo_mix)
    return d


# ----------------------------------------------------------------- baselines


def run_baseline_hf_loop(model_dir: str, spec: WorkloadSpec) -> RunResult:
    """Baseline 1 (spec section 11): naive per-request generation, no batching.

    Re-runs the full sequence every token with a fresh cache -- the honest
    "no engine at all" reference. This is what paging and batching are measured
    against.
    """
    from helios.exec.loader import load_model
    from helios.exec.paged_attn import PagedKVCache

    model = load_model(model_dir)
    cfg = model.config
    requests = spec.generate()

    import torch

    start = time.perf_counter()
    total_out = 0
    total_prompt = 0
    e2e: List[float] = []
    ttfts: List[float] = []
    tpots: List[float] = []

    with torch.inference_mode():
        for r in requests:
            r_start = time.perf_counter()
            seq = list(r["prompt_token_ids"])
            total_prompt += len(seq)
            first_token_at = None

            for n in range(r["max_tokens"]):
                n_blocks = len(seq) // 16 + 2
                caches = [
                    PagedKVCache(n_blocks, 16, cfg.num_key_value_heads, cfg.head_dim)
                    for _ in range(cfg.num_hidden_layers)
                ]
                logits = model.forward(
                    seq, list(range(len(seq))), caches, list(range(n_blocks)),
                    False, len(seq),
                )
                seq.append(int(logits.argmax()))
                total_out += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()

            done = time.perf_counter()
            e2e.append(done - r_start)
            if first_token_at is not None:
                ttfts.append(first_token_at - r_start)
                if r["max_tokens"] > 1:
                    tpots.append((done - first_token_at) / (r["max_tokens"] - 1))

    duration = time.perf_counter() - start
    result = RunResult(
        name="baseline_hf_loop",
        workload=_workload_dict(spec),
        config={"model_dir": model_dir, "note": "no paging, no batching, no cache reuse"},
        num_requests=len(requests),
        completed=len(requests),
        duration_s=duration,
        prompt_tokens=total_prompt,
        output_tokens=total_out,
        ttft=_percentiles(ttfts),
        tpot=_percentiles(tpots),
        e2e=_percentiles(e2e),
        env=_env_info(),
    )
    if duration > 0:
        result.output_throughput = total_out / duration
        result.total_throughput = (total_prompt + total_out) / duration
    return result


def run_baseline_sequential_executor(
    model_dir: str, spec: WorkloadSpec, name: str = "baseline_unbatched_executor"
) -> RunResult:
    """Baseline 3: HELIOS with the decode batching ablated away.

    Identical scheduler, identical paging, identical everything -- except each
    decoding sequence gets its own forward pass instead of sharing one batched
    GEMM. This isolates the executor's batching from every other mechanism,
    which is the only way to attribute a speedup to it (spec section 11:
    "each row attributable to one mechanism").
    """
    engine = LLMEngine(
        EngineConfig(
            model_dir=model_dir,
            kv_cache_bytes=64 * 1024 * 1024,
            block_size=16,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            max_model_len=1024,
        )
    )
    # Force the one-sequence-per-pass path that this build used before batched
    # decode existed.
    runner = engine.runner
    runner._run_decode_batch = lambda items: [
        runner._run_decode(it, 0) for it in items
    ]

    result = _drive(engine, spec, name)
    result.config["note"] = "decode batching ablated: one forward pass per sequence"
    return result


def run_baseline_unbatched_prefill(
    model_dir: str, spec: WorkloadSpec, name: str = "baseline_unbatched_prefill"
) -> RunResult:
    """Ablation: prefill chunks run one forward pass each.

    Separated from the decode ablation because the two mechanisms pay off on
    different workloads -- decode batching dominates when generations are long,
    prefill batching when prompts are long and generations short. Reporting one
    number for "batching" would hide that.
    """
    engine = LLMEngine(
        EngineConfig(
            model_dir=model_dir,
            kv_cache_bytes=64 * 1024 * 1024,
            block_size=16,
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            max_model_len=1024,
        )
    )
    runner = engine.runner
    runner._run_prefill_batch = lambda items: [
        o for o in (runner._run_prefill(it) for it in items) if o is not None
    ]
    result = _drive(engine, spec, name)
    result.config["note"] = "prefill batching ablated: one forward pass per chunk"
    return result


def run_baseline_static_batch(
    model_dir: str, spec: WorkloadSpec, batch_size: int = 8
) -> RunResult:
    """Baseline 2 (spec section 11): static batching.

    Requests are grouped into fixed batches; a batch runs until its LONGEST
    member finishes, and finished slots stay idle. That head-of-line waste is
    exactly what continuous batching removes, so this isolates the gain.

    Implemented on top of the engine with max_num_seqs=batch_size and admission
    frozen for the batch's duration, which reproduces static batching's
    semantics without a second model implementation.
    """
    requests = spec.generate()
    engine = LLMEngine(
        EngineConfig(
            model_dir=model_dir,
            kv_cache_bytes=64 * 1024 * 1024,
            block_size=16,
            max_num_seqs=batch_size,
            max_num_batched_tokens=2048,
            max_model_len=1024,
        )
    )
    servable = [
        r for r in requests if engine.scheduler.can_ever_serve(len(r["prompt_token_ids"]))
    ]

    start = time.perf_counter()
    completed = 0
    for i in range(0, len(servable), batch_size):
        chunk = servable[i : i + batch_size]
        for r in chunk:
            try:
                engine.add_request(
                    r["request_id"],
                    r["prompt_token_ids"],
                    SamplingParams(max_tokens=r["max_tokens"], temperature=0.0),
                    r["slo_class"],
                )
            except ValueError:
                continue
        # Drain the whole batch before admitting the next one.
        while engine.scheduler.has_work:
            engine.step()
        completed += len(chunk)

    duration = time.perf_counter() - start
    metrics = [m for m in engine.metrics() if m.finish_time is not None]
    result = RunResult(
        name=f"baseline_static_batch_{batch_size}",
        workload=_workload_dict(spec),
        config={"batch_size": str(batch_size), "note": "batch drains fully before next"},
        num_requests=len(servable),
        completed=len(metrics),
        duration_s=duration,
        prompt_tokens=sum(m.prompt_tokens for m in metrics),
        output_tokens=sum(m.output_tokens for m in metrics),
        ttft=_percentiles([m.ttft for m in metrics if m.ttft is not None]),
        tpot=_percentiles([m.tpot for m in metrics if m.tpot is not None]),
        e2e=_percentiles([m.e2e for m in metrics if m.e2e is not None]),
        env=_env_info(),
    )
    if duration > 0:
        result.output_throughput = result.output_tokens / duration
        result.total_throughput = (result.prompt_tokens + result.output_tokens) / duration
    if metrics:
        result.goodput_ratio = sum(1 for m in metrics if m.meets_slo()) / len(metrics)
    return result


# ----------------------------------------------------------------- ablations


ABLATIONS: Dict[str, Dict] = {
    # Spec section 11: "ablations are the interesting result, not the headline".
    # Each row differs from `full` by exactly one mechanism.
    "full": {},
    "no_prefix_cache": {"enable_prefix_cache": False},
    "no_chunked_prefill": {"enable_chunked_prefill": False},
    "with_spec_decode": {"enable_spec_decode": True, "spec_gamma": 4},
    "small_kv_pool": {"kv_cache_bytes": 8 * 1024 * 1024},
    "batch1": {"max_num_seqs": 1},
}


def main() -> None:
    ap = argparse.ArgumentParser(description="HELIOS benchmark harness")
    ap.add_argument("--model", required=True, help="model directory")
    ap.add_argument("--out", default="artifacts", help="where to write JSON records")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=16)
    ap.add_argument("--arrival-rate", type=float, default=0.0)
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt-len", type=int, default=256)
    ap.add_argument(
        "--suite",
        default="ablations",
        choices=["ablations", "baselines", "all", "quick", "prefill-heavy",
                 "prefix-cache", "kv-quant"],
    )
    ap.add_argument("--static-batch-size", type=int, default=8)
    args = ap.parse_args()

    spec = WorkloadSpec(
        num_requests=args.requests,
        arrival_rate=args.arrival_rate,
        prompt_len_mean=args.prompt_len,
        output_len_mean=args.output_len,
        max_prompt_len=args.max_prompt_len,
        shared_prefix_len=args.shared_prefix,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[RunResult] = []

    # Warm-up run, discarded. Lazy imports (transformers, tokenizers, torch
    # subsystems) cost tens of seconds on first touch, and whichever
    # configuration ran first would otherwise absorb all of it -- once making an
    # ablation look 17x slower than an identical one. Never publish a first run.
    print("[bench] warm-up (discarded) ...", flush=True)
    warmup = WorkloadSpec(
        num_requests=2, prompt_len_mean=16, output_len_mean=4, max_prompt_len=32, seed=0
    )
    run_helios(args.model, warmup, {}, name="warmup")

    if args.suite in ("ablations", "all", "quick"):
        names = ["full", "no_prefix_cache", "batch1"] if args.suite == "quick" else list(ABLATIONS)
        for name in names:
            print(f"[bench] running ablation {name} ...", flush=True)
            results.append(
                run_helios(args.model, spec, ABLATIONS[name], name=f"helios_{name}")
            )

    if args.suite == "prefill-heavy":
        # Long prompts, short generations: the regime where prefill batching --
        # not decode batching -- is the mechanism under test. One workload cannot
        # exercise both, so the suite is explicit about which it measures.
        heavy = WorkloadSpec(
            num_requests=args.requests,
            prompt_len_mean=max(96, args.prompt_len),
            prompt_len_sigma=0.3,
            output_len_mean=4,
            output_len_sigma=0.2,
            max_prompt_len=max(192, args.max_prompt_len),
            seed=args.seed,
        )
        print("[bench] prefill-heavy: full ...", flush=True)
        results.append(run_helios(args.model, heavy, {}, name="helios_prefill_heavy"))
        print("[bench] prefill-heavy: unbatched prefill ...", flush=True)
        results.append(
            run_baseline_unbatched_prefill(
                args.model, heavy, name="baseline_prefill_heavy_unbatched"
            )
        )

    if args.suite == "prefix-cache":
        # A long shared prefix is the regime the cache exists for. With a short
        # one its bookkeeping can cost more than it saves, which is a real result
        # but a misleading headline -- so the mechanism gets its own workload.
        shared = WorkloadSpec(
            num_requests=args.requests,
            prompt_len_mean=max(160, args.prompt_len),
            prompt_len_sigma=0.15,
            output_len_mean=8,
            output_len_sigma=0.2,
            max_prompt_len=max(256, args.max_prompt_len),
            shared_prefix_len=max(128, args.shared_prefix),
            seed=args.seed,
        )
        print("[bench] prefix-cache: on ...", flush=True)
        results.append(run_helios(args.model, shared, {}, name="helios_shared_prefix_cache_on"))
        print("[bench] prefix-cache: off ...", flush=True)
        results.append(
            run_helios(
                args.model, shared, {"enable_prefix_cache": False},
                name="helios_shared_prefix_cache_off",
            )
        )

    if args.suite in ("kv-quant", "all"):
        # The INT8 KV cache's whole claim is CONCURRENCY, not speed: the same byte
        # budget holds ~2x the blocks, so more sequences stay resident and the
        # batched-decode win (which scales with batch size) gets bigger.
        #
        # The pool must actually BIND, and getting that right took two attempts.
        # At 4 MiB the toy model's cache holds 512 blocks -- far more than this
        # workload needs -- so both configurations kept every sequence resident
        # (mean decode batch 14.95 in both) and the ablation measured nothing but
        # dequantization overhead: int8 came out 20% SLOWER. That is a real cost,
        # but reporting it as "the INT8 KV cache is slower" would be measuring a
        # mechanism outside the regime it exists for, which this project has
        # already got wrong once (see BENCHMARKS.md on the prefix cache).
        #
        # 256 KiB is sized so the difference is structural: fp32 gets 32 blocks
        # (512 tokens) and must preempt, int8 gets 113 (1808 tokens) and does not.
        cramped = {"kv_cache_bytes": 256 * 1024, "max_num_seqs": 64}
        print("[bench] running KV-quantization ablation (cramped pool) ...", flush=True)
        results.append(
            run_helios(args.model, spec, cramped, name="helios_kv_fp_cramped")
        )
        results.append(
            run_helios(
                args.model, spec, {**cramped, "quantize_kv": True},
                name="helios_kv_int8_cramped",
            )
        )

    if args.suite in ("baselines", "all"):
        print("[bench] running unbatched-decode ablation ...", flush=True)
        results.append(run_baseline_sequential_executor(args.model, spec))
        print("[bench] running unbatched-prefill ablation ...", flush=True)
        results.append(run_baseline_unbatched_prefill(args.model, spec))
        print("[bench] running static batching baseline ...", flush=True)
        results.append(run_baseline_static_batch(args.model, spec, args.static_batch_size))
        print("[bench] running naive loop baseline ...", flush=True)
        # The naive loop is O(n^2) in sequence length; keep it small so a run
        # finishes in reasonable time on CPU.
        small = WorkloadSpec(
            num_requests=min(8, args.requests),
            prompt_len_mean=min(32, args.prompt_len),
            output_len_mean=min(8, args.output_len),
            max_prompt_len=64,
            seed=args.seed,
        )
        results.append(run_baseline_hf_loop(args.model, small))

    for r in results:
        path = out_dir / f"{r.name}_seed{spec.seed}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(r), f, indent=2)
        print(
            f"[bench] {r.name}: {r.completed}/{r.num_requests} reqs, "
            f"{r.output_throughput:.1f} out tok/s, "
            f"ttft_p50={r.ttft.get('p50', 0):.4f}s "
            f"tpot_p50={r.tpot.get('p50', 0):.4f}s -> {path}"
        )


if __name__ == "__main__":
    main()

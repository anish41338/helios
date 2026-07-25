# HELIOS

An LLM serving engine built from scratch in Python: paged attention, a
copy-on-write paged KV allocator, iteration-level continuous batching, chunked
prefill, a radix-trie prefix cache, an OpenAI-compatible HTTP API — and a
deterministic simulation testing harness that found **15 real bugs** in the
scheduler.

Implements the "minimal defensible v1" scope of [HELIOS-SPEC.md](HELIOS-SPEC.md)
(spec §12.2). **Read [docs/SCOPE.md](docs/SCOPE.md) before making any claim about
this project** — it states exactly what is built and what is not. Short version:
no CUDA/Triton kernels, no quantization, no prefill/decode disaggregation, no
vLLM comparison. The machine this was built on has no NVIDIA GPU, and spec §19.5
says cut scope rather than ship something unverified.

## Quick start

```bash
pip install torch safetensors transformers fastapi uvicorn pytest httpx

# a small random-weight model, so everything below runs with no downloads
PYTHONPATH=python python -m helios.cli make-toy-model --out /tmp/toy

# derived KV capacity for a model + memory budget
PYTHONPATH=python python -m helios.cli info --model /tmp/toy --kv-mb 64

# offline generation
PYTHONPATH=python python -m helios.cli generate \
    --model /tmp/toy --prompt "hello" --max-tokens 16

# OpenAI-compatible server
PYTHONPATH=python python -m helios.cli serve --model /tmp/toy --port 8000
curl localhost:8000/v1/completions \
    -H 'content-type: application/json' \
    -d '{"prompt":"hello","max_tokens":16}'

# the simulation harness
PYTHONPATH=python python -m helios.cli vopr --seeds 1000
```

With a real model, point `--model` at any Llama-3.x or Qwen2.5 HuggingFace
directory (fp32/fp16 safetensors; no AWQ/GPTQ support — see SCOPE).

## Why the DST harness is the interesting part

The scheduler and allocator are written as a pure state machine — no clock, no
I/O, no threads, no unseeded RNG, no iteration over unordered containers. That
lets the whole system be driven from a seed under adversarial fault injection,
single-threaded, asserting every invariant after every step.

**50,000 seeds, 0 failures.** Any failure replays exactly:

```bash
PYTHONPATH=python python -m helios.cli vopr --seed 918273 --replay
```

It found 15 bugs, and the shape of them is the point: **13 of 15 were liveness
bugs** — livelocks and starvation — not corruption. The engine kept running,
kept looking busy, and quietly completed nothing. Ordinary integration testing
would have found almost none of them. Highlights:

- a copy-on-write accounting error that overshot the memory watermark, because a
  shared *partially-filled* tail block reported spare room while still needing a
  copy (the exact bug class spec §5.3 predicted);
- a self-sustaining livelock where a sequence that failed admission kept its
  prefix-cache pin, making the very memory it needed unevictable;
- a sequence admitted, OOM-preempted, and re-prefilled **every step forever**
  because admission reserved no room for the tokens it would generate;
- a leak of the entire KV pool via a sequence finished outside the `running`
  list, which no reaper ever visited.

Full writeup with root causes and reproductions: [docs/DST.md](docs/DST.md).

## Correctness

163 tests. The parity tests are the backbone, per spec §13.1 and §19.2 (a
tolerance is never loosened to make a test pass):

| Property | Guarantee |
|---|---|
| Paged attention vs dense reference | matches to float precision, with non-contiguous physical blocks |
| Chunked prefill vs single-shot | **bit-identical** |
| Speculative vs non-speculative decode | **bit-identical**, γ ∈ {1,2,4,8} (spec §7.2) |
| Prefix cache on vs off | identical output |
| Engine vs naive generation loop | identical tokens |
| Allocator I1–I7 | asserted after every op, 40 randomized walks |

```bash
PYTHONPATH=python python -m pytest tests -q
```

## Benchmarks

[docs/BENCHMARKS.md](docs/BENCHMARKS.md) is **generated** from JSON artifacts by
`bench/report.py` and never hand-edited (spec §11). It reports CPU measurements
on a toy model, so it characterises the *relative* cost of engine mechanisms and
nothing else — absolute tokens/second are meaningless here, and there is no
comparison against another engine.

```bash
python bench/loadgen.py --model artifacts/bench_model --suite all --requests 24
python bench/report.py
```

The harness does Poisson arrivals, TTFT/TPOT/e2e percentiles, goodput against
per-class SLO targets, and per-mechanism ablations. Two of the spec's four
baselines are implemented (naive loop, static batching); vLLM needs CUDA.

> A note on method: the first version of this harness reported one ablation as
> 17× slower than an identical one. The cause was lazy-import cost landing
> entirely on whichever configuration ran first. There is now a discarded warm-up
> run. Worth stating because it is exactly the kind of artifact that turns into a
> false claim.

## Layout

```
python/helios/
  core/          the deterministic core -- no clock, no I/O, no RNG
    allocator.py     paged blocks, ref counts, CoW, I1-I7        (spec §5)
    scheduler.py     continuous batching, admission, preemption   (spec §6)
    prefix_cache.py  radix trie, block-aligned, LRU eviction      (spec §3)
    vopr.py          the DST harness                              (spec §10)
    types.py         requests, sequences, SLO classes             (spec §6.2)
    execstep.py      the scheduler<->executor contract            (spec §10.1)
  exec/          PyTorch execution
    model.py         Llama/Qwen2: GQA, RoPE, SwiGLU, RMSNorm      (spec §8.1)
    paged_attn.py    attention over a block table                 (spec §8.3)
    loader.py        safetensors loading                          (spec §8.2)
    runner.py        ExecStep -> forward pass; speculation        (spec §7.2)
    sampler.py       greedy, temperature, top-k, top-p
  api/server.py  OpenAI-compatible HTTP + Prometheus              (spec §9)
  engine.py      scheduler + executor + metrics
  cli.py         serve / generate / vopr / info / make-toy-model
bench/           load generator, baselines, report generator      (spec §11)
tests/           allocator, scheduler, parity, e2e
docs/            SCOPE.md, DST.md, ARCHITECTURE.md, BENCHMARKS.md
```

## Design notes

**Why paging.** A contiguous per-sequence KV buffer sized to `max_seq_len`
wastes memory to internal fragmentation — a 100-token request in a 4096 slot
wastes 97.5%. Paging allocates fixed-size blocks and maps them per sequence,
exactly like OS virtual memory. The cost is gather overhead and kernel
complexity.

**Why the scheduler is pure.** Determinism is load-bearing, not stylistic (spec
§19.4). A single `time.time()` in the scheduler would make a failing seed
unreplayable and the harness worthless. A test asserts the module contains no
clock or RNG import.

**Why decode and prefill contend.** Prefill is compute-bound (many query tokens
amortise each weight read); decode is memory-bandwidth-bound (one query token
attends over the whole context with almost no reuse). Interleaved on one device
they fight, which is what chunked prefill mitigates and what the spec's
disaggregation would have solved properly.

**Why speculation must be bit-identical.** Speculation that changes output is a
bug, not a speedup (spec §7.2). Note the honest caveat in SCOPE.md: draft and
verify share the same fp32 weights here, so acceptance is 1.0 by construction
and there is no speedup — the test proves the *bookkeeping*, not the idea.

## License

Not specified.

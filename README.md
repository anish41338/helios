# HELIOS

An LLM serving engine built from scratch in Python: paged attention, a
copy-on-write paged KV allocator, iteration-level continuous batching, chunked
prefill, a radix-trie prefix cache, INT4 weight quantization with AWQ, an INT8
paged KV cache, quantization-asymmetric speculative decoding, a fused Triton
kernel, an OpenAI-compatible streaming HTTP API — and a deterministic simulation
testing harness that found **15 real bugs** in the scheduler.

**Read [docs/SCOPE.md](docs/SCOPE.md) before making any claim about this
project** — it states exactly what is built and what is not. Short version of
what is *not*: no fused int4 GEMM (so every quantization speedup here is
**modelled**, not measured), no vLLM comparison (written, never run), no real
multi-device disaggregation (the transfer FSM is verified, the transport is
simulated), no Rust.

The headline result, because it is the one that decides a design question rather
than reporting a speedup:

> **α = 0.6548** — the measured rate at which a 4-bit draft of the model's own
> weights agrees with the full-precision verifier on Qwen2.5-0.5B. The spec's
> pre-committed kill threshold was 0.6, so the feature survives; its *target* was
> 0.78, so it misses. Optimal draft length is **2**, not the planned 4, and at
> γ=8 speculation becomes a net **loss**. See [docs/QASSD.md](docs/QASSD.md).

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

# INT8 KV cache: ~2x smaller, so ~2x the resident sequences
PYTHONPATH=python python -m helios.cli generate \
    --model /tmp/toy --prompt "hello" --quantize-kv

# QASSD: draft from a 4-bit view of the same weights
PYTHONPATH=python python -m helios.cli generate \
    --model /tmp/toy --prompt "hello" --spec-decode --quantized-draft

# the simulation harness
PYTHONPATH=python python -m helios.cli vopr --seeds 1000

# measure the acceptance rate the QASSD design depends on
python bench/measure_alpha.py --model artifacts/qwen05b --gamma 4
```

With a real model, point `--model` at any Llama-3.x or Qwen2.5 HuggingFace
directory (fp32/fp16 safetensors; quantization is applied in-process rather than
loaded from a pre-quantized AWQ/GPTQ checkpoint).

## Why the DST harness is the interesting part

The scheduler and allocator are written as a pure state machine — no clock, no
I/O, no threads, no unseeded RNG, no iteration over unordered containers. That
lets the whole system be driven from a seed under adversarial fault injection,
single-threaded, asserting every invariant after every step.

**50,000 seeds, 0 failures** (3,807 s single-threaded), plus **20,000 re-verified
on the current commit** after the executor rewrite — artifacts in
`artifacts/dst_50k.json` and `artifacts/dst_20k_final.json`. Any failure replays
exactly:

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

## It serves real models

Verified on a Tesla T4 in fp16 with Qwen2.5-0.5B:

```
"The capital of France is"  ->  " Paris. It is the largest city in Europe and the
                                 second largest in the world..."
"def add(a, b):"            ->  "
    return a + b

def subtract(a, b):
                                 
    return a - b

def multiply(a,"
```

Paging, block tables, RoPE, GQA, RMSNorm, SwiGLU, batched prefill, batched decode,
fp16 numerics and sampling all have to be right at once for that to come out. The
toy model used by the test suite has random weights, so it proves
self-consistency but could never catch an error shared by the implementation and
its own reference. This does.

```bash
python bench/real_model_check.py --model artifacts/qwen05b --device cuda
```

## Correctness

178 tests (+9 GPU-only). The parity tests are the backbone, per spec §13.1 and §19.2 (a
tolerance is never loosened to make a test pass):

| Property | Guarantee |
|---|---|
| Paged attention vs dense reference | matches to float precision, with non-contiguous physical blocks |
| Chunked prefill vs single-shot | **bit-identical** |
| Speculative vs non-speculative decode | **bit-identical**, γ ∈ {1,2,4,8} (spec §7.2) |
| Prefix cache on vs off | identical output |
| Engine vs naive generation loop | identical tokens |
| Batched vs sequential decode | identical logits, and order-invariant |
| Output under forced preemption | identical after 14 recompute cycles |
| Triton kernel vs PyTorch paged path | matches within fp16 tolerance on a T4, order-invariant |
| Loader vs unmapped checkpoint tensors | refuses to load rather than drop weights |
| Allocator I1–I7 | asserted after every op, 40 randomized walks |
| **QASSD vs unspeculated decode** | **identical token ids** — the int4 draft cannot change output |
| INT4 pack/unpack | exactly invertible; error within half a quantization step |
| AWQ vs RTN | AWQ's output error never worse, and lower with salient channels |
| INT8 KV round-trip | within `s/2 + \|q\|·\|s − fp16(s)\|`, both terms derived |
| KV transfer FSM | every illegal transition raises; both partitions' blocks reclaimed on failure |
| Streamed vs non-streamed text | concatenated deltas reproduce the whole answer exactly |

```bash
PYTHONPATH=python python -m pytest tests -q
```

The QASSD row is the one to read twice. Speculation is correct **not because the
draft is good** but because the verifier decides which tokens commit — so a
degraded 4-bit draft costs throughput and nothing else. That property is what
makes it safe to ship a draft whose quality you have not characterised, and it is
asserted by test rather than argued in a comment.

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

**Measured on a Tesla T4** (sm_75, CUDA 12.8, commit `81937c9`), each number a
single-mechanism ablation with everything else identical:

| mechanism | ablation | GPU (T4) | CPU |
|---|---|---|---|
| **Triton fused kernel** | PyTorch paged path | **5.1–5.6×** on paged decode attn | n/a |
| **Batched decode** | one forward pass per sequence | **3.14×** throughput | 2.24× |
| **Continuous batching** | static batching (8) | **1.50×** throughput | 1.13× |
| **Prefix cache** | cache off | **1.38×**, 1.28× TTFT | 1.49× |
| **Batched prefill** | one pass per chunk | **1.20×** TTFT | 1.16× |
| whole engine | naive loop, no KV reuse | **6.2×** | 5.7× |

`helios_full`: **1,878 out tok/s** on the T4 (409 on CPU). `batch1` gets 4.9×
better TPOT (0.0021 s vs 0.0102 s) at 3.4× worse TTFT — the batching trade-off,
measured.

Every scheduling mechanism pays **more** on the GPU than on CPU, which is the
prediction the theory makes: batching is worth what the batch dimension is cheap,
and a GPU makes it cheapest. Mean resident decode batch was 11.6 — reported
because the win scales with it.

Two further method notes: a discarded warm-up run exists because lazy-import cost
once landed entirely on whichever ablation ran first and made it look 17× slower;
and speculation is *slower* here by design (draft and verify share fp32 weights —
see [docs/SCOPE.md](docs/SCOPE.md)).

## Layout

```
python/helios/
  core/          the deterministic core -- no clock, no I/O, no RNG
    allocator.py     paged blocks, ref counts, CoW, I1-I7        (spec §5)
    scheduler.py     continuous batching, admission, preemption   (spec §6)
    prefix_cache.py  radix trie, block-aligned, LRU eviction      (spec §3)
    disagg.py        KV transfer FSM, 7 states, 5 fault kinds     (spec §6.4)
    vopr.py          the DST harness                              (spec §10)
    types.py         requests, sequences, SLO classes             (spec §6.2)
    execstep.py      the scheduler<->executor contract            (spec §10.1)
  exec/          PyTorch execution
    model.py         Llama/Qwen2: GQA, RoPE, SwiGLU, RMSNorm      (spec §8.1)
    paged_attn.py    attention over a block table; INT8 KV cache  (spec §8.3, §7.4)
    triton_attn.py   fused paged decode kernel, 5.26x on a T4     (spec §8.3)
    quant.py         INT4 packing, group scales, AWQ search       (spec §8.2)
    qassd.py         int4 draft / fp verify, the alpha model      (spec §7)
    loader.py        safetensors loading                          (spec §8.2)
    runner.py        ExecStep -> forward pass; speculation        (spec §7.2)
    sampler.py       greedy, temperature, top-k, top-p
  api/server.py  OpenAI-compatible HTTP, incremental SSE, metrics (spec §9)
  engine.py      scheduler + executor + metrics
  cli.py         serve / generate / vopr / info / make-toy-model
bench/           load generator, baselines, alpha measurement,    (spec §11)
                 significance analysis, PDF report generator
tests/           allocator, scheduler, quant, disagg, parity, e2e
docs/            SCOPE.md, DST.md, QASSD.md, GPU.md, ARCHITECTURE.md,
                 BENCHMARKS.md (generated), HELIOS-REPORT.pdf (generated)
```

A typeset engineering report is generated from the same artifacts the docs use:

```bash
python bench/make_report.py --out docs/HELIOS-REPORT.pdf
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
they fight. Chunked prefill *bounds* that contention; disaggregation removes it,
which is why [core/disagg.py](python/helios/core/disagg.py) exists — and why the
decision of whether to transfer the KV or just recompute it is computed from
bandwidth rather than assumed. For a 262 MB KV cache, transferring wins by ~11×
over PCIe 3.0 and *loses* over 10 GbE. The crossover is at the machine boundary.

**Why speculation must be bit-identical.** Speculation that changes output is a
bug, not a speedup (spec §7.2). This holds for a reason worth stating: the
*verifier* decides which tokens commit, so the emitted sequence is the target
model's, whatever the draft proposes. That is what makes it safe to draft from a
4-bit approximation whose quality you have measured but not controlled.

**Why the kill threshold was written down first.** The 0.60 acceptance floor was
fixed before any of the quantization work existed. The measurement came in at
0.655 — close enough that a threshold picked afterwards would have been
indefensible in either direction. Because it was pre-committed, the result is a
decision rather than a rationalisation: the feature ships, its optimal γ is 2 and
not the planned 4, and its expected benefit is bounded by a number instead of by
hope.

## License

Not specified.

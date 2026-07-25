# What is built, and what is not

This document exists because HELIOS-SPEC.md describes an ambitious system and
this repository implements a subset of it. Spec section 19 forbids letting a
claim outrun the code, so the boundary is written down explicitly rather than
left for a reader to discover.

**Read this before making any claim about HELIOS, on a resume or anywhere else.**

## The constraint that determined scope

The spec's performance targets (section 11) assume an RTX 4090. Section 2 tier
T3 anticipated having only a Kaggle T4. The machine this was actually built on
has **no NVIDIA GPU at all**:

```
torch 2.5.1+cpu     cuda_available = False
```

Spec section 12.2 anticipated exactly this and defined a "minimal defensible
v1". Section 19.5 says to prefer cutting scope over shipping something
unverified. This build follows both: it implements section 12.2 in full, on CPU,
with everything tested — rather than writing thousands of lines of CUDA that
could never be compiled, run, or benchmarked here.

## Built and verified

| Subsystem | Spec section | Status |
|---|---|---|
| Paged KV cache, block tables, ref counts | 5 | Built. I1–I7 asserted, property-tested |
| Copy-on-write prefix sharing | 5.3 | Built, incl. the shared-partial-tail case |
| Host swap tier (second memory tier) | 6.3 | Built, exercised by DST |
| Iteration-level continuous batching | 6.1 | Built |
| SLO classes + token-bucket admission | 6.2 | Built, step-driven (deterministic) |
| Recompute/swap preemption | 6.3 | Built, with the ratio test |
| Chunked prefill | 6.1 | Built, bit-identical to single-shot |
| RadixTrie prefix cache | 3 | Built, block-aligned, LRU eviction |
| Paged attention | 8.3 | Built as gather+SDPA (the spec's fallback path) |
| Llama/Qwen2 architecture (GQA, RoPE, SwiGLU, RMSNorm) | 8.1 | Built from scratch |
| safetensors loading | 8.2 | Built (fp32/fp16) |
| Sampling (greedy, temp, top-k, top-p, per-request seeds) | 9.1 | Built |
| Self-speculative decode structure + accept/rollback | 7.2 | Built (see caveat below) |
| Adaptive speculation gating | 7.3 | Built (batch size + measured acceptance) |
| Deterministic simulation testing | 10 | Built. 50k seeds, fault injection, replay |
| OpenAI-compatible HTTP API | 9.1 | Built (`/v1/completions`, `/v1/chat/completions`) |
| Prometheus metrics | 9.2 | Built |
| Benchmark harness + ablations | 11 | Built (Poisson arrivals, percentiles, goodput) |
| Baselines: naive loop, static batching | 11 | Built (2 of the 4 the spec lists) |

## NOT built — do not claim any of this

| Missing | Spec section | Why |
|---|---|---|
| **Rust frontend + scheduler** | 3, 4 | Cut by section 12.2. Everything is Python |
| **Triton/CUDA paged-attention kernel** | 8.3 | No GPU; Triton has no CPU target |
| **W4A16 / AWQ / GPTQ quantization** | 8.2 | Needs GPU kernels to be meaningful |
| **W4A4 draft path — the "QA" in QASSD** | 7.1 | Same. This is the core novelty claim and **it is not implemented** |
| **Quantized KV shadow / dual-buffer** | 7.4 | Same |
| **Prefill/decode disaggregation** | 6.4 | Needs ≥2 devices; there are none |
| **KV transfer FSM across partitions** | 6.4 | Same. (Fault *kinds* are modelled in DST, the transport is not) |
| **70B pipeline-parallel across 2 cards** | 8.1 | Same |
| **vLLM baseline comparison** | 11 | vLLM requires CUDA; cannot be run here |
| **gRPC surface, shm ring buffer IPC** | 3 | Cut by section 12.2; single process |
| **Token-by-token SSE streaming** | 9.1 | Wire format is correct, but one frame per response |

### The speculative decoding caveat, stated precisely

`helios/exec/runner.py` implements the full speculative *control flow*: draft γ
tokens, verify in one parallel pass, accept the longest matching prefix, emit a
bonus token, roll back rejected positions in the KV cache. Tests prove the
output is **bit-identical** to non-speculative decoding for γ ∈ {1,2,4,8}.

But draft and verify read the **same fp32 weights**, because there is no
quantized path. So:

- acceptance rate is 1.0 **by construction**, not by merit;
- there is **no speedup** — it is strictly slower, since drafting is serial;
- it says **nothing** about whether QASSD's α ≥ 0.78 target is achievable.

What it does establish: the scheduler's accept/rollback bookkeeping and KV
truncation are correct, which is the part most likely to silently corrupt
output. The quantization asymmetry that would make it fast is future work.

## What the benchmarks do and do not show

`docs/BENCHMARKS.md` is generated from JSON artifacts by `bench/report.py` and
never hand-edited (spec section 11). It reports CPU measurements on a toy
random-weight model. It characterises **relative** cost of engine mechanisms.

It is not evidence about GPU throughput, quantized inference, or any comparison
with another engine. Absolute tokens/second numbers are meaningless here — the
model is a toy.

## Defensible claims

Adapting the wording spec section 12.2 pre-approved:

> Implemented a from-scratch LLM serving engine with paged attention, a
> copy-on-write paged KV allocator, iteration-level continuous batching,
> chunked prefill, a radix-trie prefix cache, and an OpenAI-compatible API.
> Validated the scheduler and allocator with a deterministic simulation testing
> harness (TigerBeetle/FoundationDB style) under adversarial fault injection:
> 50,000 seeds, seven invariants checked every step, every failure replayable
> from its seed. **The harness found 15 real bugs**, including a
> copy-on-write accounting error that overshot the memory watermark and six
> distinct admission/eviction livelocks — all documented with reproductions in
> `docs/DST.md`. Verified correctness by bit-exact parity: chunked vs single-shot
> prefill, speculative vs non-speculative decoding, and paged vs dense attention.

Not defensible, and not claimed anywhere in this repo: beating vLLM,
quantization-asymmetric speculation, heterogeneous multi-GPU serving,
disaggregated prefill/decode.

## If a GPU becomes available

Roughly the spec's own ordering, cheapest first:

1. Triton paged-attention kernel, validated against the existing gather+SDPA
   path (which is already the correctness oracle a kernel needs).
2. W4A16 AWQ loading + GEMM; re-run the ablation suite.
3. W4A4 draft path + quantized KV shadow. **Measure α before building the rest**
   — spec section 14 says kill the feature if α < 0.6.
4. Two devices → partition roles and the KV transfer FSM. The FSM's fault
   taxonomy already exists in the DST harness.
5. vLLM baseline on identical hardware/model/quantization.

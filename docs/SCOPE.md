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
| Iteration-level continuous batching | 6.1 | Built; batched decode executor, 2.3x ablation win |
| SLO classes + token-bucket admission | 6.2 | Built, step-driven (deterministic) |
| Recompute/swap preemption | 6.3 | Built, with the ratio test |
| Chunked prefill | 6.1 | Built, bit-identical to single-shot |
| RadixTrie prefix cache | 3 | Built, block-aligned, LRU eviction |
| Paged attention | 8.3 | Built as gather+SDPA (the spec's fallback path) |
| Batched paged decode attention | 8.3 | Built (vectorised over sequences, padded + masked) |
| **Triton fused paged-attention kernel** | 8.3 | **Built and verified on a T4: 5.26x over the PyTorch path**, 9/9 parity tests |
| Llama/Qwen2 architecture (GQA, RoPE, SwiGLU, RMSNorm) | 8.1 | Built from scratch |
| safetensors loading | 8.2 | Built (fp32/fp16) |
| Sampling (greedy, temp, top-k, top-p, per-request seeds) | 9.1 | Built |
| Self-speculative decode structure + accept/rollback | 7.2 | Built (see caveat below) |
| Adaptive speculation gating | 7.3 | Built (batch size + measured acceptance) |
| Deterministic simulation testing | 10 | Built. Seed sweeps, fault injection, replay |
| **Serving a real pretrained model** | 8.1 | **Verified on T4: Qwen2.5-0.5B generates coherent text in fp16** |
| **W4A16 INT4 weight quantization** | 8.2 | Built. Packed int4, group-wise scales, **3.8× vs fp16** measured |
| **AWQ activation-aware scaling** | 8.2 | Built. **1.58× lower output error than RTN**, measured over 168 layers |
| **INT8 quantized KV cache (the "KV shadow")** | 7.4 | Built. Per-token per-head scales, **1.94× vs fp16**, 3.4× more blocks |
| **Quantization-asymmetric speculation (QASSD)** | 7.1 | Built. **α = 0.6548 measured** on Qwen2.5-0.5B — see below |
| **KV transfer FSM (prefill→decode migration)** | 6.4 | **Verified design, NOT in the serving path** — see the note below |
| **Token-by-token SSE streaming** | 9.1 | Built. Incremental detokenization, reassembly asserted |
| OpenAI-compatible HTTP API | 9.1 | Built (`/v1/completions`, `/v1/chat/completions`) |
| Prometheus metrics | 9.2 | Built |
| Benchmark harness + ablations | 11 | Built (Poisson arrivals, percentiles, goodput) |
| Baselines: naive loop, static batching | 11 | Built (2 of the 4 the spec lists) |

## NOT built — do not claim any of this

| Missing | Spec section | Why |
|---|---|---|
| **Rust frontend + scheduler** | 3, 4 | Cut by section 12.2. Everything is Python |
| **A fused int4 GEMM** | 8.2 | The quantization is numerically real; the *speedup* is not. `QuantLinear` dequantizes then calls a normal GEMM, so on CPU it is **slower** than fp32. Every speedup from quantization in this repo is labelled **modelled** |
| **W4A4 — 4-bit activations** | 7.1 | The T4 has no fast int4 activation path, so a W4A4 draft would not be cheaper than the verify and the premise fails. Hardware constraint, not a gap |
| **Disaggregation across real devices** | 6.4 | The FSM, the block accounting, and the transfer/re-prefill decision are built and DST-verified. The *transport* is simulated — no second device to move bytes to |
| **70B pipeline-parallel across 2 cards** | 8.1 | Needs the hardware |
| **vLLM baseline comparison** | 11 | `bench/vllm_baseline.py` is **written with matched controls but has never been run** — vLLM requires CUDA. No comparison is claimed |
| **gRPC surface, shm ring buffer IPC** | 3 | Cut by section 12.2; single process |

### QASSD, stated precisely

The quantization asymmetry **is** implemented now, and measured. `docs/QASSD.md`
has the full result; the honest summary:

- **α = 0.6548** (336 drafted, 220 accepted) on Qwen2.5-0.5B with an int4 AWQ
  draft against fp32 verify.
- Spec section 14's **kill gate (α ≥ 0.6) passes.** Spec section 7.3's **target
  (α ≥ 0.78) does not.**
- Best γ is **2**, not the spec's 4. At γ=8 speculation is a net **loss** (0.94×).
- Modelled speedup at best γ: **~1.4×**, or ~1.56× using measured tokens/pass.
  **Modelled, not measured** — there is no int4 GEMM here (spec section 19.6).
- It costs **more** memory, not less: +41% on weights (both precisions resident)
  **and a whole second KV cache** for the draft. `engine.memory_report()` returns
  both, because quoting the weight figure alone understates it.

The safety property is what makes this shippable and it is asserted, not argued:
`test_asymmetric_speculation_matches_the_target_exactly` requires **identical
token ids** with and without the quantized draft. Speculation is correct because
the *verifier* decides, not because the draft is good — so a bad draft costs
throughput and nothing else.

The symmetric mode (draft and verify at the same precision, α = 1.0 by
construction) is retained as the correctness oracle: with no quantization noise,
output must be bit-identical for γ ∈ {1,2,4,8}, so there is nothing for a
bookkeeping bug to hide behind.

For a 0.5B model this is a poor trade. Quantization error at fixed bit-width
falls with model size, so a 7B model should do better — **that is a hypothesis,
not a result, and it is not claimed.**

## End-to-end verification on real weights

The strongest single correctness result in this repo, and it has now been
reproduced on **two devices independently**, which is what rules out a
device-specific fluke or a misreported run.

On CPU (fp32), the same prompts produce the same continuations token-for-token as
the T4 did, and the full OpenAI-compatible HTTP path was driven against the real
checkpoint:

```
POST /v1/completions   "The capital of France is"
  -> " Paris. It is the largest city in Europe and the second"

POST /v1/completions   stream=true, same prompt
  -> 13 SSE frames, concatenating to byte-identical text
```

So: real pretrained weights, through the paged KV path, through the scheduler,
through the HTTP frontend, with incremental streaming, reassembling exactly. That
is the whole stack, not a component.

On a Tesla T4, fp16, Qwen2.5-0.5B (630M params as loaded -- 494M plus the untied
`lm_head` copy):

```
"The capital of France is"  ->  " Paris. It is the largest city in Europe and the
                                 second largest in the world..."
"Water boils at"            ->  " 212 F and ice melts at 32 F. What is the number
                                 of degrees by which the water..."
"def add(a, b):"            ->  "
    return a + b

def subtract(a, b):
                                 
    return a - b

def multiply(a,"
```

Why this matters more than any benchmark: paging, block tables, RoPE (rotate-half
convention), GQA head expansion, RMSNorm, SwiGLU, batched prefill, batched decode,
fp16 numerics, and sampling must **all** be simultaneously correct for a
pretrained model to produce this. The toy model in `tests/` has random weights, so
it can only ever prove self-consistency -- a systematic error shared by the code
under test and its reference would pass every one of those tests. This cannot.

Note the content is the 0.5B model's own knowledge, not a claim about the engine:
Paris is not the largest city in Europe. Fluency and structure are what is being
verified, not factuality.

Reproduce:

```bash
python bench/real_model_check.py --model artifacts/qwen05b --device cuda
```

## Batched decode (what the throughput claim rests on)

The executor runs **all resident decoding sequences in one forward pass**.
Sequences are concatenated along a single token dimension, so every projection
and the MLP become one large GEMM; attention stays per-sequence over its own
paged KV, gathered into a right-padded tensor and masked.

Measured on this CPU, with the mechanism ablated and nothing else changed:

| | T4 out tok/s | CPU out tok/s |
|---|---|---|
| `helios_full` | **1878.6** | 409.4 |
| `baseline_unbatched_executor` (one pass per sequence) | 598.0 | 183.1 |
| `baseline_static_batch_8` | 1256.3 | 361.2 |
| `baseline_hf_loop` (no engine, no KV reuse) | 303.1 | 71.6 |

**3.14× from decode batching alone on the T4** (2.24× on CPU), and continuous
batching beats static batching by 1.50× there (1.13× on CPU). Every mechanism
pays more on the GPU, which is what the theory predicts. On a Tesla T4, the fused Triton kernel is a further **5.26×**
over this PyTorch path for paged decode attention (32 seqs × 512 ctx), verified
by 9/9 parity tests against it. Mean resident decode batch was 11.6 (max 24) — reported
because the win is proportional to it. Prefill batching (1.16× TTFT) and the
prefix cache (1.49× throughput) are measured on their own regimes; see
`BENCHMARKS.md`.

An earlier version of this build ran one sequence per forward pass and therefore
showed *no* batching win; `docs/BENCHMARKS.md` said so at the time. That was a
property of the implementation, not of the hardware: the fix needed no GPU, only
batched GEMMs. Worth stating because the wrong conclusion — "you need a GPU to
show this" — was documented here as fact for a while.

Still true: this is CPU, on a toy model, so absolute tokens/second mean nothing.
The **ratios between ablations** are the result, and the parity tests
(`test_batched_decode_matches_sequential`, `test_batched_decode_is_order_invariant`,
`test_engine_output_unchanged_by_decode_batching`) are what make them
trustworthy — a batching bug would otherwise mix sequences' KV and produce
plausible, wrong output.

## What the benchmarks do and do not show

`docs/BENCHMARKS.md` is generated from JSON artifacts by `bench/report.py` and
never hand-edited (spec section 11). It reports CPU measurements on a toy
random-weight model. It characterises **relative** cost of engine mechanisms.

It is not evidence about GPU throughput, quantized inference, or any comparison
with another engine. Absolute tokens/second numbers are meaningless here — the
model is a toy.

## Disaggregation: what "built" means here, stated harder than before

**`core/disagg.py` is imported by exactly one non-test module: the simulation
harness.** The engine, the scheduler, and the runner never call it. Grep it:

```bash
grep -rn "disagg" --include="*.py" python/ | grep -v core/disagg.py
```

So it is **a verified specification, not a shipped feature**, and it does not
belong in the same sentence as paged attention. An earlier version of this
document listed it as "Built" in the same table as things that run on every
request, which overstated it. Corrected.

What it actually is: a state machine with a fault model, checked by the same
harness that checks the scheduler, whose *design* is the deliverable. That is real
engineering — a protocol with partial-failure semantics worked out and
mechanically verified before any transport exists is worth more than transport
code with the semantics left implicit — but it is not a feature a user can turn
on, and calling it one would be the exact failure this document exists to prevent.

Why it was not wired in anyway: with one device there is no transfer to perform.
Blocks are already where they need to be. Routing them through a transfer FSM
would be theatre — a code path that exists to make a claim true rather than to do
work.

With that framing fixed, here is what is genuinely verified:

- **A 7-state FSM** (`core/disagg.py`) with an explicit legal-transition table.
  The table is the specification — an FSM that tolerates an unlisted edge
  specifies nothing — so an illegal transition raises rather than being absorbed.
- **The SENT/RECEIVED distinction.** Collapsing them (which an async/await
  formulation does by default) makes "sender finished, receiver never
  acknowledged" *unrepresentable* and therefore untestable. That gap is where
  every interesting fault lives.
- **Five fault kinds**: receiver OOM, link timeout, checksum mismatch, lost ack,
  mid-flight abort. Each returns blocks from **both** partitions, because handling
  one side leaks the other — and a leak is invisible until the pool is exhausted,
  which looks like a capacity problem rather than a bug.
- **DST integration.** Half of all seeds run the FSM alongside the scheduler under
  randomised faults, asserting FSM invariants D1–D6 plus two pool invariants the
  FSM cannot check itself. Coverage measured across 200 seeds: every fault kind
  reached, retry in 72 seeds, re-prefill fallback in 56.
- **Detection verified, not just coverage.** Reintroducing the classic
  partial-failure leak (`fail()` returning the sender's blocks but not the
  receiver's) is caught by **31/60 seeds** with a diagnostic message naming the
  exact accounting discrepancy. Making an abort legal after SENT is caught by
  15/60. This distinction matters because a coverage gap was already mistaken for
  a passing test once in this project — see the copy-on-write note in `DST.md`.
- **The transfer-versus-re-prefill decision** computed from bandwidth rather than
  assumed, because the answer flips with the hardware: for a 262 MB KV cache,
  transferring wins by ~11× over PCIe 3.0 and *loses* over 10 GbE. A design that
  always transferred would be strictly worse than no disaggregation at all on the
  wrong side of that line.

An error worth recording: the first version of that arithmetic put the KV at
2.6 GB instead of 262 MB — a factor of ten — which inverted the PCIe verdict. It
was caught by a test written against the real formula, not by review. An
arithmetic slip in a comment is still a claim with no reproduction behind it.

## What a deliberate audit of this repo found

Run after everything above was written, tested, and committed — 263 tests green,
5,000 DST seeds green. The point of recording it: a green suite is not evidence
that the *claims* are true, only that the code does what the tests say. Four of
these five were claim errors, not code errors, and no test was failing.

| Found | Severity | Status |
|---|---|---|
| `QuantLinear` memoised the dequantized fp weight, so after one forward pass an int4 layer held **26% MORE memory than the fp16 layer it replaced** — while `stored_bytes()` reported a 3.8× saving | **Critical.** Every quantization memory claim in the repo was false in practice | Fixed; caching is now opt-in and off by default, and `resident_bytes()` exists so the two can never be conflated again |
| The `with_spec_decode` ablation ran at a batch size above the speculation gate, so **0 of 23 decode steps actually speculated** | **High.** A row labelled "speculation" measured scheduling variance | Fixed; a `spec` suite caps the batch to 8 so the mechanism is on the critical path |
| `memory_overhead()` reported only weights, omitting the **entire second KV cache** QASSD allocates | **Medium.** The quoted number was true and incomplete, which is harder to catch than a false one | Fixed; `engine.memory_report()` returns both costs in one dict |
| `docs/SCOPE.md` listed the KV transfer FSM as "Built" beside paged attention, when **nothing outside the test harness imports it** | **Medium.** Overstated a verified design as a shipped feature | Fixed; relabelled, with the grep that proves it |
| A benchmark run was contaminated by a concurrent job on the same machine, inverting the γ=2 / γ=4 ordering | Low | Re-run on an idle machine; the inversion disappeared |

Each fix carries a regression test, and each of those tests was **mutation-checked**
— the bug it exists to catch was reintroduced deliberately and the test was
confirmed to fail. Worth noting which one caught the accept-logic mutation: the
symmetric-speculation parity tests all *passed* it, because at α = 1.0 the draft
token and the verified token are the same, so the bug is invisible. Only
`test_asymmetric_speculation_matches_the_target_exactly` failed. A test suite can
be large and green and still have no coverage of the thing that matters.

## Defensible claims

Adapting the wording spec section 12.2 pre-approved:

> Implemented a from-scratch LLM serving engine with paged attention, a
> copy-on-write paged KV allocator, iteration-level continuous batching with a
> batched decode executor (**2.2x** over one-forward-pass-per-sequence, **5.7x**
> over a naive generation loop, and **1.13x** over static batching, each
> measured as a single-mechanism ablation), chunked prefill, a radix-trie prefix
> cache, and an OpenAI-compatible API.
> Validated the scheduler and allocator with a deterministic simulation testing
> harness (TigerBeetle/FoundationDB style) under adversarial fault injection:
> 50,000 seeds, seven invariants checked every step, every failure replayable
> from its seed. **The harness found 15 real bugs**, including a
> copy-on-write accounting error that overshot the memory watermark and six
> distinct admission/eviction livelocks — all documented with reproductions in
> `docs/DST.md`. Verified correctness by bit-exact parity: chunked vs single-shot
> prefill, speculative vs non-speculative decoding, and paged vs dense attention.

And what the quantization and speculation work adds:

> Implemented W4A16 INT4 weight quantization with AWQ activation-aware channel
> scaling (**1.58× lower per-layer output error than round-to-nearest**, measured
> across 168 layers of Qwen2.5-0.5B) and an INT8 paged KV cache with per-token
> scales (**1.94× smaller**, 3.4× more resident blocks at a fixed byte budget).
> Used them to build quantization-asymmetric self-speculative decoding — drafting
> from a 4-bit view of the target's own weights — and **measured the acceptance
> rate the design depends on: α = 0.6548**. That cleared the pre-committed kill
> threshold of 0.6 but missed the 0.78 design target, and showed the optimal draft
> length is 2 rather than the planned 4, with γ=8 a net loss. Output is
> **bit-identical** to non-speculative decoding, asserted by test, because the
> verifier decides which tokens commit.

Not defensible, and not claimed anywhere in this repo: beating vLLM (never run),
any *measured* speedup from quantization (no int4 GEMM — all such figures are
labelled modelled), heterogeneous multi-GPU serving, disaggregated serving across
real devices (the FSM is verified; the transport is simulated).

## The remaining honest gaps, in dependency order

1. **A fused int4 GEMM** (Triton, W4A16 dequant-in-epilogue). This is the single
   thing standing between the α measurement and a real speedup claim. Everything
   numerical is already verified against it as an oracle.
2. **vLLM baseline.** `bench/vllm_baseline.py` is written with the controls matched
   — same model dir, same dtype, same KV byte budget, same seeded arrival trace,
   warm-up discarded — and has never been run. It is designed so that losing is a
   reportable outcome.
3. **α on a 7B model.** The 0.5B result is marginal; the hypothesis that it
   improves with scale is untested.
4. **Two devices** → real KV transport behind the existing FSM. The fault taxonomy
   and block accounting are already there and already verified, so this is
   plumbing rather than design.
5. **Rust frontend.** Cut by section 12.2 and still cut. The ExecStep seam that
   would make it possible is preserved, which was the point.

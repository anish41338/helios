# Speculative decoding — what is implemented, and what QASSD would require

Spec section 7. **The quantization asymmetry that defines QASSD is NOT
implemented.** This document explains the design, states precisely what this
repository contains, and says what would be needed to finish it — so that nobody
(including a future reader of the spec) mistakes one for the other.

## The spec's idea

Standard speculative decoding needs a small draft model: extra VRAM, extra
loading, distribution mismatch. Self-speculation reuses the target model. QASSD's
twist is to make the draft path cheap by **reading the same weights at lower
precision**:

- **draft**: W4A4 (4-bit weights, 4-bit activations) + int4/int8 KV cache — much
  cheaper in both bandwidth and compute
- **verify**: W4A16 (same 4-bit weights, fp16 activations) + fp16 KV — the
  accurate distribution

One weight set, two read precisions, two KV views. The memory cost of
speculation is only the quantized KV shadow.

## What is implemented here

The full speculative **control flow**, in `exec/runner.py::_run_speculative`:

1. draft γ tokens by γ sequential cheap forward passes
2. verify all γ positions in **one** parallel forward pass
3. accept the longest prefix where the draft matches the verified argmax
4. always emit a bonus token from the first rejected position, so a rejection
   never stalls the sequence
5. roll back KV for rejected positions

Plus the scheduler-side bookkeeping in `core/scheduler.py`: `num_drafted` /
`num_accepted` accounting, a sliding acceptance window, KV truncation on
rollback, and adaptive gating (§7.3).

### The caveat, stated precisely

Draft and verify read the **same fp32 weights**, because there is no quantized
path in this build. Therefore:

- **acceptance is 1.0 by construction**, not by merit — the draft *is* the verify
- **there is no speedup**; it is strictly slower, since drafting is serial.
  The benchmark confirms this (`with_spec_decode` is ~9% slower)
- it says **nothing** about whether the spec's α ≥ 0.78 target is reachable

What it does establish is that the accept/rollback bookkeeping and dual KV
truncation are correct — the part most likely to corrupt output silently. Tests
assert **bit-identical** output versus non-speculative decoding for
γ ∈ {1, 2, 4, 8}, which is spec §7.2's single most important requirement:
speculation that changes outputs is a bug, not a speedup.

## Adaptive gating (implemented)

`_effective_gamma` disables speculation when:

- **batch size > 8** — at large batch, verify becomes compute-bound and the
  wasted draft compute is no longer free
- **measured acceptance < 0.5** over a sliding window of 64 outcomes

Spec §7.3 notes an interviewer will ask about this, and "we always speculate" is
the wrong answer. The DST harness exercises both extremes by forcing acceptance
to 0% and 100%.

## The economics

Speedup ≈

```
    (1 + α + α² + ... + α^γ) / (1 + γ · c_draft/c_verify)
```

where α is per-token acceptance and `c_draft/c_verify` the relative cost of a
draft pass. The spec targets α ≥ 0.78 at γ=4 with `c_draft/c_verify ≈ 0.35`,
giving ~2.1–2.4× on TPOT at batch 1.

In this build `c_draft/c_verify ≈ 1.0` (identical precision), so the formula
gives `(1+α+...+α^γ)/(1+γ)` = 1.0 at α=1 — no gain, which is exactly what is
measured. The formula and the measurement agree, which is at least a sanity check
on both.

**Where it breaks** (spec §7.3): α collapses at high batch sizes and on
high-entropy outputs (creative writing, code with many valid continuations).

## What finishing QASSD requires

In dependency order, and note step 2 is a **kill gate**:

1. **W4A16 path** — AWQ/GPTQ int4 unpacking with per-group fp16 scales,
   dequantized in the GEMM epilogue (never materialising fp16 weights). This is
   the verify path and is needed regardless.
2. **Measure α first.** Quantize activations to int4 for the draft and measure
   acceptance on a 1B model *before* building anything else. Spec §14 says kill
   the feature if α < 0.6. Building the full path first and measuring later is
   the expensive mistake.
3. **W4A4 draft path** — per-token int4 activation quantization, int32
   accumulate, requantize.
4. **Quantized KV shadow** — per-head, per-token asymmetric int8 (then int4),
   with scale+zero-point stored per block. Written at commit time as a *derived*
   view of the fp16 cache, never a second source of truth.
5. **Dual-buffer rollback** — `kv_fp16` (authoritative, verify) and `kv_quant`
   (derived, draft) must truncate **atomically**. Spec §7.4 warns that a mismatch
   here produces subtly wrong drafts and a silent α collapse — a bug that looks
   like a performance problem and wastes a week.

All of this needs a GPU (see `SCOPE.md`). The existing bit-parity test is the
oracle any of it would be validated against: after adding quantization, output
will no longer be bit-identical (that is expected — quantization changes the
draft), but the *committed* tokens must still match the non-speculative W4A16
path exactly, because rejection sampling preserves the target distribution.

That last point is the one to be able to defend cold: speculation is correct not
because the draft is good, but because the *verifier* decides, and the acceptance
rule is constructed so the accepted distribution equals the target's.

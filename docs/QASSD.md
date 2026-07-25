# Quantization-asymmetric self-speculative decoding — built, and measured

Spec section 7. **Implemented and measured.** This document records the design,
the measurement that decides whether it was worth building, and the verdict.

The single most important number:

> **α = 0.6548** on Qwen2.5-0.5B, int4 AWQ draft against fp32 verify, γ=4,
> 336 drafted tokens over 84 verify passes.
>
> Spec section 14's kill gate is α ≥ 0.6. **The gate passes.** The spec's own
> section 7.3 *target* was α ≥ 0.78. **That target is not met.**

Reproduce:

```bash
python bench/measure_alpha.py --model artifacts/qwen05b --gamma 4
```

Artifacts: `artifacts/alpha.json` (AWQ), `artifacts/alpha_rtn.json` (RTN).

## The idea

Standard speculative decoding needs a small draft model: extra VRAM, extra
loading, distribution mismatch. Self-speculation reuses the target model. QASSD's
twist is to make the draft cheap by **reading the same weights at lower
precision**:

- **draft** — 4-bit weights, int8 KV shadow. Decode is memory-bandwidth-bound, so
  a quarter of the weight bytes is close to a quarter of the cost.
- **verify** — full-precision weights and KV. The accurate distribution.

One weight set conceptually, two precisions physically.

## Why it is safe regardless of how good the draft is

The property to be able to defend cold: **speculation is correct not because the
draft is good, but because the verifier decides.** An accepted token is one the
target's argmax agreed with; a rejection emits the target's own argmax. So the
committed sequence is the target's greedy sequence, whatever the draft proposes. A
bad draft costs throughput and nothing else.

That is asserted, not argued:
`tests/quant/test_qassd.py::test_asymmetric_speculation_matches_the_target_exactly`
runs the same prompts with and without the int4 draft and requires **identical
token ids**. A violation would mean the engine silently serves the int4 model's
output while reporting the fp model's — the worst failure mode this engine has.

## What is implemented

`exec/qassd.py` and `exec/runner.py::_run_speculative`:

1. draft γ tokens with the **int4** model, into its own quantized KV shadow
2. verify all γ positions in **one** full-precision forward pass
3. accept the longest prefix where the draft matches the verified argmax
4. always emit a bonus token from the first rejected position, so a rejection
   never stalls the sequence
5. re-sync the draft's KV over the committed span

Step 5 is the subtle one. The draft ran ahead and wrote keys for every token it
proposed; past the first rejection those describe a continuation that was thrown
away. Paged storage has no notion of truncation — the positions are simply
overwritten later — so without an explicit re-sync the draft would attend over a
context the engine never committed to. Silently. Spec section 7.4 flags exactly
this as the bug that "looks like a performance problem and wastes a week", because
the symptom is α quietly collapsing.

Both modes are kept:

| Mode | Draft weights | α | Purpose |
|---|---|---|---|
| symmetric (`quantized_draft=False`) | same as verify | 1.0 by construction | correctness oracle — any divergence is a bookkeeping bug |
| asymmetric (`quantized_draft=True`) | int4 AWQ | **measured 0.6548** | the real thing |

Keeping the symmetric mode is what makes the bookkeeping testable: with identical
precisions, output must be bit-identical, so there is no quantization noise to
hide a bug behind.

## What it costs — the complete accounting

Enabling QASSD is an **increase** in memory on two axes, not one. The weight-only
figure is the one that is easy to quote and it is incomplete:

| | |
|---|---|
| Draft weights (int4 copy of the same weights) | **+41%** on weights |
| Draft KV shadow (a second, quantized KV cache) | **+28%** of the main cache |

Both are real allocations the engine makes at construction. `engine.memory_report()`
returns them together, deliberately, so they cannot be quoted separately by
accident:

```python
{'target_weight_bytes': ..., 'main_kv_bytes': ...,
 'draft_weight_bytes': ..., 'draft_kv_bytes': ...,
 'total_overhead_ratio': 1.28,      # <- the number to quote
 'weight_overhead_ratio': 1.41}     # <- weights only; incomplete on its own
```

The second KV cache is not optional. The draft's keys and values are computed
from int4 weights, so they are numerically *not* the target's and cannot share
storage — sharing would corrupt the target's context with approximate keys, and
would look like a working optimisation right up until output quality was measured.
That cache is spec §7.4's "quantized KV shadow", and it is quantized for exactly
the reason the spec wanted it to be: a second full-precision cache would halve the
concurrency the engine just paid for.

**An earlier version of this document reported only the weight figure.** That
understated the cost, and the omission is the kind that survives review because
the number quoted was true — just not the whole cost. Now pinned by
`test_memory_report_includes_the_draft_kv_shadow`.

A deployment that only wants W4A16 serving keeps the int4 copy *alone* and gets a
real reduction — but that configuration has no speculation in it, and conflating
the two is the easiest misrepresentation available here.

## Why measuring α on a CPU is legitimate

α is the probability that the int4 model's argmax equals the fp model's argmax
given the same context. That is a property of two weight matrices and a prompt. It
does not depend on how fast either forward pass runs, so **it transfers to a GPU
unchanged.**

What does *not* transfer is the speedup. Realising it needs an int4 GEMM that
actually reads a quarter of the bytes; this build dequantizes to fp and then calls
a normal GEMM, so on CPU the int4 path is *slower* than fp32. Every speedup figure
below is therefore labelled **modelled**, per spec section 19.6.

This separation is the reason the feature could be evaluated at all without
hardware — and it is the argument for having built the numerical half first.

## The measurement

Qwen2.5-0.5B, 8 prompts spanning factual recall, arithmetic, code, and prose
(acceptance is strongly context-dependent, so a single prompt family would give a
number that does not generalise). AWQ calibrated on 4 **disjoint** prompts —
calibrating on the evaluation text is how a quantization method gets reported as
better than it is.

```
drafted            : 336
accepted           : 220
ALPHA              : 0.6548
tokens/verify pass : 3.12  (measured)
accepted-run histogram: {0: 15, 1: 11, 2: 7, 3: 9, 4: 42}
```

### The histogram is the interesting part

It is strongly **bimodal**: half the passes (42/84) accept all four drafted
tokens, and 15/84 accept none. That falsifies the independence assumption in the
standard speedup model — acceptances are positively correlated, because a context
the draft finds easy stays easy for several tokens.

The consequence is favourable and worth being precise about:

| | tokens per verify pass at γ=4 |
|---|---|
| independence model at α=0.6548 | 2.55 |
| **measured** | **3.12** |

So the textbook model **understates** the real value by 22% here. Using the
measured tokens/pass, the modelled speedup at γ=4 is 3.12/2.00 = **1.56×** rather
than 1.27×. The independence model is conservative, which is the safe direction
for it to be wrong in.

### Modelled speedup, from measured α

Cost assumption: a draft pass costs 0.25 of a verify pass (the bandwidth-bound
ideal for 4-bit against 16-bit weights). Optimistic — attention over the KV cache
is not quantized on the verify side and does not shrink, and kernel overhead does
not either.

| γ | E[tokens/pass] | cost | modelled speedup |
|---|---|---|---|
| 1 | 1.65 | 1.25 | 1.32× |
| 2 | 2.08 | 1.50 | **1.39×** |
| 4 | 2.55 | 2.00 | 1.27× |
| 8 | 2.83 | 3.00 | **0.94× — a loss** |

**Best γ is 2, not the spec's 4.** And at γ=8 speculation costs more than it
returns. This is a concrete, measured correction to the spec's parameter choice,
and it is exactly why the adaptive gating in section 7.3 is machinery worth having
rather than an optimisation.

### Does AWQ actually help?

Yes, at the level that matters. Mean per-layer relative output error across 168
quantized linears:

```
RTN  0.01167  ->  AWQ  0.00739     1.58x lower
```

`bench/measure_alpha.py --no-awq` measures α with plain round-to-nearest for the
end-to-end comparison; `artifacts/alpha_rtn.json` records it.

## The verdict

The gate passes, so the feature is not killed. But the honest summary is
**marginal**:

- α = 0.655 against a target of 0.78 — the target is missed by a wide margin
- best γ is 2, giving ~1.4× modelled (or ~1.56× using measured tokens/pass)
- γ=8 is a net loss
- and it costs more memory, not less: +41% on weights **plus** a second KV cache

For a 0.5B model this is a poor trade. The reason to expect better on a larger
model is that quantization error at fixed bit-width falls with model size — bigger
models are more redundant, which is the empirical finding AWQ and GPTQ both rest
on. **That is a hypothesis, not a result, and it is not claimed anywhere.**
Measuring α on a 7B model is the obvious next experiment and it needs a GPU only
because of the model size, not because of the method.

What is genuinely established: the mechanism is correct (bit-identical output),
the cost is known, the gate has a real number behind it, and the spec's γ=4 choice
is measurably wrong for this model. **A negative-leaning result that was actually
measured is worth more than an unmeasured positive one.**

## Adaptive gating — and the measurement that proved it earns its keep

`_effective_gamma` disables speculation when:

- **batch size > 8** — at large batch, verify becomes compute-bound and the wasted
  draft compute stops being free
- **measured acceptance < 0.5** over a sliding window of 64 outcomes

Both fire in practice, and the second one is doing real work. Measured on the toy
model, where an int4 draft agrees only 12.5% of the time:

```
47 decode steps, 1 with gamma>0  ->  the gate disabled speculation after ONE step
```

That matters because ungated speculation on a bad draft is catastrophic, not
merely unhelpful. Measured on this CPU at batch 8, 16 requests:

| configuration | out tok/s | vs no speculation |
|---|---|---|
| speculation off | **200.4** | — |
| symmetric, γ=2 | 62.9 | **3.19× slower** |
| symmetric, γ=4 | 65.7 | **3.05× slower** |
| QASSD γ=2 (gate active) | 141.7 | 1.41× slower |

Two things to read off that table.

**Speculation as implemented forfeits batched decode.** The draft loop is
inherently serial and runs per sequence, so with 8 resident sequences a
speculative step costs 8×(γ+1) unbatched forward passes where a normal step costs
one batched pass. On a CPU that trade is never worth it — the batching win is
larger than anything speculation can return. This is a property of *this*
implementation and of CPU economics, not of speculative decoding.

**The gate converts a 3.2× disaster into a 1.4× tax.** QASSD is faster than
symmetric speculation here not because the int4 draft is better — it is far worse,
α = 0.125 — but because the gate *noticed* and stopped. That is the mechanism that
makes it safe to ship a draft whose quality you have not characterised on every
workload: correctness comes from the verifier, and throughput protection comes
from the gate.

The measured α of 0.655 on the real model sits uncomfortably close to the 0.5
floor, which means the gate will fire on hard prompts. That is intended.

Pinned by `test_gate_disables_speculation_when_the_draft_is_bad`. The DST harness
additionally exercises both extremes by forcing acceptance to 0% and 100%.

### A benchmark that was measuring nothing

Until this was checked, the `with_spec_decode` ablation ran at the default
`max_num_seqs=64`, where the mean decode batch is 12–16. The batch-size gate
therefore fired on **every** decode step — measured, 0 of 23 steps had γ>0 — so
the row labelled "speculation" was measuring speculation being switched off. The
~9% difference it reported was scheduling variance.

The `spec` suite caps the batch to 8 so the mechanism is actually on the critical
path. Same class of error as the prefix cache on a short shared prefix and the
INT8 KV cache in a non-binding pool: **a mechanism benchmarked outside its regime
reads as noise or as a regression.** Third occurrence in this project, which is
why the fix is a dedicated suite rather than a note.

## Still not built

- **W4A4** — 4-bit *activations* as well as weights. The T4 has no fast int4
  tensor path, so a W4A4 draft would not actually be cheaper than the verify on the
  hardware available, and the premise fails. This is a hardware constraint, not a
  missing implementation.
- **A fused int4 GEMM.** Without it every speedup here stays modelled. This is the
  one remaining piece between the measurement and a real speedup claim.
- **Dual-buffer atomic rollback.** The current design re-syncs the draft cache by
  recomputation, which is correct but costs a forward pass over the committed span.
  Truncating both caches atomically would be cheaper.

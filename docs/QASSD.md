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

## What it costs

`+51% weight memory`, measured, not estimated:

```
target 2403.9 MiB fp32  +  draft 1223.9 MiB int4  =  1.51x the target alone
```

Stated because the opposite is easy to imply. "4-bit quantization" sounds like a
reduction; QASSD is an *addition*, because both precisions must be resident. A
deployment that only wants W4A16 serving keeps the int4 copy alone and gets
**1.96× smaller** — but that is a different configuration, and it has no
speculation.

(1.51× rather than 1.25× because the master here is fp32, not fp16, and because
`lm_head` stays full precision. On a GPU in fp16 the ratio would be closer to
1.3×.)

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
- and it costs +51% weight memory

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

## Adaptive gating (implemented)

`_effective_gamma` disables speculation when:

- **batch size > 8** — at large batch, verify becomes compute-bound and the wasted
  draft compute stops being free
- **measured acceptance < 0.5** over a sliding window of 64 outcomes

The measured α of 0.655 sits uncomfortably close to that 0.5 floor, which means
the gate will fire on hard prompts. That is the intended behaviour.

The DST harness exercises both extremes by forcing acceptance to 0% and 100%.

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

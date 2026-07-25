# Running the GPU track (Kaggle T4)

**Complete on a Tesla T4** (sm_75, CUDA 12.8, Triton 3.6.0). Both gates passed:
the Triton kernel is **5.1–5.6×** over the PyTorch paged path across three
independent runs, and Qwen2.5-0.5B generates coherent text in fp16. Anything still
listed as *not written* below remains unclaimed until its own gate passes on a
device (spec section 19.3).

## What the GPU runs found: four bugs, three invisible on CPU

The Triton gate passing first try was the expected outcome. The value was in the
bugs that only a real device and a real checkpoint could expose.

1. **Qwen2 attention biases silently dropped.** Qwen2 puts biases on q/k/v
   projections; Llama does not. The model hardcoded `bias=False` and the loader
   **skipped checkpoint tensors with no matching parameter, in silence** — so the
   model loaded "successfully" with three bias tensors per layer discarded. It
   guarded *missing model params* but not *ignored checkpoint params*, the same
   failure class its own docstring warns about. Now: `attention_bias` inferred
   from the architecture, and the loader raises on any unmapped tensor with an
   explicit allowlist for genuine buffers.

2. **The runner swallowed the real exception.** `ModelRunner.run` converts
   `RuntimeError` into an `ExecFault` so the scheduler can preempt and retry
   rather than commit tokens from bad KV — correct in production, useless for
   diagnosis. A forward pass that always fails became an infinite retry loop
   whose only symptom was an empty result list. Now `real_model_check.py` runs a
   smoke test with exceptions unhandled *before* involving the scheduler.

3. **Model weights were never moved to the device.** `HeliosModel(device=...)`
   only records the device for tensors it creates at runtime; its
   `nn.Embedding`/`nn.Linear` submodules are built on CPU. The loader called
   `.to(dtype)`, which is not `.to(device)`. **Structurally invisible on a
   CPU-only machine**, where requested and default device agree — so the
   regression test asserts the invariant (every parameter on the device the engine
   builds inputs on) rather than the symptom.

4. **fp32 on the GPU.** The engine ran real weights in fp32. Now fp16 on CUDA,
   decided once and shared by the weights, the KV cache, and the block-size
   arithmetic — deriving those separately is how a cache sized for fp32 ends up
   holding fp16 tensors.

Three of the four were undetectable without a GPU. That is the argument for
renting one for an afternoon even when the CPU suite is green: **178 passing tests
did not catch any of them.**

## What needs a GPU, and what it unlocks

| Item | Status | Unlocked by a T4? |
|---|---|---|
| Triton paged-attention kernel | **VERIFIED on T4: 5.26x**, 9/9 tests | done |
| Real model, meaningful absolute tok/s | **VERIFIED**: Qwen2.5-0.5B generates coherent text in fp16 | done |
| W4A16 / AWQ quantization | **Built.** Numerically verified on CPU: 3.8× vs fp16, AWQ 1.58× lower layer error | Only the *speedup* — see below |
| QASSD α measurement (the §14 kill gate) | **DONE on CPU: α = 0.6548, gate PASS.** α is a property of two weight matrices, so it did not need a GPU | already answered |
| INT8 quantized KV cache | **Built.** 1.94× smaller, 2.31× more resident sequences measured | Only the *throughput* win |
| Fused int4 GEMM | not written | **Yes** — this is the last piece between α and a real speedup |
| vLLM baseline | **written, never run** (`bench/vllm_baseline.py`) | **Yes** |
| α on a 7B model | not measured | **Yes** — needs the memory, not the method |
| W4A4 draft path | not written | **No** — T4 has no fast int4 path, so the draft would not be cheaper than the verify and the premise fails |
| Prefill/decode disaggregation transport | FSM built and DST-verified; transport simulated | Only on Kaggle's **2×T4** |

### What the CPU could and could not settle about quantization

The scoping decision that changed: "quantization needs a GPU" conflated a *speed*
claim with a *numerical* one.

- The **numerical** half is device-independent arithmetic and was fully settled
  here — compression ratios, error bounds, AWQ versus RTN, and most importantly
  **α = 0.6548**, the acceptance rate the whole QASSD design rests on. A GPU would
  not have made that number any more true.
- The **speed** half genuinely needs one. `QuantLinear` dequantizes and then calls
  a normal GEMM, so on CPU it is *slower* than fp32 — real work, no bandwidth wall
  to relieve. Every quantization speedup in this repo is labelled **modelled**.

The INT8 KV cache shows the split cleanly: **2.31× more resident sequences and
1.62× better TTFT** (both measured, in a pool small enough to bind), and **27%
worse throughput**, because dequantization costs more than it saves when the KV
read is not the bottleneck. On a T4 it should be — that is a prediction from the
same roofline argument that correctly predicted the GPU/CPU pattern in the
ablation table, and it is the next thing worth measuring.

## Step 1 — Kaggle session setup

New notebook → **Settings → Accelerator → GPU T4 x2** → Internet **On**.

## Step 2 — one cell

```python
!git clone -q https://github.com/<you>/helios.git
!cd helios && pip install -q triton && python bench/kaggle_gpu_run.py
```

That script is deliberately fail-fast and ordered so a broken rung is never
hidden by a later result:

1. **Device report** — every number below gets provenance.
2. **CPU test suite** — correctness before measurement. Aborts if it fails.
3. **Triton parity gate** — compares the kernel against the PyTorch paged path
   (itself already pinned against a dense reference on CPU). Also asserts the
   kernel is *faster*; a fused kernel that loses to gather+SDPA means the fusion
   is wrong and is not worth shipping.
4. **Real model** (Qwen2.5-0.5B) end to end. Coherent output here is the single
   strongest signal that paging, RoPE, GQA, and sampling are all correct together
   — the toy model has random weights, so it can only ever prove self-consistency.
5. **Benchmark suites** (`all`, `prefill-heavy`, `prefix-cache`) → JSON artifacts
   → regenerated `docs/BENCHMARKS.md`.

Runtime: roughly 10–15 minutes, most of it the model download.

## Step 3 — bring the results back

```python
!cd helios && git add artifacts docs/BENCHMARKS.md && \
  git -c user.email=you@example.com -c user.name=you \
  commit -m "GPU run: Triton parity verified on T4" && git push
```

## Step 4 — only then, update the claims

For each gate, **only if it passed**: record the device and measured number in
the relevant source file's status banner, move the row in `docs/SCOPE.md` from
*NOT built* to *Built*, and add it to the README's mechanisms table.

If a gate fails, leave every claim alone and treat the pytest output as the bug
report. A failing kernel is a normal outcome; a claimed-but-unverified one is not
recoverable once it is on a résumé.

## Known risks

- **Triton version drift.** Kaggle's preinstalled `triton` may not match the API
  used in `triton_attn.py` (`tl.load` masking, `tl.constexpr` signatures are the
  usual breakages). If it fails to compile, the fix is local to that one kernel;
  the PyTorch path is unaffected and the engine keeps working.
- **fp16 tolerances.** The gate uses `atol=rtol=2e-2`, appropriate for fp16
  tensor-core accumulation. Do **not** loosen them to get a pass (spec section
  19.2) — if the kernel cannot hold them, it is wrong.
- **Session limits.** 12 h per session, ~30 h/week. The script is a single
  entry point precisely so a dropped session costs one re-run, not a lost
  workflow.

## Next, in dependency order

1. **A fused int4 GEMM** (Triton, dequant in the epilogue so fp weights are never
   materialised). The single item standing between the measured α and a real
   speedup claim. The correctness oracle already exists — `QuantLinear`'s
   dequantize-then-GEMM path is pinned by `tests/quant/test_quant.py` — so this is
   a kernel with a reference waiting for it, exactly as the attention kernel was.
2. **Re-run the INT8 KV ablation on the T4.** The concurrency win is already
   measured (2.31×); what is unknown is whether halving the KV read pays for the
   dequantization once bandwidth is the binding constraint. A clean yes or no.
3. **vLLM baseline** on the same model and hardware. `bench/vllm_baseline.py` is
   written with the controls matched — same model dir, dtype, KV byte budget,
   seeded arrival trace, discarded warm-up — and designed so that losing is a
   reportable outcome. This is the comparison an interviewer reaches for.
4. **α on a 7B model.** The 0.5B result clears the gate but misses the 0.78
   target. Quantization error at fixed bit-width falls with model size, so a larger
   model should do better — untested, and not claimed. Needs GPU *memory* rather
   than GPU speed.
5. **Disaggregation** across the 2×T4 — only if 1–4 land. The FSM, the fault
   taxonomy, and the block accounting are already built and DST-verified, so this
   is transport plumbing against a test target that already exists.

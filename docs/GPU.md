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
| W4A16 / AWQ quantization | not written | **Yes** |
| Real model, meaningful absolute tok/s | **VERIFIED**: Qwen2.5-0.5B generates coherent text in fp16 | done |
| vLLM baseline | not written | **Yes** |
| QASSD α measurement (the §14 kill gate) | not written | **Yes** — a few hours |
| W4A4 draft path | not written | **No** — T4 has no fast int4 path, so the draft would not be cheaper than the verify and the premise fails |
| Prefill/decode disaggregation | not written | Only on Kaggle's **2×T4** |

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

## After the kernel: suggested order

1. **W4A16 AWQ** loading + GEMM, then re-run the ablation suite. Real memory
   win → higher concurrency, which the existing harness will show.
2. **vLLM baseline** on the same model and hardware. This is the comparison the
   spec asks for and the one an interviewer will ask about.
3. **Measure α** for speculation on a 1B model. This is a *kill gate*, not a
   build: spec section 14 says abandon QASSD if α < 0.6. Deciding not to build it,
   with a measurement behind the decision, is a better outcome than building it
   badly.
4. **Disaggregation** across the 2×T4 — only if 1–3 land. The KV transfer FSM's
   fault taxonomy already exists in the DST harness, so it has a test target
   waiting.

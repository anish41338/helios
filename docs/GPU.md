# Running the GPU track (Kaggle T4)

Everything in this document is **written but unverified** until the parity gate
below passes on a real device. Nothing here may be claimed before that (spec
section 19.3). `docs/SCOPE.md` still lists the Triton kernel under *NOT built*
for exactly this reason — moving it is the last step, not the first.

## What needs a GPU, and what it unlocks

| Item | Status | Unlocked by a T4? |
|---|---|---|
| Triton paged-attention kernel | **VERIFIED on T4: 5.26x**, 9/9 tests | done |
| W4A16 / AWQ quantization | not written | **Yes** |
| Real model, meaningful absolute tok/s | blocked by a loader bug, now fixed -- re-run | **Yes** |
| vLLM baseline | not written | **Yes** |
| QASSD α measurement (the §14 kill gate) | not written | **Yes** — a few hours |
| W4A4 draft path | not written | **No** — T4 has no fast int4 path, so the draft would not be cheaper than the verify and the premise fails |
| Prefill/decode disaggregation | not written | Only on Kaggle's **2×T4** |

## Step 0 — push the repo somewhere Kaggle can clone

There is currently **no git remote**. Kaggle needs one:

```bash
gh repo create helios --private --source=. --push
# or: git remote add origin <url> && git push -u origin master
```

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

## Step 4 — only now, update the claims

If **and only if** the parity gate passed:

- `docs/SCOPE.md`: move *Triton/CUDA paged-attention kernel* from **NOT built**
  to **Built**, and record the measured speedup.
- `python/helios/exec/triton_attn.py`: replace the `STATUS: WRITTEN, NOT YET
  VERIFIED` banner with the device and speedup it was verified on.
- `README.md`: add the kernel row to the mechanisms table.

If it failed, leave every claim alone and treat the pytest output as the bug
report. A failing kernel is a normal outcome; a claimed-but-unverified kernel is
not recoverable once it is on a résumé.

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

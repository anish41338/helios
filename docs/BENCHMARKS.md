# HELIOS benchmarks

> **GENERATED FILE -- DO NOT EDIT.**
> Produced by `bench/report.py` from the raw JSON artifacts in `artifacts/`.
> Regenerate with the command shown under Provenance.

## What these numbers are, and are not

These are measurements from **this** repository on **this** machine, using a
small randomly-initialised model. They characterise the *engine's scheduling
behaviour* -- the relative cost of paging, batching, chunked prefill, and the
prefix cache -- and nothing else.

They are **not**:

- a comparison against vLLM, TensorRT-LLM, or any other engine (no such
  baseline was run; see `docs/SCOPE.md`)
- evidence about GPU performance (this build ran on CPU; there are no CUDA or
  Triton kernels)
- evidence about quantized inference, QASSD acceptance rates, or
  prefill/decode disaggregation (not implemented -- see `docs/SCOPE.md`)
- meaningful as absolute tokens/second, since the model is a toy with random
  weights

Per spec section 19.6, no figure here is taken from a paper, and per section
19.3 every figure traces to a JSON artifact listed below.

## Provenance

- commit: `61b46a5bab0310b2abe0ff9cc5dd26e58e52a2de`
- timestamp: 2026-07-25T13:27:36
- platform: Windows-11-10.0.26200-SP0
- processor: `Intel64 Family 6 Model 186 Stepping 3, GenuineIntel`
- python: 3.12.0
- torch: 2.5.1+cpu
- device: **cpu** (cuda_available=False)
- regenerate: `python bench/report.py --artifacts artifacts --out docs/BENCHMARKS.md`

> Clock/power state is not pinned on this machine, so run-to-run variation of a few percent is expected. Differences smaller than that should not be interpreted.

## Ablations

Each row differs from `full` by one mechanism, which is what makes the delta attributable (spec section 11).

| configuration | reqs | out tok/s | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | e2e p50 | goodput |
|---|---|---|---|---|---|---|---|---|
| `batch1` | 24/24 | 162.2 | 1.8636 | 3.3137 | 0.0063 | 0.0075 | 2.1704 | 0.42 |
| `full` | 24/24 | 406.7 | 0.4963 | 0.4965 | 0.0294 | 0.0351 | 1.1040 | 0.88 |
| `no_chunked_prefill` | 24/24 | 428.6 | 0.4142 | 0.4143 | 0.0304 | 0.0344 | 1.0423 | 0.88 |
| `no_prefix_cache` | 24/24 | 419.3 | 0.4454 | 0.4456 | 0.0299 | 0.0342 | 1.0651 | 0.88 |
| `small_kv_pool` | 24/24 | 362.6 | 0.4277 | 0.4278 | 0.0358 | 0.0390 | 1.1789 | 0.88 |
| `with_spec_decode` | 24/24 | 251.9 | 0.5208 | 0.5209 | 0.0507 | 0.0542 | 1.4538 | 0.62 |

## Baselines

`baseline_hf_loop` re-runs the whole sequence per token with no cache reuse -- the 'no engine at all' reference. `baseline_static_batch_N` drains each batch fully before admitting the next, which is the head-of-line waste continuous batching removes.

> The naive loop runs a deliberately smaller workload (it is quadratic in sequence length), so its throughput is **not** directly comparable to the rows above. Compare mechanisms, not absolute numbers across different workloads.

| baseline | reqs | out tok/s | TTFT p50 | TPOT p50 | e2e p50 | workload |
|---|---|---|---|---|---|---|
| `baseline_hf_loop` | 8/8 | 60.2 | 0.0139 | 0.0141 | 0.1184 | n=8 plen~32 olen~8 |
| `baseline_static_batch_8` | 24/24 | 349.5 | 0.0939 | 0.0142 | 0.3925 | n=24 plen~48 olen~20 |
| `baseline_unbatched_executor` | 24/24 | 177.3 | 0.4310 | 0.1071 | 2.6298 | n=24 plen~48 olen~20 |

## Interpretation

**Batched decode is the dominant win: 2.29x** (406.7 vs 177.3 out tok/s). `baseline_unbatched_executor` is the *same* engine -- same scheduler, same paging, same prefix cache -- with only the executor's decode batching ablated, so the delta is attributable to that one mechanism. Every resident sequence's projections and MLP collapse into single GEMMs; attention stays per-sequence over its own paged KV.

- vs `baseline_hf_loop` (no engine at all, no KV reuse): 6.8x, though on a smaller workload -- see the caveat below.
- `batch1` reaches TPOT p50 of 0.0063s versus `full`'s 0.0294s, but TTFT p50 of 1.8636s versus 0.4963s. One sequence at a time gets the whole machine, so its own tokens come fast while everyone else queues. This is the batching trade-off, measured.
- `with_spec_decode` is **slower** (251.9 vs 406.7 out tok/s), exactly as expected: draft and verify share the same fp32 weights here, so drafting is pure serial overhead with no cheaper draft path. See `docs/SCOPE.md` -- this measures the bookkeeping, not QASSD.
- `no_prefix_cache` differs from `full` by -12.6 out tok/s. The prefix cache saves *prefill work* on repeated prompts; with a short shared prefix and a toy model, that saving is small relative to total cost. `tests/scheduler` asserts the mechanism works by counting prefill tokens directly, which is a sounder check than a wall-clock delta this size.
- vs `baseline_static_batch_8`: **1.16x** (406.7 vs 349.5 out tok/s). Continuous batching wins by reusing a finished sequence's slot in the same iteration instead of idling until the batch's longest member completes. The margin is modest because these output lengths are only moderately skewed -- the wider the spread, the more head-of-line waste static batching suffers.
  Static batching's low TTFT is an artifact of enqueueing every request at t=0: with all work available immediately, no request waits for a later batch to start.
- Operating point: `full` averaged 11.6 resident decoding sequences per step (max 24). Batching pays only in proportion to this number, so it is reported rather than left implicit -- a throughput claim from a run averaging ~1 resident sequence would be measuring nothing.
- `baseline_hf_loop` reruns the full sequence every token with no KV reuse. It runs a smaller workload (it is quadratic in length), so treat it as a sanity floor, not a ratio.

## Scheduler counters (`full` configuration)

| metric | value |
|---|---|
| `helios_aborted_total` | 0 |
| `helios_empty_steps_total` | 1 |
| `helios_exec_faults_total` | 0 |
| `helios_finished_total` | 24 |
| `helios_kv_blocks_total` | 8192 |
| `helios_kv_blocks_used` | 25 |
| `helios_kv_utilization` | 0.0031 |
| `helios_preemptions_recompute_total` | 0 |
| `helios_preemptions_swap_total` | 0 |
| `helios_prefix_cache_hit_ratio` | 0.0000 |
| `helios_running_seqs` | 0 |
| `helios_scheduler_steps_total` | 50 |
| `helios_spec_acceptance_rate` | 0.0000 |
| `helios_tokens_decode_total` | 544 |
| `helios_tokens_prefill_total` | 1335 |
| `helios_waiting_seqs` | 0 |

## Workload

| parameter | value |
|---|---|
| `arrival_rate` | 0.0 |
| `max_prompt_len` | 256 |
| `num_requests` | 24 |
| `output_len_mean` | 20 |
| `output_len_sigma` | 0.4 |
| `prompt_len_mean` | 48 |
| `prompt_len_sigma` | 0.6 |
| `seed` | 0 |
| `shared_prefix_len` | 32 |
| `slo_mix` | [0.2, 0.6, 0.2] |
| `vocab_size` | 256 |

## Source artifacts

- `artifacts/baseline_hf_loop_seed0.json` -> `baseline_hf_loop`
- `artifacts/baseline_static_batch_8_seed0.json` -> `baseline_static_batch_8`
- `artifacts/baseline_unbatched_executor_seed0.json` -> `baseline_unbatched_executor`
- `artifacts/helios_batch1_seed0.json` -> `helios_batch1`
- `artifacts/helios_full_seed0.json` -> `helios_full`
- `artifacts/helios_no_chunked_prefill_seed0.json` -> `helios_no_chunked_prefill`
- `artifacts/helios_no_prefix_cache_seed0.json` -> `helios_no_prefix_cache`
- `artifacts/helios_small_kv_pool_seed0.json` -> `helios_small_kv_pool`
- `artifacts/helios_with_spec_decode_seed0.json` -> `helios_with_spec_decode`

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

- commit: `7f08c89c7a3b79c96a3344dd4763472b3f3dc5e9`
- timestamp: 2026-07-25T13:37:10
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
| `batch1` | 24/24 | 161.5 | 1.8838 | 3.3041 | 0.0064 | 0.0077 | 2.2301 | 0.38 |
| `full` | 24/24 | 409.4 | 0.3600 | 0.3601 | 0.0341 | 0.0426 | 1.0600 | 0.88 |
| `kv_fp_cramped` | 16/16 | 73.8 | 2.2014 | 4.6475 | 0.0613 | 0.1209 | 4.4743 | 0.25 |
| `kv_int8_cramped` | 16/16 | 54.0 | 1.3560 | 1.3561 | 0.2192 | 0.2564 | 6.3502 | 0.25 |
| `no_chunked_prefill` | 24/24 | 439.5 | 0.3160 | 0.3161 | 0.0313 | 0.0359 | 0.9608 | 0.88 |
| `no_prefix_cache` | 24/24 | 443.3 | 0.3163 | 0.3164 | 0.0292 | 0.0320 | 0.9293 | 0.88 |
| `prefill_heavy` | 32/32 | 90.2 | 0.9561 | 1.1549 | 0.1536 | 0.3427 | 1.2636 | 0.28 |
| `shared_prefix_cache_off` | 24/24 | 146.3 | 0.8277 | 0.8278 | 0.0864 | 0.1454 | 1.2906 | 0.17 |
| `shared_prefix_cache_on` | 24/24 | 218.6 | 0.5360 | 0.5361 | 0.0554 | 0.0777 | 0.8695 | 0.38 |
| `small_kv_pool` | 24/24 | 460.2 | 0.2940 | 0.2941 | 0.0300 | 0.0357 | 0.9097 | 0.88 |
| `with_spec_decode` | 24/24 | 331.1 | 0.3199 | 0.3200 | 0.0356 | 0.0403 | 0.9570 | 0.88 |

## Baselines

`baseline_hf_loop` re-runs the whole sequence per token with no cache reuse -- the 'no engine at all' reference. `baseline_static_batch_N` drains each batch fully before admitting the next, which is the head-of-line waste continuous batching removes.

> The naive loop runs a deliberately smaller workload (it is quadratic in sequence length), so its throughput is **not** directly comparable to the rows above. Compare mechanisms, not absolute numbers across different workloads.

| baseline | reqs | out tok/s | TTFT p50 | TPOT p50 | e2e p50 | workload |
|---|---|---|---|---|---|---|
| `baseline_hf_loop` | 8/8 | 71.6 | 0.0132 | 0.0131 | 0.1019 | n=8 plen~32 olen~8 |
| `baseline_prefill_heavy_unbatched` | 32/32 | 78.3 | 1.1093 | 0.1826 | 1.4747 | n=32 plen~128 olen~4 |
| `baseline_static_batch_8` | 24/24 | 361.2 | 0.0716 | 0.0139 | 0.3469 | n=24 plen~48 olen~20 |
| `baseline_unbatched_executor` | 24/24 | 183.1 | 0.3318 | 0.1066 | 2.5144 | n=24 plen~48 olen~20 |
| `baseline_unbatched_prefill` | 24/24 | 381.7 | 0.3833 | 0.0377 | 1.1745 | n=24 plen~48 olen~20 |

## Interpretation

**Batched decode is the dominant win: 2.24x** (409.4 vs 183.1 out tok/s). `baseline_unbatched_executor` is the *same* engine -- same scheduler, same paging, same prefix cache -- with only the executor's decode batching ablated, so the delta is attributable to that one mechanism. Every resident sequence's projections and MLP collapse into single GEMMs; attention stays per-sequence over its own paged KV.

- vs `baseline_hf_loop` (no engine at all, no KV reuse): 5.7x, though on a smaller workload -- see the caveat below.
- `batch1` reaches TPOT p50 of 0.0064s versus `full`'s 0.0341s, but TTFT p50 of 1.8838s versus 0.3600s. One sequence at a time gets the whole machine, so its own tokens come fast while everyone else queues. This is the batching trade-off, measured.
- `with_spec_decode` is **slower** (331.1 vs 409.4 out tok/s), exactly as expected: draft and verify share the same fp32 weights here, so drafting is pure serial overhead with no cheaper draft path. See `docs/SCOPE.md` -- this measures the bookkeeping, not QASSD.
- `no_prefix_cache` differs from `full` by -33.9 out tok/s. The prefix cache saves *prefill work* on repeated prompts; with a short shared prefix and a toy model, that saving is small relative to total cost. `tests/scheduler` asserts the mechanism works by counting prefill tokens directly, which is a sounder check than a wall-clock delta this size.
- vs `baseline_static_batch_8`: **1.13x** (409.4 vs 361.2 out tok/s). Continuous batching wins by reusing a finished sequence's slot in the same iteration instead of idling until the batch's longest member completes. The margin is modest because these output lengths are only moderately skewed -- the wider the spread, the more head-of-line waste static batching suffers.
  Static batching's low TTFT is an artifact of enqueueing every request at t=0: with all work available immediately, no request waits for a later batch to start.
- **Prefix cache**, measured on a long shared prefix (the regime it exists for): **1.49x** throughput and TTFT p50 1.54x better, having skipped 1408 prefill tokens (35% of the prompt work). In the mixed ablation table above, with only a 32-token shared prefix, the cache is a wash or slightly negative -- its bookkeeping costs more than a short prefix saves. Both readings are real; which one is 'the' result depends entirely on the workload, so both are reported.
- **Prefill batching**, measured on its own regime (long prompts, short generations -- where prefill is ~55x the token work of decode): TTFT p50 0.956s vs 1.109s unbatched, **1.16x** better, and 1.15x total token throughput. Smaller than the decode win because a prefill chunk is already a large GEMM on its own -- batching adds arithmetic density where decode batching creates it from nothing. Reported separately because one workload cannot exercise both mechanisms.
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
- `artifacts/baseline_prefill_heavy_unbatched_seed7.json` -> `baseline_prefill_heavy_unbatched`
- `artifacts/baseline_static_batch_8_seed0.json` -> `baseline_static_batch_8`
- `artifacts/baseline_unbatched_executor_seed0.json` -> `baseline_unbatched_executor`
- `artifacts/baseline_unbatched_prefill_seed0.json` -> `baseline_unbatched_prefill`
- `artifacts/helios_batch1_seed0.json` -> `helios_batch1`
- `artifacts/helios_full_seed0.json` -> `helios_full`
- `artifacts/helios_kv_fp_cramped_seed0.json` -> `helios_kv_fp_cramped`
- `artifacts/helios_kv_int8_cramped_seed0.json` -> `helios_kv_int8_cramped`
- `artifacts/helios_no_chunked_prefill_seed0.json` -> `helios_no_chunked_prefill`
- `artifacts/helios_no_prefix_cache_seed0.json` -> `helios_no_prefix_cache`
- `artifacts/helios_prefill_heavy_seed7.json` -> `helios_prefill_heavy`
- `artifacts/helios_shared_prefix_cache_off_seed11.json` -> `helios_shared_prefix_cache_off`
- `artifacts/helios_shared_prefix_cache_on_seed11.json` -> `helios_shared_prefix_cache_on`
- `artifacts/helios_small_kv_pool_seed0.json` -> `helios_small_kv_pool`
- `artifacts/helios_with_spec_decode_seed0.json` -> `helios_with_spec_decode`

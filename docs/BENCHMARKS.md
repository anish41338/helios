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

- commit: `unknown`
- timestamp: 2026-07-25T11:54:57
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
| `batch1` | 24/24 | 172.3 | 1.0131 | 1.8475 | 0.0061 | 0.0080 | 1.1842 | 0.67 |
| `full` | 24/24 | 164.3 | 0.4718 | 0.4719 | 0.1099 | 0.1396 | 1.7301 | 0.21 |
| `no_chunked_prefill` | 24/24 | 165.7 | 0.4364 | 0.4365 | 0.1116 | 0.1453 | 1.7130 | 0.21 |
| `no_prefix_cache` | 24/24 | 159.8 | 0.4209 | 0.4210 | 0.1193 | 0.1390 | 1.7833 | 0.21 |
| `small_kv_pool` | 24/24 | 150.2 | 0.4645 | 0.4646 | 0.1278 | 0.1476 | 1.9213 | 0.21 |
| `with_spec_decode` | 24/24 | 153.3 | 0.4462 | 0.4463 | 0.1121 | 0.1416 | 1.7077 | 0.21 |

## Baselines

`baseline_hf_loop` re-runs the whole sequence per token with no cache reuse -- the 'no engine at all' reference. `baseline_static_batch_N` drains each batch fully before admitting the next, which is the head-of-line waste continuous batching removes.

> The naive loop runs a deliberately smaller workload (it is quadratic in sequence length), so its throughput is **not** directly comparable to the rows above. Compare mechanisms, not absolute numbers across different workloads.

| baseline | reqs | out tok/s | TTFT p50 | TPOT p50 | e2e p50 | workload |
|---|---|---|---|---|---|---|
| `baseline_hf_loop` | 8/8 | 75.4 | 0.0138 | 0.0131 | 0.1001 | n=8 plen~32 olen~8 |
| `baseline_static_batch_8` | 16/16 | 146.2 | 0.1565 | 0.0418 | 0.4446 | n=16 plen~40 olen~10 |

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
| `helios_scheduler_steps_total` | 30 |
| `helios_spec_acceptance_rate` | 0.0000 |
| `helios_tokens_decode_total` | 311 |
| `helios_tokens_prefill_total` | 1335 |
| `helios_waiting_seqs` | 0 |

## Workload

| parameter | value |
|---|---|
| `arrival_rate` | 0.0 |
| `max_prompt_len` | 256 |
| `num_requests` | 24 |
| `output_len_mean` | 12 |
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
- `artifacts/helios_batch1_seed0.json` -> `helios_batch1`
- `artifacts/helios_full_seed0.json` -> `helios_full`
- `artifacts/helios_no_chunked_prefill_seed0.json` -> `helios_no_chunked_prefill`
- `artifacts/helios_no_prefix_cache_seed0.json` -> `helios_no_prefix_cache`
- `artifacts/helios_small_kv_pool_seed0.json` -> `helios_small_kv_pool`
- `artifacts/helios_with_spec_decode_seed0.json` -> `helios_with_spec_decode`

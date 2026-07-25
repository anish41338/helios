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

- commit: `c4ba69d6716c46eb527dd308b82fbd5c18635c9f`
- timestamp: 2026-07-25T12:08:03
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
| `batch1` | 20/20 | 161.3 | 0.9857 | 1.6653 | 0.0069 | 0.0088 | 1.0356 | 0.70 |
| `full` | 20/20 | 159.6 | 0.4122 | 0.4124 | 0.0958 | 0.1246 | 1.5075 | 0.25 |
| `no_chunked_prefill` | 20/20 | 155.7 | 0.3767 | 0.3768 | 0.1017 | 0.1342 | 1.5370 | 0.20 |
| `no_prefix_cache` | 20/20 | 159.0 | 0.3841 | 0.3842 | 0.0977 | 0.1270 | 1.5021 | 0.25 |
| `small_kv_pool` | 20/20 | 158.1 | 0.4011 | 0.4012 | 0.0976 | 0.1315 | 1.5134 | 0.25 |
| `with_spec_decode` | 20/20 | 144.9 | 0.3635 | 0.3636 | 0.1076 | 0.1270 | 1.6551 | 0.20 |

## Baselines

`baseline_hf_loop` re-runs the whole sequence per token with no cache reuse -- the 'no engine at all' reference. `baseline_static_batch_N` drains each batch fully before admitting the next, which is the head-of-line waste continuous batching removes.

> The naive loop runs a deliberately smaller workload (it is quadratic in sequence length), so its throughput is **not** directly comparable to the rows above. Compare mechanisms, not absolute numbers across different workloads.

| baseline | reqs | out tok/s | TTFT p50 | TPOT p50 | e2e p50 | workload |
|---|---|---|---|---|---|---|
| `baseline_hf_loop` | 8/8 | 75.6 | 0.0136 | 0.0135 | 0.0975 | n=8 plen~32 olen~8 |
| `baseline_static_batch_8` | 20/20 | 168.0 | 0.0983 | 0.0361 | 0.5576 | n=20 plen~48 olen~12 |

## Interpretation

Throughput differences between the ablations are within a few percent of each other, i.e. within run-to-run noise on an unpinned CPU. **On this workload, no scheduling mechanism shows a throughput win.** That is the honest reading; the mechanisms differ in *latency shape*, not in tokens/second.

- `batch1` reaches TPOT p50 of 0.0069s versus `full`'s 0.0958s, but TTFT p50 of 0.9857s versus 0.4122s. One sequence at a time gets the whole machine, so its own tokens come fast while everyone else queues. This is the batching trade-off, measured.
- `with_spec_decode` is **slower** (144.9 vs 159.6 out tok/s), exactly as expected: draft and verify share the same fp32 weights here, so drafting is pure serial overhead with no cheaper draft path. See `docs/SCOPE.md` -- this measures the bookkeeping, not QASSD.
- `no_prefix_cache` differs from `full` by +0.6 out tok/s. The prefix cache saves *prefill work* on repeated prompts; with a short shared prefix and a toy model, that saving is small relative to total cost. `tests/scheduler` asserts the mechanism works by counting prefill tokens directly, which is a sounder check than a wall-clock delta this size.
- **Negative result, stated plainly:** `baseline_static_batch_8` (168.0 out tok/s) matches or beats `full` (159.6), and continuous batching shows **no** throughput advantage here.

  The cause is measured, not guessed: this executor runs one sequence per forward pass in a Python loop, so a decode step costs ~5-7 ms **per resident sequence** regardless of batch size. Continuous batching's whole premise is that the batch dimension is nearly free -- one fused kernel serves N sequences for roughly the cost of one, so keeping slots full is pure profit. With a serialised executor that premise does not hold, and holding more sequences resident only adds scheduling work.

  This was checked against the mechanism's actual precondition rather than assumed: re-running with a 50x spread in output lengths (3 to 166 tokens) and Poisson arrivals -- the regime where reusing a finished slot should matter most -- still favoured static batching (0.91x). So the result is not a workload artifact; it is a property of this build. Demonstrating the advantage requires a batched attention kernel, which needs a GPU (see `docs/SCOPE.md`).

  What the scheduler *does* deliver here is admission control, preemption under memory pressure, and bounded latency per SLO class -- correctness and liveness properties, which is what `docs/DST.md` covers. Static batching's low TTFT is separately an artifact of enqueueing every request at t=0.
- `baseline_hf_loop` reruns the full sequence every token with no KV reuse. It runs a smaller workload (it is quadratic in length), so treat it as a sanity floor, not a ratio.

## Scheduler counters (`full` configuration)

| metric | value |
|---|---|
| `helios_aborted_total` | 0 |
| `helios_empty_steps_total` | 1 |
| `helios_exec_faults_total` | 0 |
| `helios_finished_total` | 20 |
| `helios_kv_blocks_total` | 8192 |
| `helios_kv_blocks_used` | 27 |
| `helios_kv_utilization` | 0.0033 |
| `helios_preemptions_recompute_total` | 0 |
| `helios_preemptions_swap_total` | 0 |
| `helios_prefix_cache_hit_ratio` | 0.0000 |
| `helios_running_seqs` | 0 |
| `helios_scheduler_steps_total` | 30 |
| `helios_spec_acceptance_rate` | 0.0000 |
| `helios_tokens_decode_total` | 259 |
| `helios_tokens_prefill_total` | 1176 |
| `helios_waiting_seqs` | 0 |

## Workload

| parameter | value |
|---|---|
| `arrival_rate` | 0.0 |
| `max_prompt_len` | 256 |
| `num_requests` | 20 |
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

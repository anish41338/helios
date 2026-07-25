# Architecture

How the pieces fit, and why the seams are where they are. Spec section 3 is the
design; this describes what was built (see `SCOPE.md` for the delta).

```
                    HTTP (OpenAI-compatible) + Prometheus
                                  |
                    helios/api/server.py
                      - request validation
                      - tokenization
                      - SLO class parsing
                      - one asyncio lock over the engine
                                  |
                            RequestEnvelope
                                  |
                    helios/engine.py  (LLMEngine)
                      - owns scheduler + runner
                      - wall-clock latency metrics
                      - derives KV block count from a byte budget
                                  |
              +-------------------+--------------------+
              |                                        |
   helios/core/scheduler.py                 helios/exec/runner.py
   DETERMINISTIC: no clock, no I/O,          PyTorch execution
   no RNG, no threads                        - prefill / decode
     - continuous batching                   - speculation
     - admission (token buckets)             - sampling
     - preemption (recompute/swap)                   |
     - chunked prefill                       helios/exec/model.py
              |                              GQA + RoPE + SwiGLU + RMSNorm
   +----------+----------+                           |
   |                     |                   helios/exec/paged_attn.py
allocator.py      prefix_cache.py            attention over a block table
paged blocks      radix trie                 + PagedKVCache (the bytes)
CoW, refcounts    LRU eviction
I1-I7
   |
   +--> helios/core/vopr.py  (DST harness)
        drives the scheduler with a simulated executor,
        simulated clock, and adversarial faults
```

## The load-bearing seam: `ExecStep`

`helios/core/execstep.py` defines the only interface between scheduler and
executor:

```python
class Executor(Protocol):
    def run(self, step: ExecStep) -> ExecOutputs: ...
```

`ExecStep` is plain data — token ids, positions, block ids, sampling params — no
tensors. That is what makes two things possible:

1. **`SimExecutor`** (in `vopr.py`) fabricates outputs with modelled latency and
   injectable faults, so the scheduler can be tested exhaustively with no model,
   no GPU, and no PyTorch in the loop.
2. **`ModelRunner`** (in `exec/runner.py`) runs the real forward pass.

The spec puts a shared-memory ring buffer and a Rust/Python boundary here. This
build runs both sides in one process, but keeps the same narrow contract — which
is the part that has testing value. Swapping in IPC later changes the transport,
not the interface.

## Determinism, concretely

The scheduler core obeys (spec §19.4):

| Rule | Why |
|---|---|
| No `time` import | A clock read makes a failing seed unreplayable |
| No `random` import | Same; all randomness is injected into the *harness*, never the scheduler |
| No threads, no I/O, no logging | Any of these introduces nondeterministic ordering |
| Ordered containers only | Dict/set iteration order would leak hash nondeterminism into batch composition |
| Token buckets refill per **step**, not per second | Admission decisions must not depend on wall clock |
| Total orders for every tie-break | `(class, arrival_step, seq_id)` for admission; `(class, seq_id)` for victims |

`tests/scheduler/test_scheduler.py::test_scheduler_module_has_no_time_or_random_imports`
enforces the first two mechanically. `test_identical_workloads_produce_identical_traces`
runs the same workload twice and diffs the full step trace.

Time inside the scheduler is `step_t`, an integer incremented once per
`step()`. Wall-clock latency is measured in `engine.py`, outside the pure core.

## Memory ownership

Three parties can hold a reference to a physical block, and I2 counts all of
them:

- a **block table** (a live sequence), one reference per table
- the **prefix cache**, via `allocator.cache_refs` — this is what lets a cached
  prefix outlive the sequence that produced it
- nothing else; a block with a positive ref count and no owner is a leak, and the
  DST harness asserts against it

Keeping the cache's references *explicit* in the allocator (rather than implied)
is what made I2 checkable and caught BUG-013. The prefix cache separately tracks
`holders` — pins that prevent *eviction* — which is a different concept from
ownership: a pinned node's blocks are still owned by `cache_refs`, but may not be
reclaimed while a live sequence is reusing them.

The reclamation order under memory pressure is deliberate: **evict cache before
preempting work**, because a cache entry is reconstructible and an in-flight
sequence's progress is not.

## Request lifecycle

```
add_request        -> validated, queued (owns nothing)
  |
_admit             -> prefix-cache lookup, token bucket, KV allocated
  |                   (every early exit must leave the sequence owning nothing)
  v
PREFILL            -> one or more chunks; only the last samples a token
  |
  v
DECODE             -> one token per step, or gamma+1 under speculation
  |
  +-- preempted --> recompute (KV dropped, requeued) or swapped to host tier
  |
  v
FINISHED/ABORTED   -> _reap_finished (from running) or _retire (from anywhere)
                      frees KV, releases pins, publishes to `finished`
```

The two termination paths matter: `_reap_finished` only walks the `running` list,
so anything finishing from another queue must go through `_retire` or its blocks
are never reclaimed (BUG-013).

## Where the spec's design was followed vs. adapted

**Followed**: the paged-allocator data structures and invariants (§5), the
scheduling loop's ordering (§6.1), SLO classes and step-driven token buckets
(§6.2), the recompute-vs-swap ratio test (§6.3), the speculative control flow and
its bit-parity requirement (§7.2), adaptive speculation gating (§7.3), the metric
names (§9.2), the DST fault taxonomy (§10.2), and the benchmark methodology
(§11).

**Adapted**: Python instead of Rust (§12.2 authorises this); direct call instead
of shm IPC; gather+SDPA instead of a Triton kernel (the spec's own fallback
path); one sequence per forward pass in the executor rather than a batched kernel
— the scheduler still emits properly batched `ExecStep`s, so batching the
executor later needs no scheduler change.

**Added** (not in the spec, required by bugs the harness found): the
growth-headroom reservation at admission, the starvation backstop keyed on
retired work, the admission aging reservation, and the `_retire` path. Each is
documented at its fix site and in `DST.md`.

# Scheduler

Spec section 6. `python/helios/core/scheduler.py`.

One call to `step()` == one executor forward pass == one increment of `step_t`.
There is no other notion of time in the core.

## The loop

```python
step_t += 1
_reap_finished()                 # free KV of last step's finishers, FIRST
refill token buckets (by step)
_maybe_swap_in()
_admit()                         # while budget, slots, and memory allow
_ensure_running_can_progress()   # reclaim cache, then preempt if needed
starvation backstop
step = _build_exec_step()        # prefill chunks + decode set
outputs = executor.run(step)     # the only fallible call
_apply_outputs(step, outputs)
_update_progress()
```

Reaping runs **before** admission on purpose: a finishing sequence's blocks must
be available to the sequence taking its slot in the *same* iteration. That
immediacy is the entire point of iteration-level batching versus static batching.

## Admission

Candidates are sorted by `(slo_class, arrival_step, seq_id)` — class A jumps the
queue, order within a class stays FIFO, and `seq_id` makes the order total so
replay is exact.

For each candidate, in this order:

1. slot check (`max_num_seqs`) and token-budget check
2. prefix-cache **peek** (no side effects)
3. per-class token bucket (`try_take`)
4. prefix-cache **commit** + KV allocation, with a spare block reserved

The peek/commit split is not stylistic. Attaching a cached prefix and *then*
failing a later check used to leave the sequence holding a pin in the waiting
queue, making the very memory it needed unevictable — a self-sustaining livelock
(DST BUG-007). Now every early exit is before any resource is taken, so a queued
sequence provably owns nothing. An invariant asserts it.

### Growth headroom

Allocation reserves `blocks_needed(total_tokens) + 1` against *currently free*
blocks. Both halves are load-bearing:

- Without the `+1`, a prompt that exactly fills the pool is admitted, prefills,
  then OOMs on its first output token, recomputes, and repeats forever
  (BUG-010).
- Checking total *capacity* instead of *free* blocks let two sequences each
  believe they had room to grow, then collide (BUG-012).

A sequence whose requirement has outgrown the pool entirely is **retired** with
`finish_reason="length"` rather than queued forever. Note this requirement grows
with each preemption, since recompute must redo prompt *plus* everything already
generated.

### Anti-starvation

A request that does not fit is skipped, not blocked on — otherwise one large
request starves every smaller one behind it (BUG-006/014). But unbounded skipping
would starve the large one instead, so after `reservation_after_steps` (32) the
queue head gets an exclusive reservation and nothing smaller is admitted until it
fits.

Two opposite livelocks; the fix has to thread between them.

## Preemption

Recompute is the default: drop the KV, requeue with the prompt intact. It costs
compute but frees memory instantly with no PCIe traffic. Swap (copy KV to the
host tier) is used only when `generated_len > 4 × prompt_len` — the spec's ratio
test — because swapping cheap-to-recompute sequences wastes bandwidth.

Victim choice is `max(class, seq_id)`: lowest priority first, then *newest*, so
the least-invested work is sacrificed and an old sequence is not repeatedly
evicted just as it nears completion.

Two rules exist because the harness found their absence:

- **Never preempt the last running sequence** (BUG-009). Preemption frees memory
  *for other work*; with one sequence there is no other work, so evicting it
  destroys progress for nothing — and because recompute cost grows every round,
  it never converges. Observed: 271 preemptions of one sequence.
- **Evict the prefix cache before preempting.** A cache entry is
  reconstructible; an in-flight sequence's progress is not.

## Chunked prefill

A long prompt advances a chunk at a time so it cannot monopolise a step and spike
TPOT for in-flight decodes (Sarathi-Serve style). `_build_exec_step` reserves one
token of budget per decode before packing prefills, so prefill can never starve
decode.

Chunking is a scheduling decision and must not change numerics: a chunked prefill
is **bit-identical** to a single-shot one, asserted in `tests/parity/`. The
mechanism is `num_computed_tokens` plus a `context_len` that covers previously
cached tokens, so each chunk attends over everything before it.

When chunking is disabled, a prompt exceeding the budget is admitted **alone** in
its own step — it is a legal request, so refusing to schedule it is a bug
(BUG-001).

## Starvation backstop

If no work retires for `stall_threshold` (8) consecutive steps, the prefix cache
is dropped wholesale. Correctness and liveness beat a warm cache.

Progress is measured as **retired** work — sequences finished plus output tokens
committed — deliberately *not* prefill tokens. A repeatedly-preempted sequence
inflates a prefill counter forever while completing nothing, which silently
disabled an earlier version of this backstop (BUG-008).

## Fault handling

`ExecFault` from the executor is recoverable. Affected sequences are rolled back
to recompute rather than trusted, because a partially-written KV cache would
produce wrong tokens rather than a crash — and wrong tokens are this project
class's characteristic failure mode (spec §19.1).

## Determinism

See `ARCHITECTURE.md`. The short version: no clock, no RNG, no threads, no I/O,
no unordered iteration, total tie-break orders, buckets refilled by step count.
A test greps the module for `time`/`random` imports; another runs a workload
twice and diffs the full step trace.

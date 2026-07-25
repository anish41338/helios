# Deterministic simulation testing

Spec section 10. Name (`vopr`) borrowed from TigerBeetle.

## The idea

The scheduler and allocator are a pure state machine: no clock, no I/O, no
threads, no unseeded randomness, no iteration over unordered containers. That
purity means the whole system can be driven single-threaded from a seed, with a
simulated executor and an adversarial fault schedule, asserting every invariant
after every step.

A failing seed is a complete reproduction:

```bash
python -m helios.cli vopr --seed 918273 --replay
```

## What each seed randomises

Everything is derived from the seed, so a run is fully determined by it:

- **Config**: block size, KV pool size (deliberately cramped — 24–200 blocks),
  token budget, `max_model_len`, watermark, `max_num_seqs`, whether chunked
  prefill / prefix cache / speculation / swap are enabled, class-A protection.
- **Workload**: 1–60 requests, Poisson arrivals with occasional 10× bursts,
  lognormal-ish lengths, SLO class mix, 40% of prompts sharing a canned prefix
  so the prefix cache is actually exercised.
- **Faults** (spec section 10.2): OOM, simulated CUDA error, timeout, transfer
  stall / partial copy / checksum mismatch, swap tier full — injected at random
  steps, affecting a deterministic subset of the batch.
- **Cancellations**: requests aborted at arbitrary later steps.
- **Adversarial acceptance**: speculation forced to 0% (all drafts wasted) or
  100% (maximum KV growth per step).

## What is asserted, every step

Allocator invariants I1–I7 from spec section 5.4, plus scheduler-level
structural invariants added as bugs demanded them:

- I1 a block is in exactly one free list, or referenced, never both
- I2 `ref_count` equals block-table references **plus** prefix-cache references
- I3 free + referenced == total, per tier
- I4 no holes in any block table
- I5 `filled_in_last ∈ [1, BLOCK_SIZE]` for a non-empty sequence
- I6 a shared block is never written in place (enforced by CoW; tested directly)
- I7 committed blocks stay under the watermark
- no sequence appears in two queues at once
- a **waiting** sequence owns no KV and no prefix-cache pin
- a **finished** sequence holds no blocks and no pin
- every sequence is reachable from some queue, or is terminal and reaped
- prefix-cache pins are attributable to a live sequence
- **liveness**: every submitted request reaches a terminal state
- **no leaks**: every block with a positive ref count has an owner

The liveness and leak checks matter as much as the invariants: most bugs below
were livelocks, not corruption.

## Result

```
50,000 seeds — 0 failures
```

Reproduce (about 25 minutes single-threaded):

```bash
python -m helios.cli vopr --seeds 50000 --progress-every 10000
```

Artifacts: `artifacts/dst_50k.log`, `artifacts/dst_50k.json`.

## Bugs found

All fifteen were found by the harness, not by hand, and none were seeded
deliberately. Each is fixed, each fix carries a regression test, and each is
named in a code comment at the fix site.

| # | Symptom | Root cause | Fix site |
|---|---|---|---|
| 001 | Deadlock, 0/44 requests finished | With chunked prefill disabled, a prompt longer than the per-step token budget was unschedulable — and because admission `break`ed on it, it blocked every smaller request behind it forever | `scheduler.py` `_admit`, `_build_exec_step` |
| 002 | Same deadlock, remaining case | A prompt within `max_model_len` but larger than the whole KV pool was accepted at the API and queued forever. `max_model_len` was never validated against actual KV capacity | `scheduler.py` `add_request`, `max_servable_tokens` |
| 003 | `I7: committed 63 > usable 62` | `blocks_needed_to_append` reported 0 blocks for a **shared, partially-filled tail** (it had spare room), but the write still had to copy-on-write, so allocation silently overshot the watermark. Exactly the bug class spec §5.3 predicted | `allocator.py` `blocks_needed_to_append` |
| 004 | Livelock, cached blocks never reclaimed | The prefix cache held blocks from finished sequences and nothing ever evicted them under memory pressure, so a request needing more than the free pool waited forever with the memory sitting right there | `scheduler.py` `_allocate_for`, `_reclaim_from_cache` |
| 005 | Livelock, eviction found no victims | Two faults: prefix-cache `acquire`/`release` were unbalanced (a preempted-and-re-prefilled sequence acquired twice, released once), leaving `holders > 0` forever; and only trie *leaves* were evictable, so a long shared-prefix chain kept most blocks in unevictable interior nodes | `prefix_cache.py` `evict`; `scheduler.py` `_pin_prefix` |
| 006 | Head-of-line livelock | Admission stopped at the first request that did not fit, starving smaller requests behind it when no running sequence remained to free anything | `scheduler.py` `_admit` |
| 007 | Cache nodes permanently pinned | A sequence that attached a cached prefix and then failed allocation went back to the waiting queue **still holding the pin**, making the very memory it needed unevictable — a self-sustaining livelock. Fixed structurally by splitting lookup (`_peek_prefix_cache`) from commit (`_attach_prefix_cache`) so every early exit owns nothing | `scheduler.py` `_admit` |
| 008 | Starvation with memory available | No backstop when the engine stopped making progress. Also the progress metric counted *prefill* tokens, which a repeatedly-preempted sequence inflates forever while completing nothing — so the backstop never fired. Now counts only retired work | `scheduler.py` `_update_progress`, `_drop_prefix_cache` |
| 009 | 271 preemptions of one sequence | Preempting the **last** running sequence destroys progress for no benefit — there is no other work to free memory for. Worse, recompute must redo prompt + all generated tokens, so the work per attempt grows every round | `scheduler.py` `_pick_victim` |
| 010 | 68 OOM-preempts, each redoing a 204-token prefill | Admission allocated for `total_tokens` but reserved nothing for tokens the sequence would generate. A prompt that exactly filled the pool could never emit its first token: allocate → OOM on append → recompute → repeat | `scheduler.py` `_allocate_for`, `can_ever_serve` |
| 011 | Most of the pool cached and unevictable | Publishing a prompt to the cache also **pinned** it for the publisher's lifetime, so a large cached prompt held memory hostage while queued sequences starved. The allocator's cache reference already keeps blocks alive; pins are only needed for reuse | `scheduler.py` `_flush_cache_inserts` |
| 012 | Two sequences recompute-preempting each other | The growth-headroom check tested total *capacity* rather than *currently free* blocks, so two sequences were each admitted believing they had room to grow, then collided on the first output token | `scheduler.py` `_allocate_for` |
| 013 | Entire KV pool leaked | A sequence finished outside the `running` list became an orphan: marked finished, in no queue, holding 32/32 blocks forever. `_reap_finished` only walks `running` | `scheduler.py` `_retire` |
| 014 | Starvation with 9 free blocks | Requests needing 3 blocks were refused while a 14-block request sat ahead of them in the queue and the running set had stabilised. Smaller requests may now overtake — with an aging reservation so the large one cannot itself be starved | `scheduler.py` `_admit` |
| 015 | Sequence admitted and re-prefilled every step forever | A sequence needing the whole pool skipped the growth guard once it already held a block table (mid-prefill), so it cycled indefinitely instead of being retired | `scheduler.py` `_allocate_for` |

### What the pattern says

Thirteen of fifteen were **liveness** bugs — livelocks and starvation — not
memory corruption. That is worth noting because ordinary integration testing
finds crashes and wrong answers, and would have found almost none of these: the
engine kept running, kept looking busy, and quietly completed nothing. Several
only appear when the KV pool is small enough to force preemption, which is why
the harness deliberately generates cramped configurations.

Two of the fifteen (003, 005) were caught by an invariant firing. The other
thirteen were caught by the liveness and leak assertions. Invariants alone would
not have been enough.

Several fixes were also **wrong at first** and the harness said so immediately:
the BUG-004 fix regressed the pass rate from 193/200 to 157/200 by exposing the
BUG-005 pin leak, and the BUG-011 fix surfaced BUG-013's orphan leak. Each
regression showed up within seconds as a *different, sharper* failure — which is
the actual argument for this style of testing.

### Invariants added because of a bug

Three assertions exist only because a bug taught us to check for them, and each
turns a 4000-step liveness timeout into an immediate, localized failure:

- waiting sequences own nothing (BUG-007)
- every sequence is reachable from a queue (BUG-013)
- cache pins are attributable to a live sequence (BUG-007/011)

## Limitations of this harness

Stated plainly, since the point of DST is credibility:

- The **executor is simulated**. It models latency and faults, not numerics. It
  cannot catch a wrong-logits bug; that is what `tests/parity/` is for.
- The KV **transfer FSM is not implemented** (no second device), so its fault
  kinds are injected but the transport itself is untested. See `docs/SCOPE.md`.
- Fault injection is at step granularity, not instruction granularity. A real
  VOPR would also interleave partial writes.
- 50,000 seeds is not 1,000,000 (the spec's target). The seed space is explored,
  not exhausted; the sweep is trivially extendable with `--seeds`.

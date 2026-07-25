"""Scheduler behaviour and determinism tests.

Covers spec section 6 (continuous batching, admission, preemption) and the
section 19.4 determinism contract. Regression tests name the DST bug they pin;
see docs/DST.md.
"""

from __future__ import annotations

import random

import pytest

from helios.core.allocator import Allocator
from helios.core.execstep import ExecFault, ExecOutputs, ExecStep, FaultKind, SeqOutput
from helios.core.prefix_cache import PrefixCache
from helios.core.scheduler import Scheduler, SchedulerConfig
from helios.core.types import (
    FinishReason,
    Request,
    SamplingParams,
    SeqState,
    SloClass,
)
from helios.core.vopr import Simulation, run_seed

EOS = 2


def make_scheduler(
    blocks: int = 128,
    block_size: int = 16,
    **overrides,
) -> Scheduler:
    cfg_kwargs = dict(
        max_num_seqs=16,
        max_num_batched_tokens=512,
        max_model_len=512,
        block_size=block_size,
    )
    cfg_kwargs.update(overrides)
    config = SchedulerConfig(**cfg_kwargs)
    allocator = Allocator(total_vram_blocks=blocks, block_size=block_size)
    cache = PrefixCache(block_size, enabled=config.enable_prefix_cache)
    return Scheduler(config, allocator, cache)


class ScriptedExecutor:
    """Emits a fixed number of tokens per sequence, then EOS.

    Deliberately simple: these tests exercise scheduling decisions, not model
    numerics.
    """

    def __init__(self, output_len: int = 4, fault_at: int | None = None):
        self.output_len = output_len
        self.fault_at = fault_at
        self.emitted: dict[int, int] = {}
        self.steps = 0
        self.seen_batch_sizes: list[int] = []

    def run(self, step: ExecStep) -> ExecOutputs:
        self.steps += 1
        self.seen_batch_sizes.append(step.batch_size)
        if self.fault_at is not None and step.step_id == self.fault_at:
            ids = [p.seq_id for p in step.prefills] + [d.seq_id for d in step.decodes]
            raise ExecFault(FaultKind.CUDA_ERROR, "scripted", ids)

        outputs = []
        for item in step.prefills:
            if item.is_last_chunk:
                outputs.append(SeqOutput(item.seq_id, [self._next(item.seq_id)]))
        for item in step.decodes:
            outputs.append(SeqOutput(item.seq_id, [self._next(item.seq_id)]))
        return ExecOutputs(step_id=step.step_id, outputs=outputs)

    def _next(self, seq_id: int) -> int:
        n = self.emitted.get(seq_id, 0)
        self.emitted[seq_id] = n + 1
        return EOS if n + 1 >= self.output_len else 1000 + n


def add(sched: Scheduler, prompt_len: int, max_tokens: int = 4, cls=SloClass.B, rid=None):
    return sched.add_request(
        Request(
            request_id=rid or f"r{prompt_len}-{max_tokens}",
            prompt_token_ids=list(range(100, 100 + prompt_len)),
            params=SamplingParams(max_tokens=max_tokens, stop_token_ids=[EOS]),
            slo_class=cls,
        )
    )


# ------------------------------------------------------------------- basics


def test_single_request_completes():
    s = make_scheduler()
    add(s, 10, max_tokens=4)
    ex = ScriptedExecutor(output_len=4)
    for _ in range(50):
        s.step(ex)
        s.check_invariants()
        if not s.has_work:
            break
    assert s.stats.finished_seqs == 1
    assert not s.has_work


def test_rejects_empty_and_oversized_prompts():
    s = make_scheduler(max_model_len=64)
    with pytest.raises(ValueError):
        s.add_request(Request("empty", [], SamplingParams()))
    with pytest.raises(ValueError):
        add(s, 200)


def test_rejects_prompt_exceeding_kv_capacity():
    """Regression: DST BUG-002.

    A prompt within max_model_len but larger than the KV pool can never be
    admitted, so it must be refused rather than queued forever.
    """
    s = make_scheduler(blocks=8, block_size=8, max_model_len=4096)
    assert s.max_servable_tokens < 4096
    with pytest.raises(ValueError, match="KV capacity"):
        add(s, 500)


def test_continuous_batching_admits_into_freed_slots():
    """A finishing sequence's slot is reused in the same iteration."""
    s = make_scheduler(max_num_seqs=2)
    for i in range(6):
        add(s, 8, max_tokens=2, rid=f"r{i}")
    ex = ScriptedExecutor(output_len=2)
    for _ in range(200):
        s.step(ex)
        s.check_invariants()
        assert s.num_running() <= 2, "max_num_seqs must be respected"
        if not s.has_work:
            break
    assert s.stats.finished_seqs == 6


def test_token_budget_caps_prefill_per_step():
    s = make_scheduler(max_num_batched_tokens=64, enable_chunked_prefill=True)
    add(s, 200, max_tokens=1)
    ex = ScriptedExecutor(output_len=1)

    step = s._build_exec_step() if False else None  # keep the private call explicit
    s.step(ex)
    s.check_invariants()
    # The prompt is longer than the budget, so prefill must be chunked.
    seq = s.get_sequence(0)
    assert 0 < seq.num_computed_tokens <= 64


def test_chunked_prefill_eventually_completes_long_prompt():
    s = make_scheduler(max_num_batched_tokens=32, enable_chunked_prefill=True)
    add(s, 300, max_tokens=2)
    ex = ScriptedExecutor(output_len=2)
    for _ in range(500):
        s.step(ex)
        s.check_invariants()
        if not s.has_work:
            break
    assert s.stats.finished_seqs == 1


def test_unchunkable_long_prompt_still_runs():
    """Regression: DST BUG-001.

    With chunked prefill disabled, a prompt longer than the per-step token
    budget must still be schedulable -- alone in its own step -- and must not
    block smaller requests queued behind it.
    """
    s = make_scheduler(
        max_num_batched_tokens=64, enable_chunked_prefill=False, max_model_len=512
    )
    add(s, 200, max_tokens=2, rid="big")
    add(s, 8, max_tokens=2, rid="small")
    ex = ScriptedExecutor(output_len=2)
    for _ in range(400):
        s.step(ex)
        s.check_invariants()
        if not s.has_work:
            break
    assert s.stats.finished_seqs == 2, "both the oversized and the small request run"


def test_class_a_is_prioritised_over_class_c():
    s = make_scheduler(max_num_seqs=1)
    add(s, 8, max_tokens=2, cls=SloClass.C, rid="c")
    add(s, 8, max_tokens=2, cls=SloClass.A, rid="a")
    ex = ScriptedExecutor(output_len=2)
    s.step(ex)
    # Class A jumps the queue despite arriving second.
    running_classes = [x.slo_class for x in s.running]
    assert SloClass.A in running_classes


def test_abort_is_total_and_idempotent():
    """DST cancels at every state, so abort must handle all of them."""
    s = make_scheduler()
    sid = add(s, 20, max_tokens=8)
    assert s.abort(sid) is True
    assert s.abort(sid) is False        # idempotent
    s.check_invariants()
    assert s.get_sequence(sid).state is SeqState.ABORTED
    # Aborting something that never existed is a no-op, not a crash.
    assert s.abort(9999) is False


def test_abort_while_waiting_releases_everything():
    s = make_scheduler(max_num_seqs=1)
    add(s, 8, max_tokens=8, rid="running")
    queued = add(s, 8, max_tokens=8, rid="queued")
    ex = ScriptedExecutor(output_len=8)
    s.step(ex)
    s.abort(queued)
    s.check_invariants()


def test_executor_fault_rolls_sequences_back():
    s = make_scheduler()
    add(s, 20, max_tokens=4)
    ex = ScriptedExecutor(output_len=4, fault_at=2)
    for _ in range(100):
        s.step(ex)
        s.check_invariants()
        if not s.has_work:
            break
    assert s.stats.exec_faults == 1
    assert s.stats.preemptions_recompute >= 1
    assert s.stats.finished_seqs == 1, "the request still completes after recovery"


def test_finished_sequences_release_their_blocks():
    s = make_scheduler()
    free_before = s.allocator.free_vram_count
    for i in range(5):
        add(s, 16, max_tokens=3, rid=f"r{i}")
    ex = ScriptedExecutor(output_len=3)
    for _ in range(200):
        s.step(ex)
        if not s.has_work:
            break
    s.take_finished()
    # Only prefix-cache retained blocks may remain committed.
    assert s.allocator.free_vram_count + len(s.allocator.cache_refs) == free_before


def test_prefix_cache_hits_on_shared_prompt():
    """A later request reuses an earlier one's published prefix blocks.

    The two requests are run sequentially on purpose: a prompt is published to
    the cache only when its prefill completes, so requests admitted in the same
    step cannot hit each other's prefixes. Concurrent submission is covered by
    test_prefix_cache_concurrent_submission_is_a_miss.
    """
    s = make_scheduler(enable_prefix_cache=True, block_size=16)
    shared = list(range(500, 564))       # 64 tokens == 4 whole blocks
    ex = ScriptedExecutor(output_len=2)

    def run(rid: str, last: int):
        s.add_request(
            Request(rid, shared + [last], SamplingParams(max_tokens=2, stop_token_ids=[EOS]))
        )
        for _ in range(50):
            s.step(ex)
            s.check_invariants()
            if not s.has_work:
                break

    run("p0", 900)
    assert s.prefix_cache.cached_blocks == 4, "whole blocks of p0's prompt are cached"

    prefill_before = s.stats.prefill_tokens
    run("p1", 901)
    assert s.stats.finished_seqs == 2
    assert s.prefix_cache.hits >= 1, "second request should reuse the shared prefix"
    # The hit must be real work avoided: p1's 65-token prompt should cost far
    # fewer prefill tokens than p0's did, because 64 of them were cached.
    # (cached_prefix_len is not checked here -- it is cleared when the sequence
    # is freed, so it reads 0 by the time the request has completed.)
    assert s.stats.prefill_tokens - prefill_before <= 8


def test_prefix_cache_concurrent_submission_is_a_miss():
    """Two identical prompts admitted together cannot reuse each other.

    Documents the boundary of the cache rather than asserting a hit: publication
    happens at prefill completion, so simultaneous arrivals both miss. Both must
    still complete correctly.
    """
    s = make_scheduler(enable_prefix_cache=True, block_size=16)
    shared = list(range(500, 564))
    for i in range(2):
        s.add_request(
            Request(
                f"p{i}", shared + [900 + i],
                SamplingParams(max_tokens=2, stop_token_ids=[EOS]),
            )
        )
    ex = ScriptedExecutor(output_len=2)
    for _ in range(100):
        s.step(ex)
        s.check_invariants()
        if not s.has_work:
            break
    assert s.stats.finished_seqs == 2


def test_waiting_sequences_hold_no_resources():
    """Regression: DST BUG-007.

    A queued sequence must own neither KV nor a prefix-cache pin; a pin left by
    a failed admission makes cache nodes permanently unevictable.
    """
    s = make_scheduler(blocks=12, block_size=8, enable_prefix_cache=True)
    for i in range(10):
        add(s, 40, max_tokens=4, rid=f"r{i}")
    ex = ScriptedExecutor(output_len=4)
    for _ in range(200):
        s.step(ex)
        s.check_invariants()      # includes the waiting-owns-nothing assertions
        if not s.has_work:
            break


def test_no_orphaned_finished_sequences():
    """Regression: DST BUG-013.

    A sequence finished outside the running list must still have its KV freed,
    or the pool leaks permanently.
    """
    s = make_scheduler(blocks=10, block_size=8, watermark=0.0, max_model_len=256)
    add(s, 60, max_tokens=200, rid="hog")
    ex = ScriptedExecutor(output_len=200)
    for _ in range(300):
        s.step(ex)
        s.check_invariants()      # includes the orphan check
        if not s.has_work:
            break


# ------------------------------------------------------------- determinism


def test_identical_workloads_produce_identical_traces():
    """Spec section 19.4: the scheduler must be exactly replayable."""

    def run() -> list:
        s = make_scheduler(blocks=48, block_size=8, max_num_seqs=4)
        for i in range(12):
            add(s, 20 + i * 3, max_tokens=3, cls=[SloClass.A, SloClass.B, SloClass.C][i % 3], rid=f"r{i}")
        ex = ScriptedExecutor(output_len=3)
        trace = []
        for _ in range(300):
            s.step(ex)
            trace.append(s.snapshot())
            if not s.has_work:
                break
        return trace

    assert run() == run()


def test_scheduler_module_has_no_time_or_random_imports():
    """The determinism contract, enforced mechanically.

    A clock read or unseeded RNG in the scheduler core makes DST replay
    meaningless, so this is a structural guard rather than a convention.
    """
    import pathlib

    src = pathlib.Path(
        Scheduler.__module__.replace(".", "/") + ".py"
    )
    root = pathlib.Path(__file__).resolve().parents[2] / "python"
    text = (root / src).read_text(encoding="utf-8")
    for banned in ("import time", "import random", "time.time", "datetime"):
        assert banned not in text, f"scheduler must not reference {banned!r}"


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 999])
def test_dst_seeds_pass(seed: int):
    """Spot-check the simulation harness from the normal test suite."""
    result = run_seed(seed)
    assert result.ok, f"seed {seed}: {result.violations}"


def test_dst_harness_exercises_copy_on_write():
    """The harness must actually reach the CoW path, not just claim to.

    Copy-on-write is where spec section 5.3 says the classic corruption bug
    lives, and where DST BUG-003 lived. Coverage measurement once showed 0/300
    seeds reaching it: the prefix cache shares only whole blocks, so nothing
    ever wrote into a shared *partially filled* tail. `_maybe_fork` exists to
    close that gap, and this test fails if it stops working.
    """
    total_copies = 0
    seeds_with_cow = 0
    for seed in range(12):
        sim = Simulation(seed, max_steps=600)
        sim.run()
        if sim.allocator.cow_copies:
            seeds_with_cow += 1
        total_copies += sim.allocator.cow_copies

    assert seeds_with_cow >= 10, (
        f"only {seeds_with_cow}/12 seeds exercised copy-on-write; the harness "
        f"is not covering the path BUG-003 lived in"
    )
    assert total_copies > 0


def test_dst_replay_is_deterministic():
    a = Simulation(31337, max_steps=600).run()
    b = Simulation(31337, max_steps=600).run()
    assert a.steps == b.steps
    assert a.finished == b.finished
    assert a.violations == b.violations

"""KV transfer FSM tests (spec section 6.4).

The DST harness sweeps this state machine under randomised faults and catches
accounting errors across thousands of seeds. These tests pin the specific edges
that a random walk reaches rarely or reaches only in combination, where a failure
would be hard to attribute:

  * every illegal transition raises, because the transition table IS the
    specification -- an FSM that tolerates an unlisted edge specifies nothing;
  * a failure returns blocks from BOTH sides, which is the bug class that leaks;
  * the sender holds its blocks until the receiver acknowledges, so a lost ack is
    recoverable rather than fatal;
  * retry is bounded and falls back to re-prefill, so a permanently broken link
    degrades instead of hanging.

The transfer-versus-reprefill arithmetic is tested in both directions on purpose:
the answer depends on the hardware, and a design that assumed one answer would be
wrong on half the plausible deployments.
"""

from __future__ import annotations

import pytest

from helios.core.disagg import (
    DisaggConfig,
    KVTransferManager,
    Role,
    TransferFault,
    TransferInvariantViolation,
    TransferState,
    transfer_vs_reprefill,
)


@pytest.fixture
def mgr():
    return KVTransferManager(
        DisaggConfig(
            link_bandwidth_bps=1e9, max_concurrent_transfers=2,
            transfer_timeout_steps=5, max_retries=2,
        )
    )


def _submit(mgr, seq_id=1, n=4, bpb=1024):
    return mgr.submit(seq_id, list(range(100, 100 + n)), bytes_per_block=bpb)


def _drive_to_sent(mgr, t, max_steps=200):
    mgr.reserve(t, list(range(n := t.n_blocks)) if False else list(range(t.n_blocks)))
    mgr.begin(t)
    for _ in range(max_steps):
        if t.state == TransferState.SENT:
            return
        mgr.step()
    raise AssertionError(f"transfer never reached SENT (stuck in {t.state.value})")


# ------------------------------------------------------------ the happy path


def test_full_lifecycle_reaches_committed(mgr):
    t = _submit(mgr)
    assert t.state == TransferState.PENDING
    _drive_to_sent(mgr, t)
    mgr.acknowledge(t)
    assert t.state == TransferState.RECEIVED
    src = mgr.commit(t)

    assert t.state == TransferState.COMMITTED
    assert src == t.src_blocks, "the sender's blocks must come back for reuse"
    assert mgr.completed == 1
    assert mgr.bytes_moved == t.bytes_total
    mgr.check_invariants()


def test_bytes_moved_matches_the_declared_size(mgr):
    t = _submit(mgr, n=8, bpb=4096)
    assert t.bytes_total == 8 * 4096
    _drive_to_sent(mgr, t)
    assert t.bytes_sent >= t.bytes_total


def test_bandwidth_is_shared_not_multiplied(mgr):
    """Two concurrent transfers take about twice as long as one.

    Modelling concurrency as extra throughput would make disaggregation look
    better than the interconnect allows, which is the easiest way to get this
    design decision wrong on paper.
    """
    def steps_for(n_transfers):
        m = KVTransferManager(
            DisaggConfig(link_bandwidth_bps=1e9, max_concurrent_transfers=4)
        )
        # Sized so the transfer takes many steps: at ~1 step per transfer the
        # integer step granularity dominates and the ratio measures rounding
        # rather than bandwidth sharing.
        ts = [_submit(m, seq_id=i, n=4, bpb=2**22) for i in range(n_transfers)]
        for i, t in enumerate(ts):
            m.reserve(t, list(range(i * 4, i * 4 + 4)))
            m.begin(t)
        for step in range(1, 10_000):
            m.step()
            if all(t.state == TransferState.SENT for t in ts):
                return step
        raise AssertionError("never finished")

    one, two = steps_for(1), steps_for(2)
    assert 1.7 <= two / one <= 2.3, f"1 transfer: {one} steps, 2: {two}"


# ------------------------------------------------- the transition table itself


@pytest.mark.parametrize(
    "target",
    [TransferState.SENDING, TransferState.SENT, TransferState.RECEIVED,
     TransferState.COMMITTED],
)
def test_cannot_skip_ahead_from_pending(mgr, target):
    t = _submit(mgr)
    with pytest.raises(TransferInvariantViolation, match="illegal transfer transition"):
        mgr._transition(t, target)


def test_cannot_commit_without_an_acknowledgement(mgr):
    """Committing at SENT would release the only copy of the KV on a lost ack."""
    t = _submit(mgr)
    _drive_to_sent(mgr, t)
    with pytest.raises(TransferInvariantViolation):
        mgr.commit(t)


def test_terminal_states_are_terminal(mgr):
    t = _submit(mgr)
    mgr.abort(t)
    assert t.is_terminal
    with pytest.raises(TransferInvariantViolation):
        mgr._transition(t, TransferState.RESERVED)


def test_reserve_rejects_a_wrong_sized_destination(mgr):
    t = _submit(mgr, n=4)
    with pytest.raises(TransferInvariantViolation, match="destination blocks"):
        mgr.reserve(t, [0, 1])


def test_link_saturation_is_enforced(mgr):
    """max_concurrent_transfers is a hard limit, not a hint."""
    ts = [_submit(mgr, seq_id=i, n=2) for i in range(3)]
    for i, t in enumerate(ts[:2]):
        mgr.reserve(t, [i * 2, i * 2 + 1])
        mgr.begin(t)
    assert not mgr.can_begin()
    mgr.reserve(ts[2], [8, 9])
    with pytest.raises(TransferInvariantViolation, match="saturated"):
        mgr.begin(ts[2])


# ------------------------------------------------------------ partial failure


def test_failure_returns_blocks_from_both_sides(mgr):
    """The bug class this FSM exists to prevent.

    A failed transfer has blocks pinned on the sender AND reserved on the
    receiver. Handling one side leaks the other, and a leak is invisible until the
    pool is exhausted -- which looks like a capacity problem, not a bug.
    """
    t = _submit(mgr, n=4)
    mgr.reserve(t, [0, 1, 2, 3])
    mgr.begin(t)
    src, dst = mgr.fail(t, TransferFault.CHECKSUM_MISMATCH)

    assert src == t.src_blocks
    assert dst == [0, 1, 2, 3]
    assert t.fault == TransferFault.CHECKSUM_MISMATCH
    assert t.state == TransferState.FAILED


def test_sender_holds_blocks_until_the_receiver_confirms(mgr):
    """A lost ack must be recoverable, which requires the source to still exist."""
    t = _submit(mgr)
    _drive_to_sent(mgr, t)
    assert t.state == TransferState.SENT
    assert t.sender_holds_blocks, "releasing here makes a lost ack unrecoverable"
    assert t.receiver_holds_blocks

    mgr.acknowledge(t)
    assert t.sender_holds_blocks, "still not safe to release before commit"
    mgr.commit(t)
    assert not t.sender_holds_blocks


def test_a_lost_ack_is_resolved_by_the_timeout(mgr):
    """The only mechanism that can resolve a transfer stuck in SENT.

    Without it the transfer pins blocks on both partitions forever -- a hang, not
    a crash, and therefore the kind that reaches production.
    """
    t = _submit(mgr)
    _drive_to_sent(mgr, t)
    assert mgr.timed_out() == []
    for _ in range(mgr.config.transfer_timeout_steps + 1):
        mgr.step()
    assert t in mgr.timed_out()

    src, dst = mgr.fail(t, TransferFault.LINK_TIMEOUT)
    assert src and dst


def test_abort_after_send_becomes_a_failure_not_an_abort(mgr):
    """Ownership has to be resolved before anything is released.

    A plain abort at SENT would leave the receiver holding reserved blocks that
    nothing will ever claim -- the sender thinks it is done, and no acknowledgement
    is coming.
    """
    t = _submit(mgr)
    _drive_to_sent(mgr, t)
    src, dst = mgr.abort(t)
    assert t.state == TransferState.FAILED
    assert t.fault == TransferFault.ABORT
    assert dst, "the receiver's reserved blocks must be reclaimed"


def test_abort_before_send_is_a_clean_abort(mgr):
    t = _submit(mgr)
    mgr.abort(t)
    assert t.state == TransferState.ABORTED
    assert mgr.aborted == 1


# -------------------------------------------------------------------- retries


def test_retry_resets_state_and_is_bounded(mgr):
    t = _submit(mgr, n=2)
    for attempt in range(mgr.config.max_retries):
        mgr.reserve(t, [0, 1])
        mgr.begin(t)
        mgr.fail(t, TransferFault.LINK_TIMEOUT)
        assert mgr.retry(t) is True
        assert t.state == TransferState.PENDING
        assert t.dst_blocks == [] and t.bytes_sent == 0 and t.started_step is None
        assert t.retries == attempt + 1

    mgr.reserve(t, [0, 1])
    mgr.begin(t)
    mgr.fail(t, TransferFault.LINK_TIMEOUT)
    assert mgr.retry(t) is False, "retries must be bounded"
    assert mgr.fell_back_to_reprefill == 1


def test_reprefill_is_always_available_as_a_fallback():
    """A permanently broken link degrades to recompute rather than hanging.

    Same argument as recompute-preemption (spec section 6.3): the prompt tokens
    are still known, so the KV can always be rebuilt from scratch.
    """
    m = KVTransferManager(DisaggConfig(max_retries=0))
    t = _submit(m, n=1)
    m.reserve(t, [0])
    m.begin(t)
    m.fail(t, TransferFault.RECEIVER_OOM)
    assert m.retry(t) is False
    assert m.fell_back_to_reprefill == 1


# ------------------------------------------------------------------ hygiene


def test_duplicate_destination_blocks_are_caught(mgr):
    """Two live transfers writing the same destination would interleave KV."""
    a, b = _submit(mgr, seq_id=1, n=2), _submit(mgr, seq_id=2, n=2)
    mgr.reserve(a, [0, 1])
    mgr.reserve(b, [1, 2])          # block 1 double-booked
    with pytest.raises(TransferInvariantViolation, match="D3"):
        mgr.check_invariants()


def test_gc_removes_only_terminal_transfers(mgr):
    done, live = _submit(mgr, seq_id=1), _submit(mgr, seq_id=2)
    mgr.abort(done)
    assert mgr.gc() == 1
    assert live.transfer_id in mgr.transfers
    assert done.transfer_id not in mgr.transfers


def test_snapshot_is_serialisable(mgr):
    import json

    t = _submit(mgr)
    _drive_to_sent(mgr, t)
    mgr.acknowledge(t)
    mgr.commit(t)
    json.dumps(mgr.snapshot())      # must not raise


def test_roles_are_distinct():
    assert Role.PREFILL != Role.DECODE
    assert Role.BOTH.value == "both"


# ------------------------------------------- is disaggregation worth it here?


# The reference workload for all three: 2000 prompt tokens, 32 layers, 8 KV
# heads, head_dim 128, fp16 -> 262 MB of KV.
_REF = dict(n_prompt_tokens=2000, n_layers=32, n_kv_heads=8, head_dim=128,
            dtype_bytes=2)


def test_kv_size_is_262_mb_for_the_reference_workload():
    """Pin the arithmetic itself.

    Exists because the module docstring originally had this number 10x too large,
    which flipped the PCIe verdict. A stated size deserves a test, not a comment.
    """
    r = transfer_vs_reprefill(**_REF, link_bandwidth_bps=1e9,
                              prefill_tokens_per_s=1000)
    assert r["kv_bytes"] == 2 * 2000 * 32 * 8 * 128 * 2
    assert 260e6 < r["kv_bytes"] < 264e6


def test_transfer_beats_reprefill_inside_a_node():
    """PCIe 3.0 at 12 GB/s: 22 ms to move it, 250 ms to recompute it."""
    r = transfer_vs_reprefill(**_REF, link_bandwidth_bps=12e9,
                              prefill_tokens_per_s=8000)
    assert r["transfer_is_cheaper"]
    assert r["ratio"] < 0.2, "should win by roughly 10x on PCIe"


def test_reprefill_beats_transfer_across_a_commodity_network():
    """10 GbE at 1.25 GB/s with a fast GPU: recomputing wins outright.

    The reason `transfer_vs_reprefill` exists rather than a hardcoded policy. A
    design that always transferred would be strictly worse than no
    disaggregation at all on this hardware -- and cross-node disaggregation over
    Ethernet is a configuration people really do reach for.
    """
    r = transfer_vs_reprefill(**_REF, link_bandwidth_bps=1.25e9,
                              prefill_tokens_per_s=20000)
    assert not r["transfer_is_cheaper"]
    assert r["ratio"] > 2.0


def test_kv_size_formula_matches_the_allocator():
    """One sizing formula, not two that can drift apart."""
    from helios.core.allocator import Allocator

    tokens, layers, heads, dim, dtype_bytes = 512, 4, 8, 64, 2
    r = transfer_vs_reprefill(
        tokens, layers, heads, dim, dtype_bytes,
        link_bandwidth_bps=1e9, prefill_tokens_per_s=1000,
    )
    per_block = Allocator.bytes_per_block(
        block_size=16, n_kv_heads=heads, head_dim=dim, n_layers=layers,
        dtype_bytes=dtype_bytes,
    )
    assert r["kv_bytes"] == per_block * (tokens // 16)

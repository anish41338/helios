"""Allocator property tests.

Spec section 13.2: random alloc/free/fork/CoW sequences, asserting I1..I7 after
every operation. Uses a seeded random walk rather than `hypothesis` (not
installed); the seed makes every failure reproducible, which is the property
that actually matters here.
"""

from __future__ import annotations

import random

import pytest

from helios.core.allocator import (
    Allocator,
    AllocError,
    InvariantViolation,
    Tier,
)


def test_sizing_formula_matches_spec_worked_example():
    """Spec section 5.2: Llama-3.1-8B, BLOCK_SIZE=16, 8 KV heads, 128 dim, 32 layers."""
    per_block = Allocator.bytes_per_block(
        block_size=16, n_kv_heads=8, head_dim=128, n_layers=32, dtype_bytes=2
    )
    assert per_block == 2_097_152      # 2 MiB/block, exactly as the spec states


def test_blocks_needed_rounds_up():
    a = Allocator(total_vram_blocks=100, block_size=16)
    assert a.blocks_needed_for(0) == 0
    assert a.blocks_needed_for(1) == 1
    assert a.blocks_needed_for(16) == 1
    assert a.blocks_needed_for(17) == 2


def test_allocate_and_free_conserves_pool():
    a = Allocator(total_vram_blocks=64, block_size=16)
    before = a.free_vram_count
    a.allocate(1, 100)
    a.check_invariants()
    assert a.free_vram_count < before
    a.free(1)
    a.check_invariants()
    assert a.free_vram_count == before


def test_filled_in_last_tracks_partial_block():
    a = Allocator(total_vram_blocks=64, block_size=16)
    t = a.allocate(1, 20)          # 2 blocks: 16 + 4
    assert len(t.blocks) == 2
    assert t.filled_in_last == 4
    assert t.num_tokens(16) == 20
    a.check_invariants()


def test_append_fills_tail_before_taking_a_new_block():
    a = Allocator(total_vram_blocks=64, block_size=16)
    a.allocate(1, 10)
    used = a.committed_vram_blocks
    a.append_tokens(1, 6)          # exactly fills the tail block
    assert a.committed_vram_blocks == used
    a.append_tokens(1, 1)          # now needs a new one
    assert a.committed_vram_blocks == used + 1
    a.check_invariants()


def test_allocate_is_atomic_on_failure():
    """A failed allocation must not leak partially-taken blocks (I3)."""
    a = Allocator(total_vram_blocks=10, block_size=16, watermark=0.0)
    free_before = a.free_vram_count
    with pytest.raises(AllocError):
        a.allocate(1, 16 * 50)     # far more than the pool
    assert a.free_vram_count == free_before
    assert 1 not in a.tables
    a.check_invariants()


def test_watermark_is_respected():
    """I7: committed blocks never exceed usable capacity."""
    a = Allocator(total_vram_blocks=100, block_size=16, watermark=0.10)
    assert a.usable_vram_blocks == 90
    with pytest.raises(AllocError):
        a.allocate(1, 16 * 95)
    a.allocate(2, 16 * 90)
    a.check_invariants()
    with pytest.raises(AllocError):
        a.append_tokens(2, 1)


# ------------------------------------------------------------------- CoW


def test_fork_shares_blocks_and_bumps_refcounts():
    a = Allocator(total_vram_blocks=64, block_size=16)
    parent = a.allocate(1, 64)     # 4 whole blocks
    child = a.fork(1, 2)
    assert child.blocks == parent.blocks
    for bid in parent.blocks:
        assert a.blocks[bid].ref_count == 2
    a.check_invariants()


def test_cow_on_shared_partial_tail():
    """Spec section 5.3's named bug: writing a shared partially-filled tail.

    The child's append must copy the tail rather than write through it, or the
    parent's KV is silently corrupted.
    """
    a = Allocator(total_vram_blocks=64, block_size=16)
    a.allocate(1, 20)              # 2 blocks, tail has 4/16 filled
    parent_tail = a.tables[1].blocks[-1]
    a.fork(1, 2)
    assert a.blocks[parent_tail].ref_count == 2

    a.append_tokens(2, 1)
    child_tail = a.tables[2].blocks[-1]

    assert child_tail != parent_tail, "child must not write into the shared tail"
    assert a.blocks[parent_tail].ref_count == 1
    assert a.cow_copies == 1
    # The executor must be told to copy the bytes across.
    assert (parent_tail, child_tail) in a.drain_pending_copies()
    a.check_invariants()


def test_cow_is_accounted_in_capacity_check():
    """Regression: DST BUG-003.

    A shared partial tail reports spare room, but the write still needs a fresh
    block for the copy. If that is not counted, the allocation overshoots the
    watermark and breaks I7.
    """
    a = Allocator(total_vram_blocks=10, block_size=4, watermark=0.05)
    a.allocate(1, 4 * 8 + 2)       # 9 blocks, partial tail
    a.fork(1, 2)
    assert a.blocks_needed_to_append(2, 1) == 1, "CoW block must be counted"
    with pytest.raises(AllocError):
        a.append_tokens(2, 1)
    a.check_invariants()


def test_free_of_shared_blocks_only_drops_one_ref():
    a = Allocator(total_vram_blocks=64, block_size=16)
    a.allocate(1, 64)
    a.fork(1, 2)
    blocks = list(a.tables[1].blocks)
    a.free(1)
    for bid in blocks:
        assert a.blocks[bid].ref_count == 1, "child still needs these"
    a.check_invariants()
    a.free(2)
    for bid in blocks:
        assert a.blocks[bid].ref_count == 0
    a.check_invariants()


def test_truncate_frees_whole_blocks_only():
    a = Allocator(total_vram_blocks=64, block_size=16)
    a.allocate(1, 64)              # 4 blocks
    a.truncate(1, 20)              # keep 2 blocks (16 + 4)
    t = a.tables[1]
    assert len(t.blocks) == 2
    assert t.filled_in_last == 4
    a.check_invariants()


def test_truncate_to_zero():
    a = Allocator(total_vram_blocks=64, block_size=16)
    a.allocate(1, 64)
    a.truncate(1, 0)
    assert a.tables[1].blocks == []
    assert a.tables[1].filled_in_last == 0
    a.check_invariants()


# ---------------------------------------------------------------- cache refs


def test_cache_reference_keeps_block_alive_after_free():
    a = Allocator(total_vram_blocks=32, block_size=16)
    t = a.allocate(1, 32)
    bid = t.blocks[0]
    a.retain_cached_block(bid)
    a.free(1)
    assert a.blocks[bid].ref_count == 1, "cache reference outlives the sequence"
    a.check_invariants()
    a.release_cached_block(bid)
    assert a.blocks[bid].ref_count == 0
    a.check_invariants()


def test_invariants_detect_injected_corruption():
    """The invariant checker must actually catch a broken ref count.

    A checker that never fires is worse than none, so this test corrupts state
    deliberately and asserts the violation is reported.
    """
    a = Allocator(total_vram_blocks=32, block_size=16)
    t = a.allocate(1, 32)
    a.check_invariants()
    a.blocks[t.blocks[0]].ref_count = 7      # lie about the ref count
    with pytest.raises(InvariantViolation):
        a.check_invariants()


# ------------------------------------------------------------ swap tier


def test_swap_out_and_in_round_trips():
    a = Allocator(total_vram_blocks=32, block_size=16, total_host_blocks=32)
    a.allocate(1, 64)
    vram_blocks = list(a.tables[1].blocks)

    pairs = a.swap_out(1)
    assert len(pairs) == len(vram_blocks)
    assert all(a.blocks[b].tier is Tier.HOST_PINNED for b in a.tables[1].blocks)
    a.check_invariants()

    a.swap_in(1)
    assert all(a.blocks[b].tier is Tier.VRAM for b in a.tables[1].blocks)
    a.check_invariants()


def test_swap_out_fails_cleanly_when_host_tier_full():
    a = Allocator(total_vram_blocks=32, block_size=16, total_host_blocks=1)
    a.allocate(1, 64)              # 4 blocks, host tier has room for 1
    with pytest.raises(AllocError):
        a.swap_out(1)
    a.check_invariants()


# -------------------------------------------------------- randomized walk


@pytest.mark.parametrize("seed", range(40))
def test_random_operation_sequence_preserves_invariants(seed: int):
    """Spec section 12.1 phase 1 exit criterion: I1..I7 under random ops."""
    rng = random.Random(seed)
    a = Allocator(
        total_vram_blocks=rng.randint(16, 96),
        block_size=rng.choice([4, 8, 16]),
        total_host_blocks=rng.choice([0, 16]),
        watermark=rng.choice([0.0, 0.01, 0.05]),
    )
    live: list[int] = []
    next_id = 0

    for _ in range(400):
        op = rng.random()

        if op < 0.30:                                    # allocate
            n = rng.randint(1, 60)
            try:
                a.allocate(next_id, n)
                live.append(next_id)
            except AllocError:
                pass
            next_id += 1

        elif op < 0.50 and live:                         # append
            sid = rng.choice(live)
            try:
                a.append_tokens(sid, rng.randint(1, 8))
            except AllocError:
                pass

        elif op < 0.65 and live:                         # free
            sid = live.pop(rng.randrange(len(live)))
            a.free(sid)

        elif op < 0.78 and live:                         # fork (prefix sharing)
            src = rng.choice(live)
            try:
                a.fork(src, next_id)
                live.append(next_id)
            except (AllocError, ValueError):
                pass
            next_id += 1

        elif op < 0.88 and live:                         # truncate (spec rollback)
            sid = rng.choice(live)
            cur = a.tables[sid].num_tokens(a.block_size)
            if cur:
                a.truncate(sid, rng.randint(0, cur))

        elif op < 0.94 and live and a.total_host_blocks:  # swap out
            sid = rng.choice(live)
            try:
                a.swap_out(sid)
            except AllocError:
                pass

        elif live and a.total_host_blocks:                # swap in
            sid = rng.choice(live)
            try:
                a.swap_in(sid)
            except AllocError:
                pass

        a.check_invariants()

    for sid in live:
        a.free(sid)
    a.check_invariants()
    assert a.free_vram_count + len(a.cache_refs) == a.total_vram_blocks

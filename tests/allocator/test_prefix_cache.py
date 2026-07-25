"""Radix prefix-cache tests.

The cache is block-aligned on purpose: only whole, immutable blocks are
shareable, because a partially-filled block will be written into by whichever
sequence extends it (spec section 5.3). These tests pin that boundary along with
the eviction and pinning rules that the scheduler depends on.
"""

from __future__ import annotations

import random

import pytest

from helios.core.prefix_cache import PrefixCache

BS = 4  # small block size keeps the expected values easy to verify by hand


def make_cache(block_size: int = BS) -> PrefixCache:
    return PrefixCache(block_size=block_size, enabled=True)


def test_empty_cache_misses():
    c = make_cache()
    blocks, matched = c.match([1, 2, 3, 4, 5])
    assert blocks == []
    assert matched == 0
    assert c.misses == 1


def test_insert_then_match_returns_block_aligned_prefix():
    c = make_cache()
    tokens = list(range(12))            # 3 whole blocks at BS=4
    added = c.insert(tokens, [10, 11, 12], step=1)
    assert added == [10, 11, 12]
    assert c.cached_blocks == 3

    blocks, matched = c.match(tokens + [99])
    assert blocks == [10, 11, 12]
    assert matched == 12


def test_match_never_consumes_the_entire_prompt():
    """At least one token must remain for the model to compute logits from.

    A full match would leave no forward pass to sample the first output token.
    """
    c = make_cache()
    tokens = list(range(12))
    c.insert(tokens, [1, 2, 3], step=1)

    blocks, matched = c.match(tokens)    # exact same prompt
    assert matched < len(tokens)
    assert matched == 8                  # last block held back


def test_partial_block_is_never_cached():
    c = make_cache()
    # 6 tokens == 1 whole block + 2 leftover; only the whole block is cacheable.
    added = c.insert([1, 2, 3, 4, 5, 6], [70, 71], step=1)
    assert added == [70]
    assert c.cached_blocks == 1


def test_insert_does_not_duplicate_an_existing_path():
    c = make_cache()
    tokens = list(range(12))
    c.insert(tokens, [1, 2, 3], step=1)
    again = c.insert(tokens, [4, 5, 6], step=2)
    assert again == [], "already-cached blocks must not be re-owned"
    assert c.cached_blocks == 3


def test_divergent_prompts_share_only_the_common_prefix():
    c = make_cache()
    shared = list(range(8))              # 2 blocks
    a = shared + [100, 101, 102, 103]
    b = shared + [200, 201, 202, 203]

    c.insert(a, [1, 2, 3], step=1)
    c.insert(b, [1, 2, 9], step=2)

    blocks, matched = c.match(b + [0])
    assert matched == 12
    assert blocks[:2] == [1, 2], "common prefix reuses the same physical blocks"
    assert blocks[2] == 9, "divergent tail uses its own block"


def test_total_blocks_matches_counter():
    """cached_blocks is a running counter; total_blocks() recounts the trie.

    They must agree, or eviction accounting silently drifts.
    """
    c = make_cache()
    c.insert(list(range(12)), [1, 2, 3], step=1)
    c.insert(list(range(8)) + [50, 51, 52, 53], [1, 2, 7], step=2)
    assert c.cached_blocks == c.total_blocks()


# ------------------------------------------------------------------ pinning


def test_pinned_nodes_are_not_evicted():
    c = make_cache()
    tokens = list(range(12))
    c.insert(tokens, [1, 2, 3], step=1)
    c.acquire(tokens, 12)

    assert c.evict(10) == [], "a pinned prefix must survive eviction"

    c.release(tokens, 12)
    assert sorted(c.evict(10)) == [1, 2, 3]


def test_release_is_balanced_with_acquire():
    """Unbalanced pins leave nodes permanently unevictable (DST BUG-005)."""
    c = make_cache()
    tokens = list(range(12))
    c.insert(tokens, [1, 2, 3], step=1)

    c.acquire(tokens, 12)
    c.acquire(tokens, 12)
    c.release(tokens, 12)
    assert c.evict(10) == [], "still pinned once"

    c.release(tokens, 12)
    assert len(c.evict(10)) == 3


def test_release_below_zero_is_safe():
    c = make_cache()
    tokens = list(range(12))
    c.insert(tokens, [1, 2, 3], step=1)
    c.release(tokens, 12)    # never acquired
    assert len(c.evict(10)) == 3


# ----------------------------------------------------------------- eviction


def test_eviction_unwinds_a_chain_of_interior_nodes():
    """Regression: DST BUG-005.

    Only leaves are evictable, but evicting a leaf can make its parent a leaf.
    Without re-scanning, a long shared-prefix chain keeps most of its blocks in
    interior nodes forever and the pool cannot be reclaimed.
    """
    c = make_cache()
    base = list(range(8))                       # 2 blocks
    c.insert(base + [90, 91, 92, 93], [1, 2, 3], step=1)
    c.insert(base + [80, 81, 82, 83], [1, 2, 4], step=2)
    assert c.total_blocks() == 4                # 1, 2 shared + 3 + 4

    freed = c.evict(99)
    assert sorted(freed) == [1, 2, 3, 4], "every block must become reclaimable"
    assert c.total_blocks() == 0


def test_eviction_prefers_least_recently_used():
    c = make_cache()
    old = list(range(4)) + [10, 11, 12, 13]
    new = list(range(4)) + [20, 21, 22, 23]
    c.insert(old, [1, 5], step=1)
    c.insert(new, [1, 6], step=99)

    freed = c.evict(1)
    assert freed == [5], "the older leaf goes first"


def test_eviction_order_is_deterministic():
    """Replay requires a total order over eviction candidates."""

    def run() -> list:
        c = make_cache()
        for i in range(6):
            tokens = list(range(4)) + [100 + i, 200 + i, 300 + i, 400 + i]
            c.insert(tokens, [1, 10 + i], step=5)   # identical steps: tie-break
        return c.evict(99)

    assert run() == run()


def test_evict_returns_nothing_when_cache_is_empty():
    c = make_cache()
    assert c.evict(5) == []


def test_disabled_cache_is_inert():
    c = PrefixCache(block_size=BS, enabled=False)
    assert c.insert(list(range(12)), [1, 2, 3], step=1) == []
    assert c.match(list(range(12))) == ([], 0)


def test_hit_ratio_accounting():
    c = make_cache()
    tokens = list(range(12))
    c.insert(tokens, [1, 2, 3], step=1)
    c.match(tokens + [5])          # hit
    c.match([999, 998, 997, 996])  # miss
    assert c.hits == 1
    assert c.misses == 1
    assert c.hit_ratio == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(20))
def test_random_insert_match_evict_keeps_accounting_consistent(seed: int):
    """cached_blocks must always equal a fresh recount of the trie."""
    rng = random.Random(seed)
    c = make_cache(block_size=rng.choice([2, 4, 8]))
    next_block = 0

    for _ in range(80):
        op = rng.random()
        n_tokens = rng.randrange(1, 40)
        tokens = [rng.randrange(0, 12) for _ in range(n_tokens)]

        if op < 0.5:
            n_blocks = n_tokens // c.block_size + 1
            blocks = list(range(next_block, next_block + n_blocks))
            next_block += n_blocks
            c.insert(tokens, blocks, step=rng.randrange(100))
        elif op < 0.8:
            c.match(tokens)
        else:
            c.evict(rng.randrange(1, 5))

        assert c.cached_blocks == c.total_blocks(), "block accounting drifted"
        assert c.cached_blocks >= 0

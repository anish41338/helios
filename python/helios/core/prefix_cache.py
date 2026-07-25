"""Radix-trie prompt prefix cache.

Spec section 3 (frontend: "prompt-cache prefix lookup, RadixTrie") and the
SGLang / RadixAttention line of prior art in section 18.

The trie is keyed on token ids at block granularity: only whole, fully-filled
KV blocks are shareable, because a partially-filled block would be written
into by whichever sequence extends it, and sharing a mutable tail is the
corruption bug that spec section 5.3 warns about. So the cache stores
block-aligned prefixes only.

Pure and deterministic: eviction order is driven by an injected step counter,
never by wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BlockId = int


@dataclass
class TrieNode:
    """One edge-compressed run of token ids covering >= 1 whole KV block."""

    token_ids: Tuple[int, ...] = ()
    block_ids: List[BlockId] = field(default_factory=list)
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    parent: Optional["TrieNode"] = None
    # Reference count of live sequences currently using this node's blocks.
    # A node with holders > 0 is pinned and must not be evicted.
    holders: int = 0
    last_used_step: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)


class PrefixCache:
    """Block-aligned radix trie over token-id prefixes.

    Lookup returns the longest cached block-aligned prefix. The caller
    (scheduler) is responsible for bumping allocator ref counts on the
    returned blocks -- the cache tracks logical sharing, the allocator tracks
    physical ownership. Keeping those two concerns separate is what keeps I2
    checkable.
    """

    def __init__(self, block_size: int, enabled: bool = True) -> None:
        self.block_size = block_size
        self.enabled = enabled
        self.root = TrieNode()
        self.hits = 0
        self.misses = 0
        self.cached_blocks = 0
        self.evictions = 0

    # -------------------------------------------------------------- lookup

    def match(self, token_ids: List[int]) -> Tuple[List[BlockId], int]:
        """Longest cached block-aligned prefix of token_ids.

        Returns (block_ids, num_matched_tokens). num_matched_tokens is always
        a multiple of block_size.

        We never match the entire prompt: at least one token must remain for
        the model to actually run a forward pass on, otherwise there is no
        logits row to sample the first output token from.
        """
        if not self.enabled or not token_ids:
            self.misses += 1
            return [], 0

        # Leave >= 1 token unmatched so prefill has something to compute.
        max_match = ((len(token_ids) - 1) // self.block_size) * self.block_size
        if max_match <= 0:
            self.misses += 1
            return [], 0

        blocks: List[BlockId] = []
        matched = 0
        node = self.root

        while matched < max_match:
            child = node.children.get(token_ids[matched])
            if child is None:
                break
            run = child.token_ids
            end = matched + len(run)

            if end <= max_match:
                if tuple(token_ids[matched:end]) != run:
                    break
                blocks.extend(child.block_ids)
                matched = end
                node = child
                continue

            # The edge runs past what we may match (either past max_match, or
            # past where the prompt diverges). Take the block-aligned part of it
            # rather than rejecting the edge outright: rejecting meant a repeat
            # of an identical prompt matched NOTHING, since its whole prompt sits
            # on one long edge and `end > max_match` always held. Reusing a
            # prefix of an edge is safe because each block is independently
            # addressed by the block table.
            usable = max_match - matched
            common = 0
            while common < usable and run[common] == token_ids[matched + common]:
                common += 1
            common -= common % self.block_size
            if common == 0:
                break
            blocks.extend(child.block_ids[: common // self.block_size])
            matched += common
            break

        if matched > 0:
            self.hits += 1
        else:
            self.misses += 1
        return blocks, matched

    # -------------------------------------------------------------- insert

    def insert(
        self, token_ids: List[int], block_ids: List[BlockId], step: int
    ) -> List[BlockId]:
        """Cache the block-aligned prefix of token_ids -> block_ids.

        Returns the block ids the cache newly took ownership of. Blocks already
        covered by an existing path are not duplicated and are not returned, so
        the caller can retain a reference on exactly the new ones without
        double-counting.
        """
        if not self.enabled:
            return []

        n_full = min(len(token_ids) // self.block_size, len(block_ids))
        if n_full == 0:
            return []

        aligned_tokens = token_ids[: n_full * self.block_size]
        added: List[BlockId] = []
        node = self.root
        pos = 0

        while pos < len(aligned_tokens):
            child = node.children.get(aligned_tokens[pos])
            if child is None:
                # Append the remaining run as one new edge.
                run = tuple(aligned_tokens[pos:])
                start_blk = pos // self.block_size
                owned = list(block_ids[start_blk:n_full])
                new = TrieNode(
                    token_ids=run,
                    block_ids=owned,
                    parent=node,
                    last_used_step=step,
                )
                node.children[aligned_tokens[pos]] = new
                added.extend(owned)
                self.cached_blocks += len(owned)
                return added

            run = child.token_ids
            end = pos + len(run)
            incoming = tuple(aligned_tokens[pos:end])

            if incoming == run:
                child.last_used_step = step
                node = child
                pos = end
                continue

            # Diverges mid-edge. Split the existing node at the common prefix,
            # keeping it block-aligned so both halves own whole blocks.
            common = 0
            limit = min(len(run), len(aligned_tokens) - pos)
            while common < limit and run[common] == aligned_tokens[pos + common]:
                common += 1
            common -= common % self.block_size
            if common == 0:
                return added  # nothing block-aligned to share

            self._split(node, child, common)
            child = node.children[aligned_tokens[pos]]
            child.last_used_step = step
            node = child
            pos += common

        return added

    def _split(self, parent: TrieNode, node: TrieNode, at: int) -> None:
        """Split `node`'s edge at token offset `at` (a block boundary)."""
        assert at % self.block_size == 0 and 0 < at < len(node.token_ids)
        n_blocks = at // self.block_size

        head = TrieNode(
            token_ids=node.token_ids[:at],
            block_ids=node.block_ids[:n_blocks],
            parent=parent,
            holders=node.holders,
            last_used_step=node.last_used_step,
        )
        node.token_ids = node.token_ids[at:]
        node.block_ids = node.block_ids[n_blocks:]
        node.parent = head
        head.children[node.token_ids[0]] = node
        parent.children[head.token_ids[0]] = head

    # ------------------------------------------------------------ pinning

    def acquire(self, token_ids: List[int], num_tokens: int) -> None:
        """Pin the nodes covering the first num_tokens against eviction."""
        for node in self._path(token_ids, num_tokens):
            node.holders += 1

    def release(self, token_ids: List[int], num_tokens: int) -> None:
        """Unpin nodes previously acquired."""
        for node in self._path(token_ids, num_tokens):
            if node.holders > 0:
                node.holders -= 1

    def _path(self, token_ids: List[int], num_tokens: int) -> List[TrieNode]:
        out: List[TrieNode] = []
        node = self.root
        pos = 0
        while pos < num_tokens:
            child = node.children.get(token_ids[pos])
            if child is None:
                break
            end = pos + len(child.token_ids)
            if end > num_tokens:
                break
            out.append(child)
            node = child
            pos = end
        return out

    # ----------------------------------------------------------- eviction

    def evict(self, n_blocks: int) -> List[BlockId]:
        """Evict least-recently-used unpinned leaves until n_blocks are freed.

        Returns block ids the caller must release in the allocator.

        Only leaves are evictable -- an interior node's blocks are a prefix of
        its children's, so removing it would strand them. But evicting a leaf
        can turn its parent into a leaf, so we re-scan after each removal.
        Without that, a long shared-prefix chain keeps almost all of its blocks
        in interior nodes, they never become evictable, and the engine
        livelocks with most of the pool cached and nothing reclaimable. Found
        by DST seed 15 while fixing BUG-004; see docs/DST.md BUG-005.
        """
        freed: List[BlockId] = []
        while len(freed) < n_blocks:
            victim = self._lru_leaf()
            if victim is None:
                break  # everything left is pinned by a live sequence
            freed.extend(victim.block_ids)
            self.cached_blocks -= victim.num_blocks
            self.evictions += 1
            parent = victim.parent
            if parent is not None:
                parent.children.pop(victim.token_ids[0], None)
                victim.parent = None
        return freed

    def _lru_leaf(self) -> Optional[TrieNode]:
        best: Optional[TrieNode] = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is self.root or node.children or node.holders > 0:
                continue
            # Tie-break on the first token id so eviction order is total and
            # reproducible under replay.
            key = (node.last_used_step, node.token_ids[0])
            if best is None or key < (best.last_used_step, best.token_ids[0]):
                best = node
        return best

    # -------------------------------------------------------------- stats

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def total_blocks(self) -> int:
        """Recount from the trie -- used by tests to catch cached_blocks drift."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is not self.root:
                total += node.num_blocks
        return total

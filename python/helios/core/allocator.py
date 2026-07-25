"""Paged KV cache allocator.

Implements HELIOS-SPEC.md section 5. The allocator owns a fixed pool of
fixed-size physical blocks and hands them out to sequences via block tables.
Sequences may share prefix blocks by reference count; a write to a shared
block triggers copy-on-write.

This module is PURE: no clock, no I/O, no RNG, no logging. Everything is a
deterministic function of the call sequence. That purity is what makes the
DST harness in helios.core.vopr possible, and it is load-bearing per spec
section 19.4 -- do not add a time or random import here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

BlockId = int
SeqId = int

# Tokens per block. Spec section 5.2: 16 or 32, tunable. Small blocks cut
# internal fragmentation but lengthen block tables and cost more gather
# overhead per attention step.
DEFAULT_BLOCK_SIZE = 16


class Tier(Enum):
    """Where a physical block's bytes live."""

    VRAM = "vram"
    HOST_PINNED = "host_pinned"


class AllocError(Exception):
    """Raised when a request cannot be satisfied from the free pool.

    The scheduler treats this as a signal to preempt, not as a bug. The DST
    harness injects it at arbitrary points (spec section 10.2).
    """


class InvariantViolation(AssertionError):
    """An I1..I7 check failed. Always a bug in HELIOS, never in the workload."""


@dataclass
class PhysicalBlock:
    """One fixed-size slab of KV storage."""

    id: BlockId
    ref_count: int = 0
    tier: Tier = Tier.VRAM


@dataclass
class BlockTable:
    """Per-sequence map from logical block index to physical block id.

    `blocks[i]` holds tokens [i*BLOCK_SIZE, (i+1)*BLOCK_SIZE). The final
    block is partially filled; `filled_in_last` says how much of it is live.
    """

    seq_id: SeqId
    blocks: List[BlockId] = field(default_factory=list)
    filled_in_last: int = 0

    def num_tokens(self, block_size: int) -> int:
        if not self.blocks:
            return 0
        return (len(self.blocks) - 1) * block_size + self.filled_in_last

    def clone_for(self, seq_id: SeqId) -> "BlockTable":
        """Shallow copy sharing all physical blocks. Caller must bump refs."""
        return BlockTable(
            seq_id=seq_id,
            blocks=list(self.blocks),
            filled_in_last=self.filled_in_last,
        )


class Allocator:
    """Fixed-pool paged block allocator with CoW prefix sharing.

    Free lists are FIFO deques rather than sets so that allocation order is
    reproducible across runs -- a set's iteration order would leak hash
    nondeterminism into the scheduler and break replay.
    """

    def __init__(
        self,
        total_vram_blocks: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        total_host_blocks: int = 0,
        watermark: float = 0.01,
    ) -> None:
        if total_vram_blocks <= 0:
            raise ValueError("total_vram_blocks must be positive")
        if not 0.0 <= watermark < 1.0:
            raise ValueError("watermark must be in [0, 1)")

        self.block_size = block_size
        self.watermark = watermark
        self.total_vram_blocks = total_vram_blocks
        self.total_host_blocks = total_host_blocks

        self.blocks: Dict[BlockId, PhysicalBlock] = {}
        self.free_vram: deque[BlockId] = deque()
        self.free_host: deque[BlockId] = deque()

        for i in range(total_vram_blocks):
            self.blocks[i] = PhysicalBlock(id=i, tier=Tier.VRAM)
            self.free_vram.append(i)
        for i in range(total_vram_blocks, total_vram_blocks + total_host_blocks):
            self.blocks[i] = PhysicalBlock(id=i, tier=Tier.HOST_PINNED)
            self.free_host.append(i)

        self.tables: Dict[SeqId, BlockTable] = {}

        # (src, dst) pairs produced by CoW, drained by the executor each step.
        self._pending_copies: List[Tuple[BlockId, BlockId]] = []

        # Blocks the prefix cache holds a reference to, and how many. These are
        # live but belong to no block table, so I2 accounts for them separately.
        self.cache_refs: Dict[BlockId, int] = {}

        # Counters, for metrics and for test assertions about CoW behaviour.
        self.cow_copies = 0
        self.alloc_calls = 0
        self.free_calls = 0

    # ---------------------------------------------------------------- sizing

    @staticmethod
    def bytes_per_block(
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
        dtype_bytes: int = 2,
        scale_bytes_per_token: int = 0,
    ) -> int:
        """Spec section 5.2 sizing formula. Factor 2 is K and V.

        `scale_bytes_per_token` accounts for a quantized cache's per-token,
        per-head scale (exec.paged_attn.QuantizedPagedKVCache). It is zero for an
        unquantized cache, so this reduces to the spec's formula exactly.

        It belongs in the formula rather than being applied as a correction
        afterwards: this number is what the engine divides its byte budget by to
        get a block count, and if the divisor understates the true block cost the
        engine hands out more blocks than the cache can hold.
        """
        return (
            2
            * n_layers
            * block_size
            * n_kv_heads
            * (head_dim * dtype_bytes + scale_bytes_per_token)
        )

    @classmethod
    def blocks_for_budget(
        cls,
        budget_bytes: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
        dtype_bytes: int = 2,
        scale_bytes_per_token: int = 0,
    ) -> int:
        per = cls.bytes_per_block(
            block_size, n_kv_heads, head_dim, n_layers, dtype_bytes,
            scale_bytes_per_token,
        )
        return max(0, budget_bytes // per)

    # ------------------------------------------------------------ capacity

    @property
    def usable_vram_blocks(self) -> int:
        """Blocks the scheduler may commit, honouring the reserve watermark.

        The reserve exists to avoid a deadlock where every running sequence
        needs one more block to make progress and none is available (spec
        section 5.2, invariant I7).
        """
        return int(self.total_vram_blocks * (1.0 - self.watermark))

    @property
    def committed_vram_blocks(self) -> int:
        return self.total_vram_blocks - len(self.free_vram)

    @property
    def free_vram_count(self) -> int:
        return len(self.free_vram)

    def can_allocate(self, n_blocks: int) -> bool:
        """True if n_blocks more VRAM blocks fit under the watermark."""
        if n_blocks <= 0:
            return True
        if len(self.free_vram) < n_blocks:
            return False
        return self.committed_vram_blocks + n_blocks <= self.usable_vram_blocks

    def blocks_needed_for(self, n_tokens: int) -> int:
        """Blocks required to hold n_tokens from empty."""
        if n_tokens <= 0:
            return 0
        return (n_tokens + self.block_size - 1) // self.block_size

    def blocks_needed_to_append(self, seq_id: SeqId, n_tokens: int) -> int:
        """Additional blocks needed to append n_tokens to an existing seq.

        Accounts for spare room in the current partially-filled tail block,
        AND for a copy-on-write of that tail if it is shared.

        The CoW term is easy to miss and its absence is a real bug: a shared
        partial tail reports 0 blocks needed (there is spare room) but the
        write must still copy, so the allocation silently overshoots the
        watermark and breaks I7. Found by DST seeds 33/50/55; see docs/DST.md
        BUG-003, and note that spec section 5.3 predicted exactly this.
        """
        if n_tokens <= 0:
            return 0
        table = self.tables.get(seq_id)
        if table is None or not table.blocks:
            return self.blocks_needed_for(n_tokens)

        cow = 0
        if table.filled_in_last < self.block_size:
            if self.blocks[table.blocks[-1]].ref_count > 1:
                cow = 1  # the shared tail must be copied before it is written

        spare = self.block_size - table.filled_in_last
        if n_tokens <= spare:
            return cow
        return cow + self.blocks_needed_for(n_tokens - spare)

    # ------------------------------------------------------- alloc and free

    def _take_vram(self) -> BlockId:
        if not self.free_vram:
            raise AllocError("VRAM block pool exhausted")
        bid = self.free_vram.popleft()
        blk = self.blocks[bid]
        assert blk.ref_count == 0, f"block {bid} was free with ref_count>0"
        blk.ref_count = 1
        return bid

    def _release(self, bid: BlockId) -> None:
        """Drop one reference; return the block to its free list at zero."""
        blk = self.blocks[bid]
        assert blk.ref_count > 0, f"double free of block {bid}"
        blk.ref_count -= 1
        if blk.ref_count == 0:
            if blk.tier is Tier.VRAM:
                self.free_vram.append(bid)
            else:
                self.free_host.append(bid)

    def allocate(self, seq_id: SeqId, n_tokens: int) -> BlockTable:
        """Create a block table for a new sequence holding n_tokens.

        Raises AllocError without leaving partial state -- the caller can
        retry after preempting. Atomicity matters: a half-allocated table
        would violate I3.
        """
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already allocated")
        self.alloc_calls += 1

        need = self.blocks_needed_for(n_tokens)
        if not self.can_allocate(need):
            raise AllocError(
                f"cannot allocate {need} blocks for seq {seq_id}: "
                f"{len(self.free_vram)} free, "
                f"{self.usable_vram_blocks - self.committed_vram_blocks} under watermark"
            )

        taken: List[BlockId] = []
        try:
            for _ in range(need):
                taken.append(self._take_vram())
        except AllocError:
            for bid in taken:  # roll back so we never leak on the error path
                self._release(bid)
            raise

        filled = n_tokens - (need - 1) * self.block_size if need else 0
        table = BlockTable(seq_id=seq_id, blocks=taken, filled_in_last=filled)
        self.tables[seq_id] = table
        return table

    def append_tokens(self, seq_id: SeqId, n_tokens: int) -> List[BlockId]:
        """Grow a sequence by n_tokens, returning newly allocated block ids.

        Performs copy-on-write if the tail block is shared -- writing into a
        shared partially-filled tail is the corruption bug called out in
        spec section 5.3 and is exactly what I6 forbids.
        """
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        if n_tokens == 0:
            return []
        table = self.tables[seq_id]
        need = self.blocks_needed_to_append(seq_id, n_tokens)
        if not self.can_allocate(need):
            raise AllocError(
                f"cannot append {n_tokens} tokens to seq {seq_id}: needs {need} blocks"
            )

        # CoW the tail before writing into it. `need` already includes this
        # copy (see blocks_needed_to_append), so the capacity check above
        # covers the block _cow_block is about to take.
        if table.blocks and table.filled_in_last < self.block_size:
            tail = table.blocks[-1]
            if self.blocks[tail].ref_count > 1:
                self._cow_block(table, len(table.blocks) - 1)

        new_blocks: List[BlockId] = []
        remaining = n_tokens

        if table.blocks:
            spare = self.block_size - table.filled_in_last
            used = min(spare, remaining)
            table.filled_in_last += used
            remaining -= used

        while remaining > 0:
            bid = self._take_vram()
            new_blocks.append(bid)
            table.blocks.append(bid)
            used = min(self.block_size, remaining)
            table.filled_in_last = used
            remaining -= used

        return new_blocks

    def free(self, seq_id: SeqId) -> None:
        """Release every block held by a sequence and drop its table."""
        table = self.tables.pop(seq_id, None)
        if table is None:
            return
        self.free_calls += 1
        for bid in table.blocks:
            self._release(bid)

    def truncate(self, seq_id: SeqId, keep_tokens: int) -> None:
        """Shrink a sequence to keep_tokens, freeing whole blocks that fall off.

        Used by speculative rollback (spec section 7.2) and by chunked
        prefill error paths. Keeps the block table hole-free (I4).
        """
        table = self.tables[seq_id]
        if keep_tokens < 0:
            raise ValueError("keep_tokens must be >= 0")
        current = table.num_tokens(self.block_size)
        if keep_tokens >= current:
            return

        need = self.blocks_needed_for(keep_tokens)
        while len(table.blocks) > need:
            self._release(table.blocks.pop())
        table.filled_in_last = (
            keep_tokens - (need - 1) * self.block_size if need else 0
        )

    # ---------------------------------------------------- prefix sharing

    def fork(self, parent_seq: SeqId, child_seq: SeqId) -> BlockTable:
        """Share a parent's blocks with a child (prefix cache hit, beam fork).

        All blocks become shared; the first write on either side copies.
        """
        if child_seq in self.tables:
            raise ValueError(f"seq {child_seq} already allocated")
        parent = self.tables[parent_seq]
        child = parent.clone_for(child_seq)
        for bid in child.blocks:
            self.blocks[bid].ref_count += 1
        self.tables[child_seq] = child
        return child

    def retain_cached_block(self, bid: BlockId) -> None:
        """Take a reference on behalf of the prefix cache.

        A cached block outlives the sequence that produced it, so someone must
        hold a reference or it would be recycled while the trie still points at
        it. The cache is not a block table, so it gets its own accounted
        reference (see `cache_refs`) and I2 counts it explicitly.
        """
        blk = self.blocks[bid]
        assert blk.ref_count > 0, f"cannot retain freed block {bid}"
        blk.ref_count += 1
        self.cache_refs[bid] = self.cache_refs.get(bid, 0) + 1

    def release_cached_block(self, bid: BlockId) -> None:
        """Drop a prefix-cache reference, freeing the block if it was the last."""
        held = self.cache_refs.get(bid, 0)
        if held <= 0:
            return
        if held == 1:
            del self.cache_refs[bid]
        else:
            self.cache_refs[bid] = held - 1
        self._release(bid)

    def share_prefix(
        self, seq_id: SeqId, block_ids: List[BlockId], filled_in_last: int
    ) -> BlockTable:
        """Build a table over pre-existing blocks (prefix cache attach)."""
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already allocated")
        for bid in block_ids:
            if self.blocks[bid].ref_count == 0:
                raise InvariantViolation(
                    f"cannot share free block {bid} into seq {seq_id}"
                )
        for bid in block_ids:
            self.blocks[bid].ref_count += 1
        table = BlockTable(
            seq_id=seq_id, blocks=list(block_ids), filled_in_last=filled_in_last
        )
        self.tables[seq_id] = table
        return table

    def _cow_block(self, table: BlockTable, logical_idx: int) -> BlockId:
        """Replace a shared block with a private copy. Returns the new id.

        The physical byte copy is the executor's job; the allocator only
        rewires the mapping and reports the pair via the returned id so the
        caller can schedule the copy.
        """
        old = table.blocks[logical_idx]
        new = self._take_vram()
        table.blocks[logical_idx] = new
        self._release(old)
        self.cow_copies += 1
        self._pending_copies.append((old, new))
        return new

    def drain_pending_copies(self) -> List[Tuple[BlockId, BlockId]]:
        """Hand the executor the (src, dst) block copies CoW created."""
        out = list(self._pending_copies)
        self._pending_copies = []
        return out

    # --------------------------------------------------------- swap tier

    def swap_out(self, seq_id: SeqId) -> List[Tuple[BlockId, BlockId]]:
        """Move a sequence's blocks to the host tier. Returns (vram, host) pairs.

        Spec section 6.3: only worth it for long-generated sequences, since it
        costs PCIe bandwidth that competes with the KV transfer channel.
        """
        table = self.tables[seq_id]
        vram_blocks = [b for b in table.blocks if self.blocks[b].tier is Tier.VRAM]
        if len(self.free_host) < len(vram_blocks):
            raise AllocError("host swap tier full")

        pairs: List[Tuple[BlockId, BlockId]] = []
        for idx, bid in enumerate(table.blocks):
            if self.blocks[bid].tier is not Tier.VRAM:
                continue
            if self.blocks[bid].ref_count > 1:
                # Shared blocks stay resident; another sequence still needs them.
                continue
            host = self.free_host.popleft()
            self.blocks[host].ref_count = 1
            table.blocks[idx] = host
            self._release(bid)
            pairs.append((bid, host))
        return pairs

    def swap_in(self, seq_id: SeqId) -> List[Tuple[BlockId, BlockId]]:
        """Bring a swapped-out sequence back to VRAM. Returns (host, vram) pairs."""
        table = self.tables[seq_id]
        host_blocks = [
            b for b in table.blocks if self.blocks[b].tier is Tier.HOST_PINNED
        ]
        if not self.can_allocate(len(host_blocks)):
            raise AllocError("cannot swap in: insufficient VRAM")

        pairs: List[Tuple[BlockId, BlockId]] = []
        for idx, bid in enumerate(table.blocks):
            if self.blocks[bid].tier is not Tier.HOST_PINNED:
                continue
            vram = self._take_vram()
            table.blocks[idx] = vram
            self._release(bid)
            pairs.append((bid, vram))
        return pairs

    # ------------------------------------------------------- introspection

    def block_table(self, seq_id: SeqId) -> Optional[BlockTable]:
        return self.tables.get(seq_id)

    def utilization(self) -> float:
        return self.committed_vram_blocks / self.total_vram_blocks

    def iter_tables(self) -> Iterator[BlockTable]:
        # Sorted so callers cannot depend on dict insertion order.
        for seq_id in sorted(self.tables):
            yield self.tables[seq_id]

    # ---------------------------------------------------------- invariants

    def check_invariants(self) -> None:
        """Assert I1..I7 from spec section 5.4.

        Called after every scheduler step in debug builds and after every
        single operation by the DST harness and property tests. O(blocks), so
        it is gated off the hot path in benchmark runs.
        """
        free_vram = set(self.free_vram)
        free_host = set(self.free_host)

        if len(free_vram) != len(self.free_vram):
            raise InvariantViolation("I1: duplicate ids in free_vram")
        if len(free_host) != len(self.free_host):
            raise InvariantViolation("I1: duplicate ids in free_host")
        if free_vram & free_host:
            raise InvariantViolation("I1: block in both free lists")

        # I2: ref_count matches referencing block tables plus cache refs.
        observed: Dict[BlockId, int] = {}
        for table in self.tables.values():
            for bid in table.blocks:
                observed[bid] = observed.get(bid, 0) + 1
        for bid, n in self.cache_refs.items():
            observed[bid] = observed.get(bid, 0) + n

        for bid, count in observed.items():
            actual = self.blocks[bid].ref_count
            if actual != count:
                raise InvariantViolation(
                    f"I2: block {bid} ref_count={actual} but {count} holders "
                    f"(tables + cache refs) reference it"
                )
            # I1: a referenced block must not also be free.
            if bid in free_vram or bid in free_host:
                raise InvariantViolation(f"I1: block {bid} is both free and referenced")

        for bid, blk in self.blocks.items():
            if blk.ref_count == 0 and bid not in free_vram and bid not in free_host:
                raise InvariantViolation(f"I1: block {bid} leaked (ref 0, not free)")
            if blk.ref_count > 0 and bid not in observed:
                raise InvariantViolation(
                    f"I2: block {bid} has ref_count={blk.ref_count} but no "
                    f"block table or cache reference"
                )

        # I3: conservation of blocks per tier.
        vram_referenced = {
            b for b in observed if self.blocks[b].tier is Tier.VRAM
        }
        if len(free_vram) + len(vram_referenced) != self.total_vram_blocks:
            raise InvariantViolation(
                f"I3: {len(free_vram)} free + {len(vram_referenced)} used "
                f"!= {self.total_vram_blocks} total vram"
            )
        host_referenced = {
            b for b in observed if self.blocks[b].tier is Tier.HOST_PINNED
        }
        if len(free_host) + len(host_referenced) != self.total_host_blocks:
            raise InvariantViolation("I3: host tier block count mismatch")

        for table in self.tables.values():
            # I4: no holes. Every entry must be a real block id.
            for i, bid in enumerate(table.blocks):
                if bid not in self.blocks:
                    raise InvariantViolation(
                        f"I4: seq {table.seq_id} logical block {i} -> unknown {bid}"
                    )
            # I5: filled_in_last in [1, BLOCK_SIZE] for a non-empty sequence.
            if table.blocks:
                if not 1 <= table.filled_in_last <= self.block_size:
                    raise InvariantViolation(
                        f"I5: seq {table.seq_id} filled_in_last={table.filled_in_last}"
                    )
            elif table.filled_in_last != 0:
                raise InvariantViolation(
                    f"I5: empty seq {table.seq_id} has filled_in_last="
                    f"{table.filled_in_last}"
                )

        # I7: committed blocks stay under the watermark.
        if self.committed_vram_blocks > self.usable_vram_blocks:
            raise InvariantViolation(
                f"I7: committed {self.committed_vram_blocks} > "
                f"usable {self.usable_vram_blocks}"
            )

    # I6 (a block with ref_count > 1 is never written in place) is not
    # checkable from allocator state alone -- it is a property of the write
    # path. It is enforced by append_tokens' CoW and asserted by
    # tests/allocator/test_cow.py, which tracks writes explicitly.

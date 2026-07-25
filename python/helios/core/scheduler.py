"""The deterministic scheduler core.

Implements HELIOS-SPEC.md section 6: iteration-level continuous batching,
token-bucket admission per SLO class, recompute-first preemption, chunked
prefill.

DETERMINISM CONTRACT (spec section 19.4 -- load-bearing, not a nice-to-have):

  * No wall-clock reads. Time is `step_t`, an integer advanced by step().
  * No unseeded randomness. The scheduler makes no random choices at all;
    tie-breaks are total orders over (priority, seq_id).
  * No I/O, no threads, no logging.
  * No iteration over unordered containers. Dicts are iterated via sorted
    keys or maintained-order lists only.

Any change that breaks one of those makes the DST harness unable to replay a
failing seed, which defeats the whole point. The test
tests/scheduler/test_determinism.py guards this by running identical
workloads twice and diffing the full step trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .allocator import AllocError, Allocator
from .execstep import (
    BlockCopy,
    DecodeItem,
    ExecFault,
    ExecOutputs,
    ExecStep,
    Executor,
    PrefillItem,
)
from .prefix_cache import PrefixCache
from .types import (
    FinishReason,
    Request,
    SeqId,
    SeqState,
    Sequence,
    SloClass,
)


@dataclass
class SchedulerConfig:
    """Tunables. All units are tokens or steps -- never seconds."""

    max_num_seqs: int = 64
    # Token budget per forward pass. Caps prefill chunk size so a long prompt
    # cannot monopolise a step and spike TPOT for in-flight decodes
    # (spec section 6.1, Sarathi-Serve chunked prefill).
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    block_size: int = 16
    watermark: float = 0.01

    enable_chunked_prefill: bool = True
    enable_prefix_cache: bool = True
    enable_spec_decode: bool = False
    spec_gamma: int = 4

    # Adaptive speculation (spec section 7.3): speculation stops paying at
    # large batch and on high-entropy output, so we gate on both.
    spec_max_batch_size: int = 8
    spec_min_acceptance: float = 0.5
    spec_window: int = 64

    # Preemption. Swap only when recompute would cost more (spec section 6.3).
    enable_swap: bool = False
    swap_generated_ratio: float = 4.0

    # Class A protection: never preempted while at most this many are in flight.
    class_a_protect_limit: int = 8

    # Consecutive no-progress steps before the starvation backstop drops the
    # prefix cache. Small enough to recover quickly, large enough that a step
    # spent purely on prefill chunks or block copies is not mistaken for a
    # stall.
    stall_threshold: int = 8

    # Consecutive steps the waiting-queue head may be passed over before it
    # gets an exclusive reservation. Small requests are allowed to overtake a
    # too-large one (so they are not head-of-line blocked), but only for this
    # long, so the large request cannot be starved forever.
    reservation_after_steps: int = 32

    # Token-bucket admission per class. Refill is per-step, not per-second,
    # which is what keeps admission deterministic (spec section 6.2).
    bucket_capacity: Dict[SloClass, int] = field(
        default_factory=lambda: {
            SloClass.A: 16384,
            SloClass.B: 8192,
            SloClass.C: 4096,
        }
    )
    bucket_refill_per_step: Dict[SloClass, int] = field(
        default_factory=lambda: {
            SloClass.A: 2048,
            SloClass.B: 1024,
            SloClass.C: 256,
        }
    )


@dataclass
class SchedulerStats:
    """Counters exported to /metrics (spec section 9.2)."""

    step: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    finished_seqs: int = 0
    aborted_seqs: int = 0
    preemptions_recompute: int = 0
    preemptions_swap: int = 0
    spec_drafted: int = 0
    spec_accepted: int = 0
    exec_faults: int = 0
    empty_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.spec_accepted / self.spec_drafted if self.spec_drafted else 0.0


class TokenBucket:
    """Per-class admission bucket, refilled by step count.

    Prevents class C from starving class A. Deliberately step-driven: a
    wall-clock refill would make admission unreproducible.
    """

    def __init__(self, capacity: int, refill_per_step: int) -> None:
        self.capacity = capacity
        self.refill_per_step = refill_per_step
        self.tokens = capacity
        self._last_refill_step = 0

    def refill_to(self, step: int) -> None:
        if step <= self._last_refill_step:
            return
        elapsed = step - self._last_refill_step
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_step)
        self._last_refill_step = step

    def try_take(self, n: int) -> bool:
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class Scheduler:
    """Iteration-level continuous batching scheduler.

    Owns the allocator, the prefix cache, and the waiting/running queues. One
    call to step() == one executor forward pass == one increment of step_t.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        allocator: Allocator,
        prefix_cache: Optional[PrefixCache] = None,
    ) -> None:
        self.config = config
        self.allocator = allocator
        self.prefix_cache = prefix_cache or PrefixCache(
            config.block_size, enabled=config.enable_prefix_cache
        )

        self.step_t = 0
        self._next_seq_id = 0

        # Ordered containers only. waiting is FIFO within a priority class;
        # running preserves admission order so batch composition is stable.
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []
        self.finished: List[Sequence] = []

        self.seqs: Dict[SeqId, Sequence] = {}
        self.stats = SchedulerStats()

        self.buckets: Dict[SloClass, TokenBucket] = {
            cls: TokenBucket(
                config.bucket_capacity[cls], config.bucket_refill_per_step[cls]
            )
            for cls in (SloClass.A, SloClass.B, SloClass.C)
        }

        # Sliding window of recent acceptance outcomes for adaptive gamma.
        self._accept_window: deque[Tuple[int, int]] = deque(maxlen=config.spec_window)

        # Consecutive steps in which no token was committed. Drives the
        # starvation backstop; reset by any observable progress.
        self._stalled_steps = 0
        self._last_progress_tokens = 0

        # Anti-starvation bookkeeping for admission (see _admit).
        self._skipped = 0
        self._skipped_head_steps = 0

        # Sequences whose prompt tokens must be re-inserted into the prefix
        # cache once their prefill completes.
        self._pending_cache_insert: List[SeqId] = []

        # Block copies the swap tier owes the executor next step.
        self._pending_swap_in: List[BlockCopy] = []
        self._pending_swap_out: List[BlockCopy] = []

    # ------------------------------------------------------------- admission

    def add_request(self, request: Request) -> SeqId:
        """Enqueue a request. Does not allocate KV -- that happens on admission."""
        if request.prompt_len == 0:
            raise ValueError("empty prompt")
        if request.prompt_len >= self.config.max_model_len:
            raise ValueError(
                f"prompt length {request.prompt_len} exceeds max_model_len "
                f"{self.config.max_model_len}"
            )
        request.params.validate()

        # Reject what the KV pool physically cannot hold, even if it is within
        # max_model_len. Such a request could never be admitted, and queueing
        # it forever both starves it and (since admission stops at the first
        # request it cannot place) blocks everything behind it.
        # Found by DST seed 2; see docs/DST.md BUG-002.
        if not self.can_ever_serve(request.prompt_len):
            raise ValueError(
                f"prompt length {request.prompt_len} exceeds KV capacity "
                f"({self.max_servable_tokens} tokens); reduce prompt or "
                f"increase kv pool size"
            )

        seq_id = self._next_seq_id
        self._next_seq_id += 1
        request.arrival_step = self.step_t

        seq = Sequence(seq_id=seq_id, request=request)
        self.seqs[seq_id] = seq
        self.waiting.append(seq)
        return seq_id

    @property
    def max_servable_tokens(self) -> int:
        """Longest sequence the KV pool can hold, honouring the watermark.

        This is the real context limit on a given box -- it is min'd with
        max_model_len because a config may promise more context than the
        cache can physically back.
        """
        return min(
            self.allocator.usable_vram_blocks * self.config.block_size,
            self.config.max_model_len,
        )

    def can_ever_serve(self, prompt_len: int) -> bool:
        """True if a prompt of this length could fit given an empty cache.

        Requires room for the prompt AND at least one more block to grow into.
        A sequence that exactly fills the pool cannot emit even one token: it
        would OOM on its first append, be preempted, re-prefill, and repeat
        forever. Reserving growth headroom at admission is what makes decode
        progress guaranteed rather than hoped for. DST BUG-010 (seed 140).
        """
        return (
            self.allocator.blocks_needed_for(prompt_len) + 1
            <= self.allocator.usable_vram_blocks
        )

    def abort(self, seq_id: SeqId) -> bool:
        """Cancel a sequence in any state. Idempotent.

        DST cancels at every FSM state (spec section 10.2), so this must be
        total: waiting, running, swapped, or already gone.
        """
        seq = self.seqs.get(seq_id)
        if seq is None or seq.is_finished:
            return False

        if seq in self.running:
            self.running.remove(seq)
            self._free_seq_kv(seq)
        elif seq in self.swapped:
            self.swapped.remove(seq)
            self._free_seq_kv(seq)
        else:
            try:
                self.waiting.remove(seq)
            except ValueError:
                pass
            # A preempted-recompute seq holds no KV, but a waiting seq that was
            # never admitted might still have a cache pin; _free_seq_kv is safe
            # either way.
            self._free_seq_kv(seq)

        seq.state = SeqState.ABORTED
        seq.finish_reason = FinishReason.ABORTED
        self.stats.aborted_seqs += 1
        self.finished.append(seq)
        return True

    # ------------------------------------------------------------------ step

    def step(self, executor: Executor) -> ExecOutputs:
        """Run exactly one scheduling iteration and one executor pass.

        Order matters and mirrors spec section 6.1's pseudocode:
          1. reap finished sequences (frees KV before we try to admit)
          2. admit from waiting while budget and memory allow
          3. preempt until every running sequence can take a step
          4. build the ExecStep, run it, apply outputs
        """
        self.step_t += 1
        self.stats.step = self.step_t

        self._reap_finished()
        for bucket in self.buckets.values():
            bucket.refill_to(self.step_t)

        self._maybe_swap_in()
        self._admit()
        self._ensure_running_can_progress()

        # Starvation backstop. Trigger on lack of *progress*, not on an empty
        # running list: the pathological case is a single sequence that holds
        # most of the pool, cannot grow, and cannot be preempted (BUG-009), so
        # `running` stays at 1 forever while nothing advances. Cached blocks are
        # the only reclaimable memory left, so drop them wholesale.
        # DST BUG-008 (seeds 140/167/2).
        if self.waiting and (
            self._stalled_steps >= self.config.stall_threshold
            or (not self.running and self.allocator.cache_refs)
        ):
            # Nothing running with requests queued means the cache is holding
            # memory that no live sequence needs -- reclaim it immediately
            # rather than waiting out the stall threshold.
            if self._drop_prefix_cache():
                self._stalled_steps = 0
                self._admit()  # retry now that memory is back

        step = self._build_exec_step()
        if step.is_empty:
            self.stats.empty_steps += 1
            self._update_progress()  # an empty step is a stalled step
            return ExecOutputs(step_id=self.step_t)

        try:
            outputs = executor.run(step)
        except ExecFault as fault:
            self.stats.exec_faults += 1
            self._handle_fault(fault)
            self._update_progress()  # a faulted step made no progress
            return ExecOutputs(step_id=self.step_t)

        self._apply_outputs(step, outputs)
        self._update_progress()
        return outputs

    def _update_progress(self) -> None:
        """Track whether the engine is advancing, for the starvation backstop.

        Progress is measured as *retired* work -- sequences finished plus output
        tokens committed -- deliberately NOT prefill tokens. Prefill work that
        is repeatedly preempted and redone inflates a prefill-token counter
        forever while the engine completes nothing, which silently disabled this
        backstop (seed 2). Only work that cannot be undone counts.
        """
        retired = (
            self.stats.finished_seqs
            + self.stats.aborted_seqs
            + self.stats.decode_tokens
        )
        if retired > self._last_progress_tokens:
            self._last_progress_tokens = retired
            self._stalled_steps = 0
        else:
            self._stalled_steps += 1

    # -------------------------------------------------------------- internals

    def _reap_finished(self) -> None:
        """Remove sequences that completed last step and free their KV.

        Runs before admission so a finishing sequence's blocks are available
        to the sequence taking its slot in the same iteration -- that
        immediacy is the point of continuous batching.
        """
        still_running: List[Sequence] = []
        for seq in self.running:
            if seq.is_finished:
                self._free_seq_kv(seq)
                self.finished.append(seq)
                self.stats.finished_seqs += 1
            else:
                still_running.append(seq)
        self.running = still_running

    def _admit(self) -> None:
        """Move sequences from waiting to running while resources permit."""
        # Sort waiting by (class, arrival) so class A jumps the queue but
        # order within a class stays FIFO -- no starvation inside a class.
        # Stable sort over a deque snapshot; seq_id breaks any remaining tie.
        pending = sorted(
            self.waiting, key=lambda s: (int(s.slo_class), s.request.arrival_step, s.seq_id)
        )

        # Anti-starvation reservation. If the head of the queue has been passed
        # over for too many consecutive steps, stop admitting anything smaller
        # and let memory drain until it fits. Without this, allowing smaller
        # requests to overtake (BUG-014) would let a steady stream of small
        # requests starve a large one indefinitely -- trading one livelock for
        # another. Applies per class so it cannot invert SLO priority.
        blocked_head: Optional[Sequence] = None
        if pending and self._skipped_head_steps >= self.config.reservation_after_steps:
            blocked_head = pending[0]

        admitted: List[Sequence] = []
        self._skipped = 0
        budget = self.config.max_num_batched_tokens
        # Decodes already claimed part of this step's budget.
        budget -= len(self.running)

        for seq in pending:
            if len(self.running) + len(admitted) >= self.config.max_num_seqs:
                break
            if budget <= 0:
                break
            if blocked_head is not None and seq is not blocked_head:
                # A reservation is active: only the starved head may be
                # admitted this step. Everything else waits for memory to free.
                continue

            # Decide everything BEFORE taking any resource. Every early exit
            # below must leave the sequence owning nothing, so the cheap
            # checks (budget, rate limit) come first and the prefix-cache
            # attachment happens only once admission is certain to proceed.
            # Interleaving the two is how BUG-007 arose.
            cache_blocks, cache_hit = self._peek_prefix_cache(seq)
            new_tokens = seq.total_tokens - cache_hit
            chunk = min(new_tokens, budget)
            if not self.config.enable_chunked_prefill and new_tokens > budget:
                # The prompt cannot be split, so it only fits in a step it has
                # to itself. Admit it alone rather than skipping it forever:
                # without this, any prompt longer than max_num_batched_tokens
                # is unschedulable and -- because admission used to `break`
                # here -- it also blocked every smaller request behind it.
                # Found by DST seed 2 (see docs/DST.md, BUG-001).
                if self.running or admitted:
                    break  # retry next step, once the batch has drained
                chunk = new_tokens

            bucket = self.buckets[seq.slo_class]
            if not bucket.try_take(chunk):
                continue  # class is rate-limited this step; try the next one

            self._attach_prefix_cache(seq, cache_blocks, cache_hit)

            if not self._allocate_for(seq):
                # Out of memory. Refund the tokens we took.
                bucket.tokens = min(bucket.capacity, bucket.tokens + chunk)
                # Roll back the prefix-cache attachment made above. A sequence
                # that stays in the waiting queue must hold no KV and no pin:
                # a pin left behind by a failed admission makes those cache
                # nodes permanently unevictable, so the memory needed to admit
                # the sequence can never be reclaimed -- a self-sustaining
                # livelock. DST BUG-007 (seeds 15/28/46).
                self._detach_prefix_cache(seq)
                # Keep scanning rather than stopping. The binding constraint is
                # memory, not queue position: a request that does not fit right
                # now must not block a smaller one behind it that does. Stopping
                # here starved requests needing 3 blocks while 9 were free,
                # because a 14-block request sat ahead of them -- indefinitely,
                # since the running sequences it was waiting on had already
                # stabilised. DST BUG-006/BUG-014 (seeds 2/25/78/1136/3218).
                #
                # Fairness is preserved by the `pending` sort (class, then
                # arrival): a large request keeps its place and is retried every
                # step, and _skipped tracks it so it cannot be starved forever
                # by a stream of small ones.
                self._skipped += 1
                continue

            seq.state = SeqState.PREFILL if not seq.is_prefill_done else SeqState.DECODE
            admitted.append(seq)
            budget -= chunk

        # Track whether the queue head is being starved, to decide when the
        # reservation above should kick in.
        if pending and pending[0] not in admitted:
            self._skipped_head_steps += 1
        else:
            self._skipped_head_steps = 0

        for seq in admitted:
            try:
                self.waiting.remove(seq)
            except ValueError:
                pass
            self.running.append(seq)

    def _peek_prefix_cache(self, seq: Sequence) -> Tuple[List[int], int]:
        """Look up a cached prefix WITHOUT taking any reference.

        Split from the attach step so admission can bail out at any point
        without having to unwind cache state (see BUG-007).
        """
        if not self.config.enable_prefix_cache or not seq.request.prefix_cache:
            return [], 0
        if seq.num_computed_tokens > 0:
            return [], 0  # resuming a chunked prefill; already attached
        if self.allocator.block_table(seq.seq_id) is not None:
            return [], 0
        return self.prefix_cache.match(seq.request.prompt_token_ids)

    def _attach_prefix_cache(
        self, seq: Sequence, blocks: List[int], matched: int
    ) -> bool:
        """Commit a prefix-cache hit found by _peek_prefix_cache."""
        if matched == 0:
            return False
        try:
            self.allocator.share_prefix(seq.seq_id, blocks, self.config.block_size)
        except Exception:
            return False

        self._pin_prefix(seq, matched)
        seq.cached_prefix_len = matched
        seq.num_computed_tokens = matched
        return True

    def _detach_prefix_cache(self, seq: Sequence) -> None:
        """Undo a prefix-cache attachment, returning the sequence to pristine.

        Used when admission fails after attaching: the sequence goes back to
        the waiting queue owning nothing.
        """
        self._unpin_prefix(seq)
        self.allocator.free(seq.seq_id)
        seq.cached_prefix_len = 0
        seq.num_computed_tokens = 0

    def _pin_prefix(self, seq: Sequence, num_tokens: int) -> None:
        """Hold a prefix-cache pin of exactly num_tokens for this sequence.

        Releases any pin the sequence already held first, so acquire/release
        stay balanced no matter how many times a sequence is preempted and
        re-prefilled. Unbalanced pins leave cache nodes with permanently
        positive `holders`, making them unevictable and eventually livelocking
        the engine. Found by DST seed 15; see docs/DST.md BUG-005.
        """
        if seq.pinned_prefix_len == num_tokens:
            return
        self._unpin_prefix(seq)
        if num_tokens > 0:
            self.prefix_cache.acquire(seq.request.prompt_token_ids, num_tokens)
            seq.pinned_prefix_len = num_tokens

    def _unpin_prefix(self, seq: Sequence) -> None:
        if seq.pinned_prefix_len > 0:
            self.prefix_cache.release(
                seq.request.prompt_token_ids, seq.pinned_prefix_len
            )
            seq.pinned_prefix_len = 0

    def _allocate_for(self, seq: Sequence) -> bool:
        """Ensure seq has KV capacity for its currently-known token count.

        On failure, reclaim blocks from the prefix cache and retry once. The
        prefix cache is a cache: it must yield to a request that would
        otherwise be unschedulable. Without this the engine livelocks --
        cached blocks from finished sequences are never reclaimed, so a
        request needing more than the free pool waits forever even though the
        memory is right there. Found by DST seed 90; see docs/DST.md BUG-004.
        """
        # Admitting a sequence that would leave no block to grow into just
        # queues up an immediate OOM-preempt cycle (BUG-010), so require one
        # spare block beyond what the sequence needs right now.
        #
        # This uses total_tokens, not prompt_len: a preempted sequence must
        # recompute prompt + everything it already generated, so its
        # requirement grows each time it is preempted.
        #
        # A sequence needing the whole pool can never emit a token -- it fills
        # memory during prefill then OOMs on its first append -- so retire it
        # rather than admitting it every step forever. Checked regardless of
        # whether it already holds a table: a sequence mid-prefill has one and
        # would otherwise skip the growth guard below and cycle indefinitely.
        # DST BUG-015 (seed 2282).
        full_need = self.allocator.blocks_needed_for(seq.total_tokens)
        if full_need + 1 > self.allocator.usable_vram_blocks:
            if seq in self.waiting:
                self.waiting.remove(seq)
            if seq in self.running:
                self.running.remove(seq)
            self._retire(
                seq,
                FinishReason.LENGTH if seq.num_output_tokens else FinishReason.ABORTED,
            )
            return False

        # Reserve the growth block at ADMISSION time against blocks actually
        # available, not against total capacity. Checking capacity only meant
        # two sequences could each be admitted believing they had room to grow,
        # then collide on the first output token and recompute-preempt each
        # other indefinitely -- 68 OOM-preempts of one sequence, each redoing a
        # full 204-token prefill. DST BUG-012 (seeds 261/139/296).
        if self.allocator.block_table(seq.seq_id) is None:
            if not self.allocator.can_allocate(full_need + 1):
                # Try reclaiming before refusing; the cache is expendable.
                deficit = full_need + 1 - self.allocator.free_vram_count
                if deficit > 0:
                    self._reclaim_from_cache(deficit)
                if not self.allocator.can_allocate(full_need + 1):
                    return False

        for attempt in (0, 1):
            try:
                if self.allocator.block_table(seq.seq_id) is None:
                    self.allocator.allocate(seq.seq_id, seq.total_tokens)
                else:
                    have = self.allocator.block_table(seq.seq_id).num_tokens(
                        self.config.block_size
                    )
                    deficit = seq.total_tokens - have
                    if deficit > 0:
                        self.allocator.append_tokens(seq.seq_id, deficit)
                return True
            except AllocError:
                if attempt == 1:
                    return False
                # Reclaim the actual shortfall, not the total requirement:
                # blocks already held by this sequence still count toward it.
                have = 0
                table = self.allocator.block_table(seq.seq_id)
                if table is not None:
                    have = len(table.blocks)
                deficit = (
                    self.allocator.blocks_needed_for(seq.total_tokens)
                    - have
                    - self.allocator.free_vram_count
                )
                if not self._reclaim_from_cache(max(1, deficit)):
                    return False
        return False

    def _drop_prefix_cache(self) -> int:
        """Release every unpinned cached block. Returns blocks reclaimed.

        The nuclear option, used only as a starvation backstop: correctness and
        liveness beat a warm cache. Pinned nodes (belonging to live sequences)
        are still respected.
        """
        if not self.config.enable_prefix_cache:
            return 0
        freed = self.prefix_cache.evict(self.allocator.total_vram_blocks)
        for bid in freed:
            self.allocator.release_cached_block(bid)
        return len(freed)

    def _reclaim_from_cache(self, n_blocks: int) -> bool:
        """Evict unpinned prefix-cache entries and free their blocks.

        Returns True if anything was reclaimed. Only unpinned leaves are
        evictable, so blocks belonging to live sequences are never taken. If
        the cache cannot yield enough, the caller falls back to preemption.
        """
        if not self.config.enable_prefix_cache:
            return False
        freed = self.prefix_cache.evict(n_blocks)
        for bid in freed:
            self.allocator.release_cached_block(bid)
        return bool(freed)

    def _ensure_running_can_progress(self) -> None:
        """Preempt victims until every running sequence can take a step.

        A decoding sequence needs at most one new block per step. If we cannot
        guarantee that for all of them, we are one step from deadlock, which
        is precisely what the watermark reserve exists to detect.
        """
        guard = 0
        while self.running:
            guard += 1
            if guard > self.config.max_num_seqs + 1:
                break  # cannot happen; bounds the loop for DST safety

            need = 0
            for seq in self.running:
                if seq.is_prefill_done:
                    need += self.allocator.blocks_needed_to_append(seq.seq_id, 1)
            if self.allocator.can_allocate(need):
                return

            # Prefer evicting cache over killing work: the cache is
            # reconstructible, an in-flight sequence's progress is not.
            if self._reclaim_from_cache(
                max(1, need - self.allocator.free_vram_count)
            ):
                continue

            victim = self._pick_victim()
            if victim is None:
                return  # nothing preemptible; the step will simply do less
            self._preempt(victim)

    def _pick_victim(self) -> Optional[Sequence]:
        """Deterministic victim choice: lowest priority, then highest seq_id.

        Highest seq_id means newest, so we sacrifice the least-invested work
        and avoid livelocking an old sequence that keeps getting evicted.
        Class A is protected while few are in flight (spec section 6.2).
        """
        class_a_count = sum(1 for s in self.running if s.slo_class is SloClass.A)
        protect_a = class_a_count <= self.config.class_a_protect_limit

        # Never preempt the last running sequence. Preemption exists to free
        # memory for *other* work; with a single sequence there is no other
        # work, so evicting it just destroys progress and re-admits it next
        # step. A sequence needing most of the pool would otherwise recompute
        # forever -- and because recompute must redo prompt + already-generated
        # tokens, the work per attempt grows every round. Observed as 271
        # preemptions of one sequence. DST BUG-009 (seed 140).
        if len(self.running) <= 1:
            return None

        candidates = [
            s
            for s in self.running
            if not (protect_a and s.slo_class is SloClass.A)
        ]
        if not candidates:
            return None
        # Highest class value (lowest priority) first, then highest seq_id.
        return max(candidates, key=lambda s: (int(s.slo_class), s.seq_id))

    def _preempt(self, seq: Sequence) -> None:
        """Recompute by default; swap only when the ratio test says so."""
        self.running.remove(seq)

        if self._should_swap(seq):
            try:
                pairs = self.allocator.swap_out(seq.seq_id)
                self._pending_swap_out.extend(BlockCopy(src=a, dst=b) for a, b in pairs)
                seq.state = SeqState.PREEMPTED
                self.swapped.append(seq)
                self.stats.preemptions_swap += 1
                seq.preempt_count += 1
                return
            except AllocError:
                pass  # host tier full -> fall through to recompute

        self._free_seq_kv(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)  # front of queue: it already waited once
        self.stats.preemptions_recompute += 1

    def _should_swap(self, seq: Sequence) -> bool:
        """Spec section 6.3: swap only if generated_len > 4 * prompt_len."""
        if not self.config.enable_swap:
            return False
        if self.allocator.total_host_blocks == 0:
            return False
        return seq.num_output_tokens > self.config.swap_generated_ratio * seq.prompt_len

    def _maybe_swap_in(self) -> None:
        """Bring swapped sequences back when VRAM allows, oldest first."""
        if not self.swapped:
            return
        for seq in sorted(self.swapped, key=lambda s: s.seq_id):
            table = self.allocator.block_table(seq.seq_id)
            if table is None:
                continue
            try:
                pairs = self.allocator.swap_in(seq.seq_id)
            except AllocError:
                break
            self._pending_swap_in.extend(BlockCopy(src=a, dst=b) for a, b in pairs)
            self.swapped.remove(seq)
            seq.state = SeqState.DECODE
            self.running.append(seq)

    def _build_exec_step(self) -> ExecStep:
        """Compose this iteration's forward pass.

        Prefills are packed first (they carry the token budget), then decodes
        fill the remainder. Mixing them in one step is chunked prefill: a long
        prompt advances a chunk at a time while decodes keep flowing.
        """
        step = ExecStep(step_id=self.step_t)
        budget = self.config.max_num_batched_tokens

        decode_seqs = [s for s in self.running if s.is_prefill_done]
        prefill_seqs = [s for s in self.running if not s.is_prefill_done]

        # Reserve one token of budget per decode so prefill cannot starve them.
        reserved = min(len(decode_seqs), budget)
        prefill_budget = max(0, budget - reserved)

        for seq in prefill_seqs:
            if prefill_budget <= 0:
                break
            remaining = seq.remaining_prompt_tokens
            chunk_len = min(remaining, prefill_budget)
            if not self.config.enable_chunked_prefill:
                # An unsplittable prompt must run whole. If it exceeds the
                # budget it can only go in an otherwise-empty step; admission
                # guarantees it was admitted alone, so honour that here rather
                # than skipping it (which would strand it forever). BUG-001.
                chunk_len = remaining
                if chunk_len > prefill_budget and (step.prefills or decode_seqs):
                    continue

            start = seq.num_computed_tokens
            token_ids = seq.request.prompt_token_ids[start : start + chunk_len]
            table = self.allocator.block_table(seq.seq_id)
            step.prefills.append(
                PrefillItem(
                    seq_id=seq.seq_id,
                    token_ids=token_ids,
                    start_pos=start,
                    block_ids=list(table.blocks) if table else [],
                    params=seq.request.params,
                    is_last_chunk=(start + chunk_len >= seq.prompt_len),
                )
            )
            prefill_budget -= chunk_len

        gamma = self._effective_gamma(len(decode_seqs))
        step.spec_gamma = gamma

        for seq in decode_seqs:
            table = self.allocator.block_table(seq.seq_id)
            if table is None:
                continue
            all_tokens = seq.all_token_ids()
            step.decodes.append(
                DecodeItem(
                    seq_id=seq.seq_id,
                    last_token_id=all_tokens[-1],
                    position=len(all_tokens) - 1,
                    block_ids=list(table.blocks),
                    params=seq.request.params,
                    context_len=len(all_tokens),
                )
            )

        step.block_copies = [
            BlockCopy(src=a, dst=b)
            for a, b in self.allocator.drain_pending_copies()
        ]
        step.swap_in, self._pending_swap_in = self._pending_swap_in, []
        step.swap_out, self._pending_swap_out = self._pending_swap_out, []
        return step

    def _effective_gamma(self, batch_size: int) -> int:
        """Adaptive speculation gate (spec section 7.3).

        Disable when the batch is large (verify becomes compute-bound and the
        wasted draft compute is no longer free) or when measured acceptance
        has collapsed. "We always speculate" is the wrong answer.
        """
        if not self.config.enable_spec_decode or batch_size == 0:
            return 0
        if batch_size > self.config.spec_max_batch_size:
            return 0
        if len(self._accept_window) >= 8:
            drafted = sum(d for d, _ in self._accept_window)
            accepted = sum(a for _, a in self._accept_window)
            if drafted and accepted / drafted < self.config.spec_min_acceptance:
                return 0
        return self.config.spec_gamma

    def _apply_outputs(self, step: ExecStep, outputs: ExecOutputs) -> None:
        """Fold executor results back into scheduler state."""
        by_seq = outputs.by_seq()

        for item in step.prefills:
            seq = self.seqs[item.seq_id]
            seq.num_computed_tokens += item.num_tokens
            self.stats.prefill_tokens += item.num_tokens

            if item.is_last_chunk:
                out = by_seq.get(item.seq_id)
                if out and out.token_ids:
                    self._accept_tokens(seq, out.token_ids)
                self._pending_cache_insert.append(item.seq_id)
                if not seq.is_finished:
                    seq.state = SeqState.DECODE

        for item in step.decodes:
            seq = self.seqs[item.seq_id]
            out = by_seq.get(item.seq_id)
            if out is None or not out.token_ids:
                continue
            self._accept_tokens(seq, out.token_ids)
            self.stats.decode_tokens += len(out.token_ids)

            if out.num_drafted:
                self.stats.spec_drafted += out.num_drafted
                self.stats.spec_accepted += out.num_accepted
                self._accept_window.append((out.num_drafted, out.num_accepted))

        self._flush_cache_inserts()

    def _accept_tokens(self, seq: Sequence, token_ids: List[int]) -> None:
        """Commit generated tokens, growing KV and applying stop conditions.

        More than one token arrives only under speculation, where the accepted
        prefix plus the bonus token all commit in one step.
        """
        params = seq.request.params
        for token_id in token_ids:
            if seq.is_finished:
                break
            try:
                self.allocator.append_tokens(seq.seq_id, 1)
            except AllocError:
                # No room to grow. Try to reclaim cache; failing that, preempt
                # rather than corrupt -- the token is dropped and regenerated
                # after recompute.
                if self._reclaim_from_cache(1):
                    try:
                        self.allocator.append_tokens(seq.seq_id, 1)
                    except AllocError:
                        self._preempt_for_oom(seq)
                        return
                else:
                    self._preempt_for_oom(seq)
                    return

            seq.append_token(token_id, self.step_t)

            if (
                token_id in params.stop_token_ids
                and not seq.is_finished
            ):
                self._finish(seq, FinishReason.EOS)
            elif seq.num_output_tokens >= params.max_tokens:
                self._finish(seq, FinishReason.LENGTH)
            elif seq.total_tokens >= self.config.max_model_len:
                self._finish(seq, FinishReason.LENGTH)

    def _preempt_for_oom(self, seq: Sequence) -> None:
        """Roll a sequence back to recompute after an out-of-memory append.

        If this is the only sequence in flight, recompute cannot help: there is
        no other work whose completion would free memory, so the sequence would
        re-prefill and OOM again forever. Reclaim the cache if we can; if even
        that leaves no room, the sequence has outgrown the KV pool and is
        finished at its current length rather than retried indefinitely. That
        is a truthful outcome (the client sees finish_reason="length") instead
        of a livelock. DST BUG-010.
        """
        if len(self.running) <= 1 and not self.waiting:
            if seq in self.running:
                self.running.remove(seq)
            # Free the KV and publish to `finished` explicitly. Removing a
            # sequence from `running` and marking it finished is NOT enough:
            # _reap_finished only walks `running`, so a sequence finished
            # outside it becomes an orphan whose blocks are never reclaimed --
            # here, a leak of the entire pool. DST BUG-013 (seed 2452).
            self._retire(seq, FinishReason.LENGTH)
            return

        if seq in self.running:
            self.running.remove(seq)
        self._free_seq_kv(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)
        self.stats.preemptions_recompute += 1

    def _finish(self, seq: Sequence, reason: FinishReason) -> None:
        """Mark a sequence complete. It stays in `running` for _reap_finished.

        Use this only for sequences that ARE in `running`; anything finished
        out-of-band must go through _retire, or its KV is never reclaimed.
        """
        seq.state = SeqState.FINISHED
        seq.finish_reason = reason

    def _retire(self, seq: Sequence, reason: FinishReason) -> None:
        """Finish a sequence that is not in `running`, freeing its KV now.

        _reap_finished only walks the running list, so a sequence completed from
        any other queue would otherwise become an orphan: finished, in no queue,
        and holding its blocks forever. See BUG-013.
        """
        self._finish(seq, reason)
        self._free_seq_kv(seq)
        self.finished.append(seq)
        self.stats.finished_seqs += 1

    def _flush_cache_inserts(self) -> None:
        """Publish completed prompts into the prefix cache.

        Done after prefill completes, when the prompt's blocks are stable and
        fully filled. Only block-aligned whole blocks are inserted.
        """
        if not self.config.enable_prefix_cache:
            self._pending_cache_insert.clear()
            return
        for seq_id in self._pending_cache_insert:
            seq = self.seqs.get(seq_id)
            table = self.allocator.block_table(seq_id)
            if seq is None or table is None:
                continue
            n_full = seq.prompt_len // self.config.block_size
            if n_full == 0:
                continue
            added = self.prefix_cache.insert(
                seq.request.prompt_token_ids,
                table.blocks[:n_full],
                self.step_t,
            )
            # Take a cache-owned reference on newly cached blocks so they
            # survive this sequence being freed. `insert` reports exactly the
            # blocks it took ownership of, so we never double-retain one that
            # was already cached by an earlier sequence.
            for bid in added:
                self.allocator.retain_cached_block(bid)
            # Deliberately NOT pinned here. The allocator's cache reference
            # (retain_cached_block above) already keeps these blocks alive, and
            # the sequence's own block table keeps the ones it is using. Pinning
            # on publish additionally made the nodes unevictable for as long as
            # the publisher lived, so a large cached prompt could hold most of
            # the pool hostage while queued sequences starved -- eviction found
            # no victims precisely when memory was scarcest. DST BUG-011
            # (seeds 2/78/140). Pins are for prefix *reuse* only (_attach).
            seq.cached_prefix_len = max(
                seq.cached_prefix_len, n_full * self.config.block_size
            )
        self._pending_cache_insert.clear()

    def _free_seq_kv(self, seq: Sequence) -> None:
        """Release a sequence's KV and any prefix-cache pin it holds."""
        self._unpin_prefix(seq)
        seq.cached_prefix_len = 0
        self.allocator.free(seq.seq_id)

    def _handle_fault(self, fault: ExecFault) -> None:
        """Recover from an executor fault by preempting the affected sequences.

        Correctness over throughput: any sequence that might have partially
        written KV is rolled back to recompute rather than trusted.
        """
        targets = fault.seq_ids or [s.seq_id for s in list(self.running)]
        for seq_id in targets:
            seq = self.seqs.get(seq_id)
            if seq is None or seq.is_finished:
                continue
            if seq in self.running:
                self.running.remove(seq)
            self._free_seq_kv(seq)
            seq.reset_for_recompute()
            self.waiting.appendleft(seq)
            self.stats.preemptions_recompute += 1

    # ------------------------------------------------------------ inspection

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    def num_running(self) -> int:
        return len(self.running)

    def num_waiting(self) -> int:
        return len(self.waiting)

    def get_sequence(self, seq_id: SeqId) -> Optional[Sequence]:
        return self.seqs.get(seq_id)

    def take_finished(self) -> List[Sequence]:
        """Drain completed sequences for the frontend to stream out."""
        out, self.finished = self.finished, []
        return out

    def check_invariants(self) -> None:
        """Allocator I1..I7 plus scheduler-level structural invariants."""
        self.allocator.check_invariants()

        running_ids = [s.seq_id for s in self.running]
        if len(set(running_ids)) != len(running_ids):
            raise AssertionError("duplicate sequence in running list")

        waiting_ids = {s.seq_id for s in self.waiting}
        overlap = waiting_ids & set(running_ids)
        if overlap:
            raise AssertionError(f"sequences both waiting and running: {overlap}")

        swapped_ids = {s.seq_id for s in self.swapped}
        if swapped_ids & set(running_ids):
            raise AssertionError("sequence both swapped and running")

        for seq in self.running:
            if self.allocator.block_table(seq.seq_id) is None:
                raise AssertionError(f"running seq {seq.seq_id} has no block table")

        # A waiting sequence owns nothing: no KV, no prefix-cache pin. A pin
        # held by a queued sequence makes cache nodes unevictable and livelocks
        # admission, so this is checked directly rather than left to surface as
        # a liveness timeout thousands of steps later. See docs/DST.md BUG-007.
        for seq in self.waiting:
            if seq.pinned_prefix_len:
                raise AssertionError(
                    f"waiting seq {seq.seq_id} holds a prefix-cache pin of "
                    f"{seq.pinned_prefix_len} tokens"
                )
            if self.allocator.block_table(seq.seq_id) is not None:
                raise AssertionError(
                    f"waiting seq {seq.seq_id} still holds a block table"
                )

        # A finished sequence must hold no KV -- this is the leak check.
        for seq in self.finished:
            if self.allocator.block_table(seq.seq_id) is not None:
                raise AssertionError(f"finished seq {seq.seq_id} still holds blocks")
            if seq.pinned_prefix_len:
                raise AssertionError(
                    f"finished seq {seq.seq_id} still pins {seq.pinned_prefix_len} "
                    f"prefix tokens"
                )

        # Every sequence must be reachable from exactly one queue, or be
        # terminal and already reaped. An unreachable sequence holding blocks is
        # a permanent leak that no reaper will ever visit (BUG-013).
        queued = (
            {s.seq_id for s in self.running}
            | {s.seq_id for s in self.waiting}
            | {s.seq_id for s in self.swapped}
            | {s.seq_id for s in self.finished}
        )
        for seq_id, seq in self.seqs.items():
            if seq_id in queued:
                continue
            if not seq.is_finished:
                raise AssertionError(
                    f"seq {seq_id} (state={seq.state.value}) is in no queue but "
                    f"is not finished"
                )
            if self.allocator.block_table(seq_id) is not None:
                raise AssertionError(
                    f"orphaned seq {seq_id} is finished, in no queue, and still "
                    f"holds KV blocks"
                )

        # Prefix-cache pins must be attributable to a live sequence. A pin whose
        # owner is gone makes its cache nodes permanently unevictable, which
        # starves admission -- and it surfaces 4000 steps later as an opaque
        # liveness timeout unless caught here, at the step that caused it.
        live_pins = sum(
            s.pinned_prefix_len > 0
            for s in list(self.running) + list(self.waiting) + list(self.swapped)
        )
        held = self._count_pinned_nodes()
        if held and not live_pins:
            raise AssertionError(
                f"{held} prefix-cache nodes are pinned but no live sequence "
                f"holds a pin (unevictable cache -> admission starvation)"
            )

    def _count_pinned_nodes(self) -> int:
        """Number of prefix-cache nodes with a positive holder count."""
        count = 0
        stack = [self.prefix_cache.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node is not self.prefix_cache.root and node.holders > 0:
                count += 1
        return count

    def snapshot(self) -> Dict[str, object]:
        """Deterministic state digest, used by the replay-equality test."""
        return {
            "step": self.step_t,
            "running": sorted(
                (s.seq_id, s.num_computed_tokens, s.num_output_tokens)
                for s in self.running
            ),
            "waiting": [s.seq_id for s in self.waiting],
            "swapped": sorted(s.seq_id for s in self.swapped),
            "free_blocks": self.allocator.free_vram_count,
            "prefill_tokens": self.stats.prefill_tokens,
            "decode_tokens": self.stats.decode_tokens,
            "finished": self.stats.finished_seqs,
            "preempt_recompute": self.stats.preemptions_recompute,
            "preempt_swap": self.stats.preemptions_swap,
        }

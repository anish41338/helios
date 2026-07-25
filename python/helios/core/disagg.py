"""Prefill/decode disaggregation and the KV transfer FSM (spec section 6.4).

Why disaggregate
----------------
Prefill and decode have opposite bottlenecks. A prefill step has many query
tokens attending over a short context: arithmetic intensity is high, it saturates
compute, and it takes as long as it takes. A decode step has one query token per
sequence attending over the whole context: it reads the entire KV cache to do a
handful of FLOPs, so it is bound by memory bandwidth and its latency is set by
how many bytes must move.

Interleaving them on one device makes each worse. A long prefill chunk sitting in
front of a decode batch adds its full duration to every resident sequence's
inter-token latency -- one 2000-token prompt can stall 30 sequences. That is the
head-of-line blocking chunked prefill exists to bound, and bounding is not
eliminating: the chunk still costs what it costs.

Disaggregation removes the contention instead of bounding it. Prefill runs on one
partition, decode on another, and the prompt's KV cache is transferred between
them. TTFT is then governed by prefill capacity and TPOT by decode capacity, and
neither can inflate the other.

What it costs, and where the design actually breaks
--------------------------------------------------
The transfer. A 2000-token prompt on a 32-layer model with 8 KV heads at
head_dim 128 in fp16 is

    2 (K and V) * 2000 * 32 * 8 * 128 * 2 bytes = 262 MB

and re-prefilling that prompt at 8000 tok/s costs 250 ms. So:

    NVLink   300 GB/s ->   0.9 ms     transfer wins by ~270x
    PCIe 3.0  12 GB/s ->  21.8 ms     transfer wins by ~11x
    10 GbE   1.25 GB/s -> 209.7 ms    roughly break-even
    (and at 20000 tok/s prefill, 10 GbE loses outright)

The conclusion that falls out: **within a node, transferring is clearly right;
across a commodity network it is not.** The crossover is at the machine boundary,
not inside it. That is worth stating precisely because a design that always
transferred would be strictly worse than no disaggregation at all on the wrong
side of that line -- and `transfer_vs_reprefill` below exists so the choice is
computed from the hardware rather than assumed.

(The first version of this docstring put the KV at 2.6 GB and the PCIe transfer
at 220 ms -- a factor of ten out, which inverted the PCIe verdict from "wins by
11x" to "roughly break-even". It was caught by
tests/disagg/test_disagg.py::test_reprefill_beats_transfer_on_a_slow_interconnect
failing against the real formula. Worth leaving a note about: an arithmetic slip
in a comment is still a claim without a reproduction behind it, which is exactly
what spec section 19.3 is about.)

And a transfer is a distributed operation with partial-failure modes that a
single-device engine does not have:

  * the sender finishes and the receiver never acknowledges;
  * the receiver runs out of blocks mid-transfer;
  * bytes arrive corrupted or truncated;
  * the request is aborted while in flight.

Each one can leak blocks on one side, on the other, or on both. That is exactly
the class of bug the DST harness was built to find (spec section 10), so this is
a state machine over explicit states with an invariant checker, not a sequence of
awaits. `vopr.py` drives it under fault injection.

Ownership is the load-bearing invariant: at every instant, each transfer's blocks
are owned by exactly one partition, or deliberately by both during the window
where the sender must not release yet. Never by neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Role(str, Enum):
    """What a partition is for."""

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"      # the co-located configuration, i.e. no disaggregation


class TransferState(str, Enum):
    """Lifecycle of one sequence's KV migration.

    Deliberately explicit about the difference between SENT (the sender believes
    it is done) and RECEIVED (the receiver confirms it): the gap between those
    two is where every interesting fault lives. Collapsing them -- which an
    async/await formulation does by default -- makes the "sender finished,
    receiver never acknowledged" case unrepresentable, and therefore untestable.
    """

    PENDING = "pending"        # queued, receiver blocks not yet reserved
    RESERVED = "reserved"      # receiver has committed destination blocks
    SENDING = "sending"        # bytes in flight
    SENT = "sent"              # sender done, awaiting the receiver's ack
    RECEIVED = "received"      # receiver has the bytes and verified them
    COMMITTED = "committed"    # sender released its blocks; migration complete
    FAILED = "failed"          # terminal: blocks reclaimed on both sides
    ABORTED = "aborted"        # terminal: cancelled by the scheduler


class TransferFault(str, Enum):
    """The ways a transfer can go wrong. Injected by the DST harness."""

    RECEIVER_OOM = "receiver_oom"           # no destination blocks available
    LINK_TIMEOUT = "link_timeout"           # in flight too long
    CHECKSUM_MISMATCH = "checksum_mismatch"  # bytes arrived corrupted
    ACK_LOST = "ack_lost"                   # receiver has it, sender never learns
    ABORT = "abort"                         # client cancelled mid-flight


# Terminal states: no transition leaves these.
_TERMINAL = {TransferState.COMMITTED, TransferState.FAILED, TransferState.ABORTED}

# The only legal transitions. Anything else is a bug in the caller, and
# `KVTransferManager` raises rather than tolerating it -- an FSM that silently
# accepts an illegal edge is not a specification of anything.
_LEGAL: Dict[TransferState, Tuple[TransferState, ...]] = {
    TransferState.PENDING: (
        TransferState.RESERVED, TransferState.FAILED, TransferState.ABORTED,
    ),
    TransferState.RESERVED: (
        TransferState.SENDING, TransferState.FAILED, TransferState.ABORTED,
    ),
    TransferState.SENDING: (
        TransferState.SENT, TransferState.FAILED, TransferState.ABORTED,
    ),
    # No ABORTED from SENT: once the bytes are on the wire, a client abort has to
    # resolve the transfer one way or the other before blocks can be released, or
    # the receiver is left holding reserved blocks that nothing will ever claim.
    # This is the difference between cancelling a request and cancelling a
    # transfer -- the scheduler may do the former at any time, and it becomes the
    # latter only at a point where ownership is unambiguous.
    TransferState.SENT: (
        TransferState.RECEIVED, TransferState.FAILED,
    ),
    TransferState.RECEIVED: (
        TransferState.COMMITTED, TransferState.FAILED,
    ),
}


@dataclass
class KVTransfer:
    """One sequence's KV cache migrating from a prefill to a decode partition."""

    transfer_id: int
    seq_id: int
    n_blocks: int
    src_blocks: List[int]
    dst_blocks: List[int] = field(default_factory=list)
    state: TransferState = TransferState.PENDING
    # Step at which each phase happened, for latency accounting. The scheduler is
    # clock-free (spec section 19.4), so these are step counts, not seconds.
    created_step: int = 0
    started_step: Optional[int] = None
    completed_step: Optional[int] = None
    bytes_total: int = 0
    bytes_sent: int = 0
    fault: Optional[TransferFault] = None
    retries: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL

    @property
    def is_in_flight(self) -> bool:
        return self.state in (TransferState.SENDING, TransferState.SENT)

    @property
    def sender_holds_blocks(self) -> bool:
        """The prefill partition cannot release until the receiver confirms.

        Releasing at SENT would be wrong: if the ack is lost and the transfer has
        to be retried or failed, the only copy of the KV would already be gone
        and the sequence would need a full re-prefill. Holding through RECEIVED
        is what makes a lost ack recoverable rather than fatal.

        False once terminal, because reaching a terminal state is precisely the
        point at which `commit`/`fail`/`abort` handed the blocks back.
        """
        return not self.is_terminal

    @property
    def receiver_holds_blocks(self) -> bool:
        return self.state in (
            TransferState.RESERVED,
            TransferState.SENDING,
            TransferState.SENT,
            TransferState.RECEIVED,
            TransferState.COMMITTED,
        )


class TransferInvariantViolation(AssertionError):
    """A KV transfer invariant broke. Always a bug in this module."""


@dataclass
class DisaggConfig:
    """Sizing and policy for a two-partition deployment."""

    # Effective interconnect bandwidth, bytes per second. The default is PCIe
    # 3.0 x16 measured, not theoretical (16 GB/s theoretical, ~12 GB/s real).
    link_bandwidth_bps: float = 12e9
    # Transfers the link will carry at once. More than a few does not help --
    # the link is the bottleneck, not the request count -- and each in-flight
    # transfer pins blocks on both sides.
    max_concurrent_transfers: int = 4
    # Steps a transfer may stay in flight before the link is declared dead.
    transfer_timeout_steps: int = 64
    # A failed transfer is retried this many times before the sequence falls back
    # to re-prefilling on the decode partition. Re-prefill is always available as
    # a fallback because the prompt tokens are still known -- the same reason
    # recompute is a valid preemption strategy (spec section 6.3).
    max_retries: int = 2


class KVTransferManager:
    """Tracks every in-flight KV migration between two partitions.

    Pure: no clock, no IO, no RNG. Progress happens only when `step()` is called,
    and faults arrive only because a caller injected them. That is what lets the
    DST harness explore the state space deterministically and replay any failure
    from its seed (spec section 19.4).
    """

    def __init__(self, config: Optional[DisaggConfig] = None) -> None:
        self.config = config or DisaggConfig()
        self.transfers: Dict[int, KVTransfer] = {}
        self._next_id = 0
        self.step_count = 0

        # Counters for metrics and for test assertions. `failed` counts fail()
        # *events*, some of which were subsequently retried -- it is not the
        # number of sequences that lost their KV. `fell_back_to_reprefill` is
        # that number.
        self.completed = 0
        self.failed = 0
        self.aborted = 0
        self.retried = 0
        self.fell_back_to_reprefill = 0
        self.bytes_moved = 0
        self.total_transfer_steps = 0

    # ---------------------------------------------------------------- submit

    def submit(
        self,
        seq_id: int,
        src_blocks: List[int],
        bytes_per_block: int,
    ) -> KVTransfer:
        """Queue a migration. Does not reserve anything yet."""
        self._next_id += 1
        t = KVTransfer(
            transfer_id=self._next_id,
            seq_id=seq_id,
            n_blocks=len(src_blocks),
            src_blocks=list(src_blocks),
            created_step=self.step_count,
            bytes_total=len(src_blocks) * bytes_per_block,
        )
        self.transfers[t.transfer_id] = t
        return t

    # ----------------------------------------------------------- transitions

    def _transition(self, t: KVTransfer, to: TransferState) -> None:
        legal = _LEGAL.get(t.state, ())
        if to not in legal:
            raise TransferInvariantViolation(
                f"illegal transfer transition {t.state.value} -> {to.value} "
                f"for transfer {t.transfer_id} (legal: {[s.value for s in legal]})"
            )
        t.state = to

    def reserve(self, t: KVTransfer, dst_blocks: List[int]) -> None:
        """The receiver commits destination blocks."""
        if len(dst_blocks) != t.n_blocks:
            raise TransferInvariantViolation(
                f"transfer {t.transfer_id} needs {t.n_blocks} destination blocks, "
                f"got {len(dst_blocks)}"
            )
        t.dst_blocks = list(dst_blocks)
        self._transition(t, TransferState.RESERVED)

    def begin(self, t: KVTransfer) -> None:
        if self.n_in_flight >= self.config.max_concurrent_transfers:
            raise TransferInvariantViolation(
                f"link is saturated ({self.n_in_flight} in flight); the caller "
                "must check can_begin() first"
            )
        t.started_step = self.step_count
        self._transition(t, TransferState.SENDING)

    def can_begin(self) -> bool:
        return self.n_in_flight < self.config.max_concurrent_transfers

    def acknowledge(self, t: KVTransfer) -> None:
        """The receiver confirms it has verified bytes."""
        self._transition(t, TransferState.RECEIVED)

    def commit(self, t: KVTransfer) -> List[int]:
        """Finish the migration. Returns the source blocks, now free to reuse."""
        self._transition(t, TransferState.COMMITTED)
        t.completed_step = self.step_count
        self.completed += 1
        self.bytes_moved += t.bytes_total
        if t.started_step is not None:
            self.total_transfer_steps += self.step_count - t.started_step
        return list(t.src_blocks)

    def fail(self, t: KVTransfer, fault: TransferFault) -> Tuple[List[int], List[int]]:
        """Terminate a transfer. Returns (src_blocks, dst_blocks) to reclaim.

        Both sides are returned because a failed transfer leaks on both if only
        one is handled. That asymmetry -- the receiver reserved blocks the sender
        does not know about -- is the whole reason this returns a pair.
        """
        t.fault = fault
        self._transition(t, TransferState.FAILED)
        t.completed_step = self.step_count
        self.failed += 1
        return list(t.src_blocks), list(t.dst_blocks)

    def abort(self, t: KVTransfer) -> Tuple[List[int], List[int]]:
        """Cancel before the bytes are irrevocably in flight.

        A request cancelled after SENT is failed rather than aborted, because at
        that point the receiver may already hold valid data and ownership must be
        resolved before anything is released.
        """
        if t.state in (TransferState.SENT, TransferState.RECEIVED):
            return self.fail(t, TransferFault.ABORT)
        t.fault = TransferFault.ABORT
        self._transition(t, TransferState.ABORTED)
        t.completed_step = self.step_count
        self.aborted += 1
        return list(t.src_blocks), list(t.dst_blocks)

    def retry(self, t: KVTransfer) -> bool:
        """Requeue a failed transfer. False if it is out of retries.

        A False here means the sequence must be re-prefilled on the decode
        partition instead. That fallback always exists because the prompt tokens
        are still known -- the same argument that makes recompute a valid
        preemption strategy (spec section 6.3).
        """
        if t.retries >= self.config.max_retries:
            self.fell_back_to_reprefill += 1
            return False
        t.retries += 1
        self.retried += 1
        t.state = TransferState.PENDING     # a deliberate reset, not an edge
        t.fault = None
        t.dst_blocks = []
        t.bytes_sent = 0
        t.started_step = None
        t.completed_step = None
        return True

    # -------------------------------------------------------------- stepping

    def step(self, bandwidth_bytes_per_step: Optional[float] = None) -> List[KVTransfer]:
        """Advance every in-flight transfer by one step.

        Returns transfers that reached SENT this step -- the caller must decide
        whether each is acknowledged (normal) or has its ack lost (a fault).

        Bandwidth is shared equally among in-flight transfers, which is what a
        saturated link actually does. Concurrency does not add throughput here,
        and modelling it as if it did would make disaggregation look better than
        it is.
        """
        self.step_count += 1
        sending = [t for t in self.transfers.values() if t.state == TransferState.SENDING]
        if not sending:
            return []

        if bandwidth_bytes_per_step is None:
            bandwidth_bytes_per_step = self.config.link_bandwidth_bps / 1000.0
        share = bandwidth_bytes_per_step / len(sending)

        newly_sent: List[KVTransfer] = []
        for t in sending:
            t.bytes_sent = min(t.bytes_total, t.bytes_sent + share)
            if t.bytes_sent >= t.bytes_total:
                self._transition(t, TransferState.SENT)
                newly_sent.append(t)
        return newly_sent

    def timed_out(self) -> List[KVTransfer]:
        """In-flight transfers that have exceeded the timeout."""
        limit = self.config.transfer_timeout_steps
        return [
            t
            for t in self.transfers.values()
            if t.is_in_flight
            and t.started_step is not None
            and self.step_count - t.started_step > limit
        ]

    # ------------------------------------------------------------ inspection

    @property
    def n_in_flight(self) -> int:
        return sum(1 for t in self.transfers.values() if t.is_in_flight)

    @property
    def n_pending(self) -> int:
        return sum(1 for t in self.transfers.values() if t.state == TransferState.PENDING)

    @property
    def n_active(self) -> int:
        return sum(1 for t in self.transfers.values() if not t.is_terminal)

    def mean_transfer_steps(self) -> float:
        return self.total_transfer_steps / self.completed if self.completed else 0.0

    def gc(self) -> int:
        """Drop terminal transfers. Returns how many were removed.

        Without this the dict grows without bound over a long run, which in DST
        shows up as the harness slowing down rather than as a failure -- the kind
        of leak a fixed-length test never sees.
        """
        dead = [tid for tid, t in self.transfers.items() if t.is_terminal]
        for tid in dead:
            del self.transfers[tid]
        return len(dead)

    def check_invariants(self) -> None:
        """Assert the ownership and accounting rules. Called every DST step.

        D1. Every transfer's state is reachable and its block lists are
            consistent with it.
        D2. A reserved transfer has exactly n_blocks destination blocks.
        D3. No physical destination block is claimed by two live transfers.
        D4. Bytes sent never exceeds bytes total.
        D5. An in-flight transfer has a start step (needed for timeout detection;
            without it a transfer can hang forever and never be reaped).
        D6. Concurrency never exceeds the configured limit.
        """
        seen_dst: Dict[int, int] = {}
        for t in self.transfers.values():
            # D1/D2
            if t.state in (
                TransferState.RESERVED, TransferState.SENDING,
                TransferState.SENT, TransferState.RECEIVED,
            ):
                if len(t.dst_blocks) != t.n_blocks:
                    raise TransferInvariantViolation(
                        f"D2: transfer {t.transfer_id} in {t.state.value} has "
                        f"{len(t.dst_blocks)} dst blocks, expected {t.n_blocks}"
                    )
            if len(t.src_blocks) != t.n_blocks:
                raise TransferInvariantViolation(
                    f"D1: transfer {t.transfer_id} has {len(t.src_blocks)} src "
                    f"blocks, expected {t.n_blocks}"
                )
            # D3
            if not t.is_terminal:
                for b in t.dst_blocks:
                    if b in seen_dst:
                        raise TransferInvariantViolation(
                            f"D3: destination block {b} is claimed by both "
                            f"transfer {seen_dst[b]} and {t.transfer_id}"
                        )
                    seen_dst[b] = t.transfer_id
            # D4
            if t.bytes_sent > t.bytes_total:
                raise TransferInvariantViolation(
                    f"D4: transfer {t.transfer_id} sent {t.bytes_sent} of "
                    f"{t.bytes_total} bytes"
                )
            # D5
            if t.is_in_flight and t.started_step is None:
                raise TransferInvariantViolation(
                    f"D5: transfer {t.transfer_id} is in flight with no start "
                    "step, so it can never time out"
                )
        # D6
        if self.n_in_flight > self.config.max_concurrent_transfers:
            raise TransferInvariantViolation(
                f"D6: {self.n_in_flight} transfers in flight, limit is "
                f"{self.config.max_concurrent_transfers}"
            )

    def snapshot(self) -> Dict[str, object]:
        return {
            "step": self.step_count,
            "active": self.n_active,
            "in_flight": self.n_in_flight,
            "pending": self.n_pending,
            "completed": self.completed,
            "failed": self.failed,
            "aborted": self.aborted,
            "retried": self.retried,
            "reprefill_fallbacks": self.fell_back_to_reprefill,
            "gb_moved": round(self.bytes_moved / 1e9, 3),
            "mean_transfer_steps": round(self.mean_transfer_steps(), 2),
        }


def transfer_vs_reprefill(
    n_prompt_tokens: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    link_bandwidth_bps: float,
    prefill_tokens_per_s: float,
) -> Dict[str, float]:
    """Is transferring this prompt's KV cheaper than recomputing it?

    The decision disaggregation actually turns on, and the reason this module
    models bandwidth instead of assuming the link is free. Transfer cost is
    linear in prompt length; re-prefill cost is *also* roughly linear in tokens
    at fixed throughput, so the comparison is a ratio of two constants -- which
    means for a given deployment one of the two always wins, and which one is a
    property of the hardware, not of the request.

    That is a useful thing to be able to state: if the interconnect is NVLink the
    transfer wins by a wide margin, and if it is PCIe with a fast GPU it can lose
    outright. A design that assumed transfer is always right would be wrong on
    half the plausible hardware.
    """
    kv_bytes = 2 * n_prompt_tokens * n_layers * n_kv_heads * head_dim * dtype_bytes
    transfer_s = kv_bytes / link_bandwidth_bps
    reprefill_s = n_prompt_tokens / prefill_tokens_per_s
    return {
        "kv_bytes": float(kv_bytes),
        "kv_mib": kv_bytes / 2**20,
        "transfer_s": transfer_s,
        "reprefill_s": reprefill_s,
        "transfer_is_cheaper": transfer_s < reprefill_s,
        "ratio": transfer_s / reprefill_s if reprefill_s else float("inf"),
    }

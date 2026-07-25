"""helios-vopr: the deterministic simulation testing harness.

Implements HELIOS-SPEC.md section 10. Name borrowed from TigerBeetle's VOPR.

The idea, from the FoundationDB / TigerBeetle tradition: the scheduler is a
pure state machine, so we can drive it single-threaded with a simulated clock,
a fabricated executor, and an adversarial fault schedule -- then assert every
invariant after every step. A failing seed is a complete, exact reproduction:

    python -m helios.cli vopr --seed 918273 --replay

Every source of nondeterminism is injected: the clock is an integer counter,
all randomness comes from one seeded random.Random, and the scheduler itself
makes no random choices. That is what makes replay exact rather than
approximate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .allocator import AllocError, Allocator, InvariantViolation
from .disagg import TransferInvariantViolation
from .execstep import (
    ExecFault,
    ExecOutputs,
    ExecStep,
    FaultKind,
    SeqOutput,
)
from .prefix_cache import PrefixCache
from .scheduler import Scheduler, SchedulerConfig
from .types import Request, SamplingParams, SeqId, SloClass

# A reserved token id the simulated model emits to end a sequence.
SIM_EOS = 2


class SimClock:
    """Simulated monotonic clock in nanoseconds.

    The scheduler never reads this -- it exists so the harness can model
    executor latency and so metrics have a time axis under simulation.
    """

    def __init__(self) -> None:
        self.now_ns = 0

    def advance(self, ns: int) -> None:
        if ns < 0:
            raise ValueError("clock cannot go backwards")
        self.now_ns += ns

    @property
    def now_s(self) -> float:
        return self.now_ns / 1e9


@dataclass
class LatencyModel:
    """Crude roofline-flavoured cost model for the simulated executor.

    Prefill is compute-bound so it scales with token count; decode is
    bandwidth-bound so it scales with batch size but is nearly flat in
    context length. These constants are invented for simulation and carry NO
    claim about real hardware (spec section 19.6).
    """

    prefill_ns_per_token: int = 40_000
    decode_ns_per_seq: int = 900_000
    decode_base_ns: int = 4_000_000
    block_copy_ns: int = 120_000

    def step_ns(self, step: ExecStep) -> int:
        ns = 0
        if step.prefills:
            ns += step.num_prefill_tokens * self.prefill_ns_per_token
        if step.decodes:
            ns += self.decode_base_ns + len(step.decodes) * self.decode_ns_per_seq
        ns += (
            len(step.block_copies) + len(step.swap_in) + len(step.swap_out)
        ) * self.block_copy_ns
        return ns


@dataclass
class FaultSchedule:
    """When and what to break.

    Faults are keyed by step index so the schedule is fully determined by the
    seed and can be replayed without re-deriving it.
    """

    at_step: Dict[int, FaultKind] = field(default_factory=dict)
    cancel_at_step: Dict[int, List[int]] = field(default_factory=dict)
    oom_steps: Set[int] = field(default_factory=set)
    # Acceptance rate forced by the adversary; None means use the default.
    forced_acceptance: Optional[float] = None

    def fault_for(self, step: int) -> Optional[FaultKind]:
        return self.at_step.get(step)


class SimExecutor:
    """Fabricated executor with modelled latency and injectable faults.

    Generates deterministic pseudo-tokens: a sequence ends when it has emitted
    the output length the workload generator assigned it, which is how the
    harness can assert termination.
    """

    def __init__(
        self,
        rng: random.Random,
        clock: SimClock,
        target_lengths: Dict[SeqId, int],
        faults: FaultSchedule,
        latency: Optional[LatencyModel] = None,
    ) -> None:
        self.rng = rng
        self.clock = clock
        self.target_lengths = target_lengths
        self.faults = faults
        self.latency = latency or LatencyModel()

        self.emitted: Dict[SeqId, int] = {}
        self.steps_run = 0
        self.faults_raised = 0

    def run(self, step: ExecStep) -> ExecOutputs:
        self.steps_run += 1

        fault = self.faults.fault_for(step.step_id)
        if fault is not None:
            self.faults_raised += 1
            affected = [p.seq_id for p in step.prefills] + [
                d.seq_id for d in step.decodes
            ]
            # Fault a deterministic subset, not always the whole batch, so
            # partial-failure recovery paths get exercised too.
            if affected and len(affected) > 1:
                affected = affected[: 1 + (step.step_id % len(affected))]
            self.clock.advance(self.latency.step_ns(step) // 2)
            raise ExecFault(fault, f"injected at step {step.step_id}", affected)

        self.clock.advance(self.latency.step_ns(step))

        outputs: List[SeqOutput] = []
        gamma = step.spec_gamma

        for item in step.prefills:
            if not item.is_last_chunk:
                continue  # only the final chunk produces a logits row
            outputs.append(SeqOutput(seq_id=item.seq_id, token_ids=[self._next(item.seq_id)]))

        for item in step.decodes:
            if gamma > 0:
                outputs.append(self._speculative(item.seq_id, gamma))
            else:
                outputs.append(
                    SeqOutput(seq_id=item.seq_id, token_ids=[self._next(item.seq_id)])
                )

        return ExecOutputs(
            step_id=step.step_id,
            outputs=outputs,
            duration_s=self.latency.step_ns(step) / 1e9,
        )

    def _speculative(self, seq_id: SeqId, gamma: int) -> SeqOutput:
        """Model an accept/reject run. Commits accepted prefix plus bonus token."""
        alpha = (
            self.faults.forced_acceptance
            if self.faults.forced_acceptance is not None
            else 0.78  # the section 7.3 target, as a simulation input only
        )
        accepted = 0
        for _ in range(gamma):
            if self.rng.random() < alpha:
                accepted += 1
            else:
                break

        tokens = [self._next(seq_id) for _ in range(accepted + 1)]
        return SeqOutput(
            seq_id=seq_id,
            token_ids=tokens,
            num_drafted=gamma,
            num_accepted=accepted,
        )

    def _next(self, seq_id: SeqId) -> int:
        """Emit the next token, or SIM_EOS once the target length is reached."""
        n = self.emitted.get(seq_id, 0)
        target = self.target_lengths.get(seq_id, 8)
        self.emitted[seq_id] = n + 1
        if n + 1 >= target:
            return SIM_EOS
        # Deterministic non-EOS token id.
        return 1000 + (seq_id * 31 + n) % 5000


class DisaggSimulator:
    """Drives the KV transfer FSM under fault injection (spec section 6.4).

    Models a decode partition's block pool and a prefill partition handing
    sequences over to it. Faults come from the seeded RNG, so any failure replays
    exactly.

    The bug class this is aimed at is block accounting across a partial failure.
    A transfer that dies between SENT and RECEIVED has blocks reserved on the
    receiver that the sender does not know about, and blocks held on the sender
    that the receiver cannot see. Getting that wrong leaks on one side, on the
    other, or on both -- and leaks are invisible until the pool is exhausted,
    which in a real deployment is hours later and looks like a capacity problem
    rather than a correctness one.

    Two things are asserted that the FSM cannot check by itself, because they
    involve the pool it does not own:

      P1. receiver_free + receiver_reserved == receiver_total, every step. A
          reserved-but-forgotten block breaks this immediately.
      P2. at quiescence, every block is either free or held by a committed
          sequence. Nothing is in limbo.
    """

    def __init__(self, rng: random.Random, total_receiver_blocks: int) -> None:
        from .disagg import DisaggConfig, KVTransferManager

        self.rng = rng
        self.total_receiver_blocks = total_receiver_blocks
        self.free_receiver = set(range(total_receiver_blocks))
        # Blocks belonging to sequences that completed their migration and are
        # now decoding on the receiver. Freed when the sequence finishes.
        self.landed: Dict[int, List[int]] = {}

        self.manager = KVTransferManager(
            DisaggConfig(
                # Small on purpose: a link that never saturates never exercises
                # queueing, and a generous timeout never exercises reaping.
                link_bandwidth_bps=rng.choice([1e9, 12e9, 50e9]),
                max_concurrent_transfers=rng.randint(1, 4),
                transfer_timeout_steps=rng.randint(3, 20),
                max_retries=rng.randint(0, 3),
            )
        )
        self.next_seq = 0
        self.reprefills = 0
        self.oom_rejections = 0

    def _alloc_receiver(self, n: int) -> Optional[List[int]]:
        if len(self.free_receiver) < n:
            return None
        # Sorted, not popped arbitrarily: set iteration order is stable within a
        # run but not across Python versions, and DST replay has to survive that.
        blocks = sorted(self.free_receiver)[:n]
        self.free_receiver.difference_update(blocks)
        return blocks

    def _release_receiver(self, blocks: List[int]) -> None:
        for b in blocks:
            if b in self.free_receiver:
                raise TransferInvariantViolation(
                    f"receiver block {b} was freed twice -- double release"
                )
            self.free_receiver.add(b)

    def step(self, submit: bool) -> None:
        from .disagg import TransferFault, TransferState

        mgr = self.manager
        rng = self.rng

        # 1. A prefill finished; hand it over.
        if submit:
            n_blocks = rng.randint(1, 8)
            src = [10_000 + self.next_seq * 8 + i for i in range(n_blocks)]
            mgr.submit(self.next_seq, src, bytes_per_block=rng.choice([2**16, 2**20]))
            self.next_seq += 1

        # 2. Reserve destination blocks for queued transfers.
        for t in [x for x in mgr.transfers.values() if x.state == TransferState.PENDING]:
            dst = self._alloc_receiver(t.n_blocks)
            if dst is None:
                # Receiver is full. Not a failure yet -- it stays queued, which is
                # the correct behaviour and also the shape of a potential livelock
                # if nothing ever drains. The liveness check at the end is what
                # would catch that.
                self.oom_rejections += 1
                if rng.random() < 0.2:
                    src, d = mgr.fail(t, TransferFault.RECEIVER_OOM)
                    self._release_receiver(d)
                    if not mgr.retry(t):
                        self.reprefills += 1
                continue
            mgr.reserve(t, dst)

        # 3. Start what the link can carry.
        for t in [x for x in mgr.transfers.values() if x.state == TransferState.RESERVED]:
            if not mgr.can_begin():
                break
            mgr.begin(t)

        # 4. Move bytes.
        newly_sent = mgr.step()

        # 5. Resolve each completed send: verified, corrupted, or ack lost.
        for t in newly_sent:
            r = rng.random()
            if r < 0.05:
                src, dst = mgr.fail(t, TransferFault.CHECKSUM_MISMATCH)
                self._release_receiver(dst)
                if not mgr.retry(t):
                    self.reprefills += 1
            elif r < 0.10:
                # Ack lost: the receiver has the bytes but the sender never
                # learns. Left in SENT deliberately so the timeout path is what
                # resolves it -- that is the only mechanism that can.
                pass
            else:
                mgr.acknowledge(t)

        # 6. Commit acknowledged transfers; the sender's blocks are now free.
        for t in [x for x in mgr.transfers.values() if x.state == TransferState.RECEIVED]:
            mgr.commit(t)
            self.landed[t.seq_id] = list(t.dst_blocks)

        # 7. Reap timeouts.
        for t in mgr.timed_out():
            src, dst = mgr.fail(t, TransferFault.LINK_TIMEOUT)
            self._release_receiver(dst)
            if not mgr.retry(t):
                self.reprefills += 1

        # 8. Occasionally cancel a live transfer.
        live = [x for x in mgr.transfers.values() if not x.is_terminal]
        if live and rng.random() < 0.05:
            t = live[rng.randrange(len(live))]
            src, dst = mgr.abort(t)
            self._release_receiver(dst)

        # 9. Sequences finish decoding and give their blocks back.
        for seq_id in list(self.landed):
            if rng.random() < 0.25:
                self._release_receiver(self.landed.pop(seq_id))

        mgr.check_invariants()
        self.check_pool_invariants()

    def check_pool_invariants(self) -> None:
        """P1: every receiver block is free, reserved by a live transfer, or landed."""
        from .disagg import TransferState

        reserved: Dict[int, int] = {}
        for t in self.manager.transfers.values():
            if t.state in (
                TransferState.RESERVED, TransferState.SENDING,
                TransferState.SENT, TransferState.RECEIVED,
            ):
                for b in t.dst_blocks:
                    reserved[b] = t.transfer_id

        landed_blocks = {b for blocks in self.landed.values() for b in blocks}
        overlap = set(reserved) & landed_blocks
        if overlap:
            raise TransferInvariantViolation(
                f"P1: blocks {sorted(overlap)[:8]} are both reserved for a live "
                "transfer and held by a landed sequence"
            )
        double = (set(reserved) | landed_blocks) & self.free_receiver
        if double:
            raise TransferInvariantViolation(
                f"P1: blocks {sorted(double)[:8]} are in the free list while "
                "still owned"
            )
        total = len(self.free_receiver) + len(reserved) + len(landed_blocks)
        if total != self.total_receiver_blocks:
            raise TransferInvariantViolation(
                f"P1: receiver blocks do not add up: free={len(self.free_receiver)} "
                f"+ reserved={len(reserved)} + landed={len(landed_blocks)} "
                f"= {total}, expected {self.total_receiver_blocks}"
            )

    def drain(self, max_steps: int = 500) -> List[str]:
        """Run with no new submissions until quiescent. Returns any violations."""
        for _ in range(max_steps):
            if self.manager.n_active == 0:
                break
            try:
                self.step(submit=False)
            except TransferInvariantViolation as exc:
                return [f"disagg drain: {exc}"]

        out: List[str] = []
        if self.manager.n_active:
            # P2 as a liveness property: a transfer that can never be resolved is
            # a hang, and the only reason one should survive a drain is a missing
            # timeout path.
            states = [t.state.value for t in self.manager.transfers.values() if not t.is_terminal]
            out.append(
                f"disagg liveness: {self.manager.n_active} transfers never reached "
                f"a terminal state (states: {sorted(set(states))})"
            )
        for seq_id in list(self.landed):
            self._release_receiver(self.landed.pop(seq_id))
        if not out and len(self.free_receiver) != self.total_receiver_blocks:
            out.append(
                f"disagg leak: {self.total_receiver_blocks - len(self.free_receiver)} "
                "receiver blocks were never returned"
            )
        return out


@dataclass
class SimWorkload:
    """One generated workload: arrivals, lengths, classes, cancellations."""

    arrivals: List[Tuple[int, Request]] = field(default_factory=list)
    target_output_len: Dict[int, int] = field(default_factory=dict)
    cancel: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return len(self.arrivals)


class Simulation:
    """One seeded simulation run.

    Everything -- workload, fault schedule, config jitter -- derives from the
    seed, so `Simulation(seed).run()` is a total, reproducible experiment.
    """

    def __init__(
        self,
        seed: int,
        max_steps: int = 4000,
        check_every_step: bool = True,
        adversarial: bool = True,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.check_every_step = check_every_step
        self.adversarial = adversarial

        self.clock = SimClock()
        self.config = self._gen_config()
        self.allocator = Allocator(
            total_vram_blocks=self.config_total_blocks,
            block_size=self.config.block_size,
            total_host_blocks=self.config_host_blocks,
            watermark=self.config.watermark,
        )
        self.scheduler = Scheduler(
            self.config,
            self.allocator,
            PrefixCache(self.config.block_size, self.config.enable_prefix_cache),
        )

        self.workload = self._gen_workload()
        self.faults = self._gen_fault_schedule()
        self.executor = SimExecutor(
            rng=random.Random(seed ^ 0x5EED),  # separate stream from workload gen
            clock=self.clock,
            target_lengths={},
            faults=self.faults,
        )

        # Separate RNG stream so adding fork events cannot shift the
        # workload or fault schedules derived from the main stream.
        self.fork_rng = random.Random(seed ^ 0xF00D)

        # Disaggregation runs on its own RNG stream for the same reason: adding
        # it must not perturb any previously-verified seed's scheduler workload.
        # Half the seeds exercise it, so the co-located path stays covered too.
        self.disagg_rng = random.Random(seed ^ 0xD15A)
        self.disagg: Optional[DisaggSimulator] = None
        if adversarial and self.disagg_rng.random() < 0.5:
            self.disagg = DisaggSimulator(
                self.disagg_rng, total_receiver_blocks=self.disagg_rng.randint(8, 64)
            )

        # Forked block tables held across steps (see _maybe_fork). These are a
        # harness artifact and are released before the leak check.
        # Forked tables are created and freed within a single step; see
        # _maybe_fork for why they must not persist.

        self.trace: List[Dict[str, object]] = []
        self.submitted: Dict[int, SeqId] = {}   # request index -> seq_id
        self.steps_taken = 0

    # ------------------------------------------------------------ generation

    def _gen_config(self) -> SchedulerConfig:
        """Randomise the configuration so we explore the parameter space too.

        Deliberately includes cramped memory settings -- most interesting
        scheduler bugs only appear under memory pressure.
        """
        rng = self.rng
        block_size = rng.choice([8, 16, 32])
        # Small pools on purpose: forces preemption and CoW paths.
        self.config_total_blocks = rng.randint(24, 200)
        self.config_host_blocks = rng.choice([0, 0, 32, 64])

        return SchedulerConfig(
            max_num_seqs=rng.randint(2, 32),
            max_num_batched_tokens=rng.choice([128, 256, 512, 2048]),
            max_model_len=rng.choice([256, 512, 1024]),
            block_size=block_size,
            watermark=rng.choice([0.0, 0.01, 0.05]),
            enable_chunked_prefill=rng.random() < 0.8,
            enable_prefix_cache=rng.random() < 0.7,
            enable_spec_decode=rng.random() < 0.5,
            spec_gamma=rng.choice([2, 4, 8]),
            enable_swap=self.config_host_blocks > 0 and rng.random() < 0.6,
            class_a_protect_limit=rng.choice([0, 4, 8]),
        )

    def _gen_workload(self) -> SimWorkload:
        rng = self.rng
        n = rng.randint(1, 60)
        workload = SimWorkload()

        # Poisson-ish arrivals in step units, with occasional 10x bursts
        # (spec section 10.2: "extreme arrival bursts").
        step = 0
        burst_until = -1
        for i in range(n):
            if self.adversarial and rng.random() < 0.08:
                burst_until = step + rng.randint(3, 12)
            lam = 0.2 if step <= burst_until else rng.choice([1.0, 3.0, 8.0])
            step += max(0, int(rng.expovariate(1.0 / lam)))

            # Keep prompt + output inside max_model_len so requests are
            # satisfiable; the harness tests scheduling, not validation.
            headroom = self.config.max_model_len
            prompt_len = rng.randint(1, max(1, headroom // 2))
            out_len = rng.randint(1, max(1, min(64, headroom - prompt_len)))

            shared_prefix = rng.random() < 0.4
            if shared_prefix:
                # Reuse a canned prefix so the prefix cache actually hits.
                prefix = [7000 + j for j in range(min(prompt_len, 64))]
                rest = [rng.randint(10, 9000) for _ in range(prompt_len - len(prefix))]
                tokens = prefix + rest
            else:
                tokens = [rng.randint(10, 9000) for _ in range(prompt_len)]

            req = Request(
                request_id=f"sim-{self.seed}-{i}",
                prompt_token_ids=tokens,
                params=SamplingParams(
                    max_tokens=out_len,
                    temperature=0.0,
                    stop_token_ids=[SIM_EOS],
                ),
                slo_class=rng.choice([SloClass.A, SloClass.B, SloClass.C]),
                prefix_cache=rng.random() < 0.9,
            )
            workload.arrivals.append((step, req))
            workload.target_output_len[i] = out_len

            # Cancellation at an arbitrary later step (spec section 10.2).
            if self.adversarial and rng.random() < 0.12:
                cancel_step = step + rng.randint(1, 40)
                workload.cancel.setdefault(cancel_step, []).append(i)

        return workload

    def _gen_fault_schedule(self) -> FaultSchedule:
        sched = FaultSchedule()
        if not self.adversarial:
            return sched

        rng = self.rng
        horizon = max(20, self.workload.arrivals[-1][0] + 60 if self.workload.arrivals else 20)
        n_faults = rng.randint(0, 6)
        kinds = [
            FaultKind.OOM,
            FaultKind.CUDA_ERROR,
            FaultKind.TIMEOUT,
            FaultKind.TRANSFER_STALL,
            FaultKind.TRANSFER_CHECKSUM,
            FaultKind.TRANSFER_PARTIAL,
            FaultKind.SWAP_TIER_FULL,
        ]
        for _ in range(n_faults):
            sched.at_step[rng.randint(1, horizon)] = rng.choice(kinds)

        # Adversarial acceptance: 0% (all drafts wasted) and 100% (maximum
        # KV growth per step) are both interesting extremes.
        r = rng.random()
        if r < 0.15:
            sched.forced_acceptance = 0.0
        elif r < 0.30:
            sched.forced_acceptance = 1.0

        return sched

    # ------------------------------------------------------------------- run

    def _maybe_fork(self, step: int) -> None:
        """Occasionally fork a running sequence's block table, then write to it.

        This exists to reach the copy-on-write path, which is otherwise
        STRUCTURALLY UNREACHABLE in simulation: the prefix cache shares only
        whole blocks, so no sequence ever writes into a shared *partially
        filled* tail -- exactly the case spec section 5.3 names as the classic
        corruption bug, and exactly where BUG-003 lived. Coverage measurement
        showed 0/300 seeds exercising it before this was added.

        A fork models beam search / parallel sampling (`n > 1`), which the
        engine does not expose yet. The forked table is a scheduler-invisible
        shadow: it is created, appended to (forcing CoW), and freed here, so it
        tests the allocator's sharing logic without inventing scheduler
        semantics that do not exist.
        """
        if not self.adversarial or not self.scheduler.running:
            return
        if self.fork_rng.random() > 0.15:
            return


        alloc = self.allocator
        parent = self.scheduler.running[self.fork_rng.randrange(len(self.scheduler.running))]
        table = alloc.block_table(parent.seq_id)
        if table is None or not table.blocks:
            return

        # The fork lives for exactly this step. Holding it longer was tried and
        # reverted: the scheduler has no knowledge of these tables, so it can
        # neither evict nor preempt them, and a persisted shadow starves
        # admission and reports a liveness failure that belongs to the harness
        # rather than the engine (it cost ~60 spurious failures per 600 seeds).
        # Same-step forks still reach the CoW-at-the-watermark condition
        # whenever the scheduler happens to sit at the watermark; the case is
        # additionally pinned deterministically by
        # tests/allocator/test_allocator.py::test_cow_is_accounted_in_capacity_check,
        # which is the right place for a test that needs exact control of the
        # pool state.
        shadow_id = -(step + 1)  # negative ids cannot collide with real seq ids
        try:
            alloc.fork(parent.seq_id, shadow_id)
        except (ValueError, KeyError):
            return

        try:
            # Writing one token forces a CoW of the shared tail when that tail
            # is partially filled -- the case that matters.
            alloc.append_tokens(shadow_id, 1)
        except AllocError:
            pass

        alloc.free(shadow_id)
        # The executor never saw these blocks, so drop the copy requests.
        alloc.drain_pending_copies()

    def run(self) -> "SimResult":
        """Drive the scheduler to quiescence, asserting invariants throughout."""
        arrivals_by_step: Dict[int, List[Tuple[int, Request]]] = {}
        for idx, (step, req) in enumerate(self.workload.arrivals):
            arrivals_by_step.setdefault(step, []).append((idx, req))

        violations: List[str] = []
        last_arrival = (
            max(arrivals_by_step) if arrivals_by_step else 0
        )

        for step in range(self.max_steps):
            for idx, req in arrivals_by_step.get(step, []):
                try:
                    seq_id = self.scheduler.add_request(req)
                except ValueError:
                    continue  # rejected at admission; not a scheduler bug
                self.submitted[idx] = seq_id
                self.executor.target_lengths[seq_id] = self.workload.target_output_len[idx]

            for idx in self.workload.cancel.get(step, []):
                seq_id = self.submitted.get(idx)
                if seq_id is not None:
                    self.scheduler.abort(seq_id)

            self.scheduler.step(self.executor)
            self.steps_taken = step + 1

            self._maybe_fork(step)

            # KV transfers advance in lockstep with scheduler steps: the FSM is
            # step-driven for the same reason the scheduler is, so that a
            # simulated run is a total function of the seed (spec section 19.4).
            if self.disagg is not None:
                try:
                    self.disagg.step(submit=self.disagg_rng.random() < 0.3)
                except TransferInvariantViolation as exc:
                    violations.append(f"step {step}: disagg: {exc}")
                    break

            if self.check_every_step:
                try:
                    self.scheduler.check_invariants()
                except (InvariantViolation, AssertionError) as exc:
                    violations.append(f"step {step}: {exc}")
                    break

            self.trace.append(self.scheduler.snapshot())

            if step > last_arrival and not self.scheduler.has_work:
                break

        result = SimResult(
            seed=self.seed,
            steps=self.steps_taken,
            violations=violations,
            requests=self.workload.num_requests,
            submitted=len(self.submitted),
            finished=self.scheduler.stats.finished_seqs,
            aborted=self.scheduler.stats.aborted_seqs,
            faults_raised=self.executor.faults_raised,
            stats=self.scheduler.stats,
            sim_time_s=self.clock.now_s,
        )

        # Liveness: every submitted request must reach a terminal state.
        # Starvation is a real bug class here (spec section 10.3
        # assert_no_starvation), so it is checked, not assumed.
        if not violations and self.scheduler.has_work:
            result.violations.append(
                f"liveness: {self.scheduler.num_running()} running / "
                f"{self.scheduler.num_waiting()} waiting after {self.steps_taken} steps "
                f"(possible starvation or livelock)"
            )

        # No leaked blocks: every committed block must have an owner -- either a
        # live block table or an explicit prefix-cache reference. A block with a
        # positive ref count and no owner is unreachable and unreclaimable.
        #
        # Note this is checked by attribution, not by "the pool is empty at
        # quiescence": a sequence that finished but has not yet been reaped
        # still legitimately holds its table, and the prefix cache legitimately
        # retains blocks across requests.
        if not violations:
            owned = set()
            for table in self.allocator.iter_tables():
                owned.update(table.blocks)
            owned.update(self.allocator.cache_refs)
            orphaned = [
                bid
                for bid, blk in self.allocator.blocks.items()
                if blk.ref_count > 0 and bid not in owned
            ]
            if orphaned:
                result.violations.append(
                    f"leak: {len(orphaned)} blocks have a positive ref count "
                    f"but no owning block table or cache reference: "
                    f"{orphaned[:8]}"
                )

        # Drain the transfer FSM to quiescence and check the same two properties
        # there: everything terminal, and nothing leaked. A transfer stuck in SENT
        # forever is the disaggregated analogue of a scheduler livelock.
        if not violations and self.disagg is not None:
            result.violations.extend(self.disagg.drain())
            result.disagg_stats = self.disagg.manager.snapshot()

        return result


@dataclass
class SimResult:
    """Outcome of one seed."""

    seed: int
    steps: int
    violations: List[str] = field(default_factory=list)
    requests: int = 0
    submitted: int = 0
    finished: int = 0
    aborted: int = 0
    faults_raised: int = 0
    stats: Optional[object] = None
    sim_time_s: float = 0.0
    disagg_stats: Optional[Dict[str, object]] = None

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        extra = ""
        if self.disagg_stats:
            d = self.disagg_stats
            extra = (
                f" kvxfer={d['completed']}ok/{d['failed']}fail/"
                f"{d['reprefill_fallbacks']}reprefill"
            )
        return (
            f"[{status}] seed={self.seed} steps={self.steps} "
            f"reqs={self.submitted}/{self.requests} "
            f"finished={self.finished} aborted={self.aborted} "
            f"faults={self.faults_raised}{extra}"
        )


def run_seed(seed: int, max_steps: int = 4000, adversarial: bool = True) -> SimResult:
    """Run one seed. The unit the sweep driver parallelises over."""
    return Simulation(seed, max_steps=max_steps, adversarial=adversarial).run()


def sweep(
    start: int = 0,
    count: int = 1000,
    max_steps: int = 4000,
    stop_on_fail: bool = True,
    progress_every: int = 0,
) -> Tuple[int, List[SimResult]]:
    """Run seeds [start, start+count). Returns (n_passed, failures)."""
    passed = 0
    failures: List[SimResult] = []
    for seed in range(start, start + count):
        result = run_seed(seed, max_steps=max_steps)
        if result.ok:
            passed += 1
        else:
            failures.append(result)
            if stop_on_fail:
                break
        if progress_every and (seed - start + 1) % progress_every == 0:
            print(f"  ... {seed - start + 1}/{count} seeds, {len(failures)} failures")
    return passed, failures


def replay(seed: int, max_steps: int = 4000, verbose: bool = True) -> SimResult:
    """Re-run one seed with full tracing, for debugging a reported failure."""
    sim = Simulation(seed, max_steps=max_steps)
    if verbose:
        print(f"replaying seed {seed}")
        print(f"  config: {sim.config}")
        print(f"  vram_blocks={sim.config_total_blocks} host_blocks={sim.config_host_blocks}")
        print(f"  requests: {sim.workload.num_requests}")
        print(f"  faults: {sim.faults.at_step}")
        print(f"  forced_acceptance: {sim.faults.forced_acceptance}")
        print(f"  cancels: {sim.workload.cancel}")
    result = sim.run()
    if verbose:
        print(result.summary())
        for v in result.violations:
            print(f"  VIOLATION: {v}")
        if sim.trace:
            print("  last 5 steps:")
            for snap in sim.trace[-5:]:
                print(f"    {snap}")
    return result

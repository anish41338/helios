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

from .allocator import Allocator, InvariantViolation
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

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return (
            f"[{status}] seed={self.seed} steps={self.steps} "
            f"reqs={self.submitted}/{self.requests} "
            f"finished={self.finished} aborted={self.aborted} "
            f"faults={self.faults_raised}"
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

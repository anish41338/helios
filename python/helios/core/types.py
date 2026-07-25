"""Core request and sequence types shared by scheduler, executor, and frontend.

Pure data. Spec sections 6.2 (SLO classes) and 9.1 (API surface).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional

SeqId = int


class SloClass(IntEnum):
    """Spec section 6.2. Ordered so that lower value == higher priority.

    IntEnum rather than Enum so sort keys are total and stable.
    """

    A = 0  # interactive chat: TTFT < 200ms, TPOT < 25ms
    B = 1  # agentic / tool loops: TTFT < 1s, TPOT < 50ms
    C = 2  # batch / offline: best effort, first victim

    @classmethod
    def parse(cls, value: str) -> "SloClass":
        try:
            return cls[value.strip().upper()]
        except KeyError:
            raise ValueError(f"unknown slo_class {value!r}; expected A, B, or C")


# TTFT / TPOT targets in seconds, used for goodput accounting in the bench
# harness. These are SLO definitions, not measured results.
SLO_TARGETS: Dict[SloClass, Dict[str, Optional[float]]] = {
    SloClass.A: {"ttft": 0.200, "tpot": 0.025},
    SloClass.B: {"ttft": 1.000, "tpot": 0.050},
    SloClass.C: {"ttft": None, "tpot": None},
}


class SeqState(Enum):
    WAITING = "waiting"      # admitted to the queue, no KV yet
    PREFILL = "prefill"      # KV allocated, prompt not fully processed
    DECODE = "decode"        # generating
    PREEMPTED = "preempted"  # KV dropped or swapped, will resume
    FINISHED = "finished"    # EOS or max_tokens
    ABORTED = "aborted"      # cancelled by client or unrecoverable fault


class FinishReason(Enum):
    EOS = "stop"
    LENGTH = "length"
    STOP_STRING = "stop"
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    """Decoding controls. Mirrors the OpenAI surface in spec section 9.1."""

    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0            # 0 disables
    stop: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    seed: Optional[int] = None
    logprobs: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.logprobs is not None and not 0 <= self.logprobs <= 20:
            raise ValueError("logprobs must be in [0, 20]")


@dataclass
class Request:
    """A client request as it enters the scheduler (spec section 3 RequestEnvelope)."""

    request_id: str
    prompt_token_ids: List[int]
    params: SamplingParams
    slo_class: SloClass = SloClass.B
    arrival_step: int = 0
    prefix_cache: bool = True
    spec_gamma: Optional[int] = None   # None = use engine default

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


@dataclass
class Sequence:
    """Scheduler-side mutable state for one in-flight request.

    `num_computed_tokens` tracks how far prefill has progressed, which is what
    makes chunked prefill resumable: a sequence can be scheduled for a partial
    prompt chunk this step and the rest later.
    """

    seq_id: SeqId
    request: Request
    state: SeqState = SeqState.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    finish_reason: Optional[FinishReason] = None

    # Step indices, for latency accounting. Simulated-step units in DST,
    # real steps in production; the frontend records wall clock separately.
    first_token_step: Optional[int] = None
    last_token_step: Optional[int] = None
    preempt_count: int = 0

    # Number of prompt tokens satisfied by a prefix-cache hit.
    cached_prefix_len: int = 0

    # Length of the prefix-cache pin this sequence currently holds, in tokens.
    # Tracked separately from cached_prefix_len because the two diverge: a
    # sequence can hold a pin over more tokens than it reused (it pins what it
    # published on prefill completion), and acquire/release must balance
    # exactly or cache nodes leak holders and become permanently unevictable.
    pinned_prefix_len: int = 0

    @property
    def prompt_len(self) -> int:
        return self.request.prompt_len

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        """Prompt plus generated -- the KV length this sequence needs."""
        return self.prompt_len + self.num_output_tokens

    @property
    def slo_class(self) -> SloClass:
        return self.request.slo_class

    @property
    def is_prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.prompt_len

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(0, self.prompt_len - self.num_computed_tokens)

    @property
    def is_finished(self) -> bool:
        return self.state in (SeqState.FINISHED, SeqState.ABORTED)

    def all_token_ids(self) -> List[int]:
        return self.request.prompt_token_ids + self.output_token_ids

    def append_token(self, token_id: int, step: int) -> None:
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1
        if self.first_token_step is None:
            self.first_token_step = step
        self.last_token_step = step

    def reset_for_recompute(self) -> None:
        """Drop computed-token progress but keep generated output.

        Recompute preemption (spec section 6.3) re-prefills prompt AND the
        tokens generated so far, since their KV was discarded. Output tokens
        are retained so the client sees no duplication on resume.
        """
        self.num_computed_tokens = 0
        self.cached_prefix_len = 0
        self.state = SeqState.PREEMPTED
        self.preempt_count += 1

    def kv_len_needed(self) -> int:
        """Tokens of KV this sequence must have resident to take a decode step."""
        return self.total_tokens


@dataclass
class CompletionOutput:
    """What the frontend streams back."""

    request_id: str
    token_ids: List[int]
    text: str = ""
    finish_reason: Optional[FinishReason] = None
    prompt_tokens: int = 0
    cached_tokens: int = 0

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids)

"""The scheduler <-> executor contract.

Spec section 10.1: the executor sits behind a narrow interface so that a
simulated implementation can stand in for the real one. ExecStep is the only
thing that crosses the boundary in, ExecOutputs the only thing that crosses
back. Keeping it a plain dataclass (rather than tensors) is what lets the DST
harness fabricate steps cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol

from .types import SamplingParams, SeqId


class Phase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class PrefillItem:
    """One (possibly partial) prompt chunk to run through the model."""

    seq_id: SeqId
    token_ids: List[int]          # the chunk only, not the whole prompt
    start_pos: int                # position of token_ids[0] in the sequence
    block_ids: List[BlockId] = field(default_factory=list)  # type: ignore[name-defined]
    params: SamplingParams = field(default_factory=SamplingParams)
    is_last_chunk: bool = True    # only the last chunk samples a token

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class DecodeItem:
    """One sequence taking a single decode step."""

    seq_id: SeqId
    last_token_id: int
    position: int                 # position of the token being generated
    block_ids: List[int] = field(default_factory=list)
    params: SamplingParams = field(default_factory=SamplingParams)
    context_len: int = 0          # KV length to attend over


@dataclass
class BlockCopy:
    """A physical block copy the executor must perform before compute.

    Produced by copy-on-write and by the swap tier.
    """

    src: int
    dst: int


@dataclass
class ExecStep:
    """Everything the executor needs for one forward pass.

    A step may mix prefill and decode items (chunked prefill / stall-free
    batching, Sarathi-Serve style) -- that is how a long prompt avoids
    blocking in-flight decodes on a single device.
    """

    step_id: int
    prefills: List[PrefillItem] = field(default_factory=list)
    decodes: List[DecodeItem] = field(default_factory=list)
    block_copies: List[BlockCopy] = field(default_factory=list)
    swap_in: List[BlockCopy] = field(default_factory=list)
    swap_out: List[BlockCopy] = field(default_factory=list)
    spec_gamma: int = 0           # 0 = speculation disabled this step

    @property
    def is_empty(self) -> bool:
        return not (
            self.prefills
            or self.decodes
            or self.block_copies
            or self.swap_in
            or self.swap_out
        )

    @property
    def num_prefill_tokens(self) -> int:
        return sum(p.num_tokens for p in self.prefills)

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decodes)

    @property
    def batch_size(self) -> int:
        return len(self.prefills) + len(self.decodes)


@dataclass
class SeqOutput:
    """Per-sequence result of one step."""

    seq_id: SeqId
    token_ids: List[int] = field(default_factory=list)  # >1 only under speculation
    logprobs: Optional[List[Dict[int, float]]] = None
    # Speculation bookkeeping (spec section 7.2). num_drafted counts candidates
    # proposed, num_accepted how many survived verification.
    num_drafted: int = 0
    num_accepted: int = 0


@dataclass
class ExecOutputs:
    """What comes back from a forward pass."""

    step_id: int
    outputs: List[SeqOutput] = field(default_factory=list)
    # Wall-clock duration of the executor call, seconds. Simulated in DST.
    duration_s: float = 0.0

    def by_seq(self) -> Dict[SeqId, SeqOutput]:
        return {o.seq_id: o for o in self.outputs}


class FaultKind(Enum):
    """Spec section 10.2 fault taxonomy."""

    OOM = "oom"
    CUDA_ERROR = "cuda_error"
    TIMEOUT = "timeout"
    TRANSFER_STALL = "transfer_stall"
    TRANSFER_CHECKSUM = "transfer_checksum"
    TRANSFER_PARTIAL = "transfer_partial"
    SWAP_TIER_FULL = "swap_tier_full"


class ExecFault(Exception):
    """The executor failed this step. Recoverable: the scheduler preempts and retries.

    Carries the affected sequences so the scheduler knows what to roll back
    rather than having to abort the whole batch.
    """

    def __init__(
        self,
        kind: FaultKind,
        message: str = "",
        seq_ids: Optional[List[SeqId]] = None,
    ) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.seq_ids = seq_ids or []


class Executor(Protocol):
    """Spec section 10.1. Production impl runs PyTorch; SimExecutor fabricates."""

    def run(self, step: ExecStep) -> ExecOutputs:
        """Execute one step. Raises ExecFault on recoverable failure."""
        ...


BlockId = int

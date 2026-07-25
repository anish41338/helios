"""The production executor: consumes ExecStep, runs PyTorch, returns ExecOutputs.

Spec section 3 puts this behind a shared-memory ring buffer from a Rust
scheduler. This build runs scheduler and executor in one Python process, so the
boundary is a direct call -- but it is still the *same* narrow ExecStep /
ExecOutputs interface (helios.core.execstep), which is what lets SimExecutor
substitute for this class in the DST harness. Preserving that seam matters more
than the transport.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from ..core.execstep import (
    ExecFault,
    ExecOutputs,
    ExecStep,
    FaultKind,
    SeqOutput,
)
from .model import HeliosModel
from .paged_attn import PagedKVCache
from .sampler import Sampler, greedy_token


class ModelRunner:
    """Executes scheduler steps against a real model and paged KV cache."""

    def __init__(
        self,
        model: HeliosModel,
        num_blocks: int,
        block_size: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.config = model.config
        self.block_size = block_size
        self.device = device
        self.sampler = Sampler()

        # One cache per layer. Allocated up front so that a block id from the
        # scheduler is always backed by real storage -- the allocator's job is
        # to never hand out an id outside this range.
        self.kv_caches: List[PagedKVCache] = [
            PagedKVCache(
                num_blocks=num_blocks,
                block_size=block_size,
                n_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.config.num_hidden_layers)
        ]

        self.steps_run = 0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0

    @property
    def kv_bytes(self) -> int:
        return sum(c.nbytes for c in self.kv_caches)

    # --------------------------------------------------------------- executor

    def run(self, step: ExecStep) -> ExecOutputs:
        """Execute one scheduler step. Implements the Executor protocol."""
        start = time.perf_counter()
        self.steps_run += 1

        try:
            self._apply_block_copies(step)
            outputs: List[SeqOutput] = []

            with torch.inference_mode():
                for item in step.prefills:
                    out = self._run_prefill(item)
                    if out is not None:
                        outputs.append(out)

                for item in step.decodes:
                    outputs.append(self._run_decode(item, step.spec_gamma))

        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover (CPU build)
            raise ExecFault(FaultKind.OOM, str(exc)) from exc
        except (IndexError, RuntimeError) as exc:
            # A bad block id or shape mismatch means the scheduler and executor
            # disagree about cache layout. Surface it as a fault so the
            # scheduler rolls the affected sequences back rather than
            # committing tokens computed from the wrong KV.
            affected = [p.seq_id for p in step.prefills] + [
                d.seq_id for d in step.decodes
            ]
            raise ExecFault(FaultKind.CUDA_ERROR, str(exc), affected) from exc

        return ExecOutputs(
            step_id=step.step_id,
            outputs=outputs,
            duration_s=time.perf_counter() - start,
        )

    def _apply_block_copies(self, step: ExecStep) -> None:
        """Perform CoW and swap block copies before compute reads the cache."""
        for copy in step.block_copies:
            for cache in self.kv_caches:
                cache.copy_block(copy.src, copy.dst)
        # Swap in/out would move bytes between VRAM and host tiers. With a
        # single CPU tier there is nothing to move, but the ids still have to be
        # copied so the destination block holds the right data.
        for copy in list(step.swap_in) + list(step.swap_out):
            for cache in self.kv_caches:
                cache.copy_block(copy.src, copy.dst)

    def _run_prefill(self, item) -> Optional[SeqOutput]:
        """Process one prompt chunk. Only the final chunk samples a token."""
        positions = list(range(item.start_pos, item.start_pos + item.num_tokens))
        context_len = item.start_pos + item.num_tokens

        logits = self.model.forward(
            token_ids=item.token_ids,
            positions=positions,
            kv_caches=self.kv_caches,
            block_ids=item.block_ids,
            is_decode=False,
            context_len=context_len,
        )
        self.total_prefill_tokens += item.num_tokens

        if not item.is_last_chunk:
            return None  # mid-prompt: KV is written, but no token is due yet

        token = self.sampler.sample(logits[-1], item.params, str(item.seq_id))
        return SeqOutput(seq_id=item.seq_id, token_ids=[token])

    def _run_decode(self, item, gamma: int) -> SeqOutput:
        """Generate for one sequence, speculatively if gamma > 0."""
        if gamma > 0:
            return self._run_speculative(item, gamma)

        logits = self.model.forward(
            token_ids=[item.last_token_id],
            positions=[item.position],
            kv_caches=self.kv_caches,
            block_ids=item.block_ids,
            is_decode=True,
            context_len=item.context_len,
        )
        self.total_decode_tokens += 1
        token = self.sampler.sample(logits[-1], item.params, str(item.seq_id))
        return SeqOutput(seq_id=item.seq_id, token_ids=[token])

    def _run_speculative(self, item, gamma: int) -> SeqOutput:
        """Self-speculative decode: draft gamma tokens, verify in one pass.

        This is the *structure* of QASSD (spec section 7) without the
        quantization asymmetry: draft and verify read the same fp32 weights, so
        acceptance is 100% by construction and there is no speedup. It exists
        to prove the accept/rollback bookkeeping and the KV truncation are
        correct, which is the part the scheduler interacts with. The
        quantization-asymmetric version needs W4A4/W4A16 kernels -- see
        docs/SCOPE.md.

        Because draft and verify are numerically identical here, this path is
        also the strongest available check that speculation does not change
        outputs: any divergence from the non-speculative path is a bug in the
        bookkeeping, not a quantization artefact.
        """
        drafted: List[int] = []
        cur_token = item.last_token_id
        cur_pos = item.position
        ctx = item.context_len

        # DRAFT: gamma sequential cheap forward passes.
        for _ in range(gamma):
            logits = self.model.forward(
                token_ids=[cur_token],
                positions=[cur_pos],
                kv_caches=self.kv_caches,
                block_ids=item.block_ids,
                is_decode=True,
                context_len=ctx,
            )
            token = greedy_token(logits[-1])
            drafted.append(token)
            cur_token = token
            cur_pos += 1
            ctx += 1
            if ctx > len(item.block_ids) * self.block_size:
                break  # would overrun the sequence's allocated blocks

        if not drafted:
            return self._run_decode(item, 0)

        # VERIFY: one pass over all drafted positions.
        verify_tokens = [item.last_token_id] + drafted[:-1]
        verify_positions = list(range(item.position, item.position + len(verify_tokens)))
        all_logits = self.model.forward_all_logits(
            token_ids=verify_tokens,
            positions=verify_positions,
            kv_caches=self.kv_caches,
            block_ids=item.block_ids,
            context_len=item.context_len + len(verify_tokens) - 1,
        )

        # ACCEPT: longest prefix where the draft matches the verified argmax.
        accepted: List[int] = []
        for i, tok in enumerate(drafted):
            if greedy_token(all_logits[i]) == tok:
                accepted.append(tok)
            else:
                break

        # Always emit at least one token: the verified distribution's choice at
        # the first rejected position (the "bonus" token). Without it a
        # rejection would stall the sequence.
        if len(accepted) < len(drafted):
            bonus = greedy_token(all_logits[len(accepted)])
            committed = accepted + [bonus]
        else:
            committed = accepted

        self.total_decode_tokens += len(committed)
        return SeqOutput(
            seq_id=item.seq_id,
            token_ids=committed,
            num_drafted=len(drafted),
            num_accepted=len(accepted),
        )


def build_runner(
    model_dir: str,
    num_blocks: int,
    block_size: int = 16,
    device: str = "cpu",
) -> ModelRunner:
    """Load a model from disk and wrap it in a runner."""
    from .loader import load_model

    model = load_model(model_dir, device=device)
    return ModelRunner(
        model=model, num_blocks=num_blocks, block_size=block_size, device=device
    )

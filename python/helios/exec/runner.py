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
from .model import BatchedAttnMeta, HeliosModel
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
                # Prefill chunks are concatenated into one pass too. On a
                # prompt-heavy workload prefill is the overwhelming majority of
                # token work (measured 54:1 over decode), so batching it is what
                # moves TTFT.
                if step.prefills:
                    outputs.extend(self._run_prefill_batch(step.prefills))

                # Decodes go through one batched forward pass. Every projection
                # and MLP becomes a single GEMM across all resident sequences,
                # which is what makes continuous batching pay: measured ~10x on
                # CPU at 32 sequences versus one pass per sequence. Speculation
                # still runs per sequence (its draft loop is inherently serial).
                if step.decodes:
                    if step.spec_gamma > 0:
                        for item in step.decodes:
                            outputs.append(self._run_speculative(item, step.spec_gamma))
                    else:
                        outputs.extend(self._run_decode_batch(step.decodes))

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

    def _run_prefill_batch(self, items) -> List[SeqOutput]:
        """Run every prompt chunk in this step in ONE forward pass.

        Chunks have different lengths, so they are concatenated along the token
        dimension rather than padded into a rectangle -- padding to the longest
        chunk would waste work proportional to the length spread, which for
        prompts is large. Attention is still per-sequence (each chunk attends
        over its own prefix, causally), but the projections and MLP see one big
        GEMM over every prompt token in the step.

        Only final chunks produce a logits row, and the LM head is applied to
        just those rows: with a 128k vocab, projecting every prompt token would
        cost more than the rest of the layer stack.
        """
        token_ids: List[int] = []
        positions: List[int] = []
        slices = []
        block_tables = []
        context_lens = []
        start_positions = []

        for item in items:
            lo = len(token_ids)
            token_ids.extend(item.token_ids)
            positions.extend(
                range(item.start_pos, item.start_pos + item.num_tokens)
            )
            slices.append((lo, len(token_ids)))
            block_tables.append(list(item.block_ids))
            context_lens.append(item.start_pos + item.num_tokens)
            start_positions.append(item.start_pos)
            self.total_prefill_tokens += item.num_tokens

        meta = BatchedAttnMeta(
            token_slices=slices,
            block_tables=block_tables,
            context_lens=context_lens,
            start_positions=start_positions,
            is_decode=False,
        )

        # Sample only where a prompt actually completed this step.
        final = [i for i, it in enumerate(items) if it.is_last_chunk]
        if not final:
            # KV still has to be written, so run the pass and take no logits.
            self.model.forward_batched(
                token_ids, positions, self.kv_caches, meta, logits_at=[0]
            )
            return []

        logits_at = [slices[i][1] - 1 for i in final]
        logits = self.model.forward_batched(
            token_ids, positions, self.kv_caches, meta, logits_at=logits_at
        )

        out: List[SeqOutput] = []
        for row, i in enumerate(final):
            item = items[i]
            token = self.sampler.sample(logits[row], item.params, str(item.seq_id))
            out.append(SeqOutput(seq_id=item.seq_id, token_ids=[token]))
        return out

    def _run_decode_batch(self, items) -> List[SeqOutput]:
        """Run every decoding sequence in ONE forward pass.

        Sequences are concatenated along the token dimension (one token each),
        so the projections and MLP are single large GEMMs while attention is
        done per sequence over its own paged KV. Output is bit-identical to
        running them one at a time -- asserted by
        tests/parity/test_parity.py::test_batched_decode_matches_sequential,
        because a batching bug here would silently mix sequences' KV, which is
        the worst failure mode this engine has (spec section 19.1).
        """
        token_ids = [it.last_token_id for it in items]
        positions = [it.position for it in items]
        meta = BatchedAttnMeta(
            token_slices=[(i, i + 1) for i in range(len(items))],
            block_tables=[list(it.block_ids) for it in items],
            context_lens=[it.context_len for it in items],
            start_positions=positions,
            is_decode=True,
        )

        logits = self.model.forward_batched(
            token_ids=token_ids,
            positions=positions,
            kv_caches=self.kv_caches,
            meta=meta,
        )
        self.total_decode_tokens += len(items)

        out: List[SeqOutput] = []
        for i, item in enumerate(items):
            token = self.sampler.sample(logits[i], item.params, str(item.seq_id))
            out.append(SeqOutput(seq_id=item.seq_id, token_ids=[token]))
        return out

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

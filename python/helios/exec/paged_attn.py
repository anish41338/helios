"""Paged attention over a block table.

Spec section 8.3. The spec's v1 plan is a Triton kernel with a PyTorch
gather+SDPA fallback. This build ships the fallback path only, because the
machine it was developed on has no CUDA device (torch 2.5.1+cpu) and Triton
does not target CPU -- an unrunnable kernel would be unverifiable code, which
spec section 19.5 forbids. The gather+SDPA path is the correctness reference
the kernel would have been validated against, so it is the right half to build
first regardless.

The core operation: a sequence's KV lives in scattered fixed-size physical
blocks, so attention must gather those blocks into a contiguous view before
computing scores. That indirection is the cost of paging; the memory it saves
(no per-sequence max_seq_len reservation) is the benefit.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


class PagedKVCache:
    """Physical KV storage, shaped [num_blocks, n_kv_heads, block_size, head_dim].

    One such cache per layer. The scheduler's allocator owns the *mapping* from
    sequences to block ids; this class owns the bytes. Keeping those separate is
    what let the allocator be tested exhaustively without a GPU.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_blocks, n_kv_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2

    def write(
        self,
        block_ids: List[int],
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter k/v for consecutive positions into the sequence's blocks.

        k, v: [n_tokens, n_kv_heads, head_dim]. `start_pos` is the position of
        k[0] within the sequence, which is what makes chunked prefill work --
        successive chunks write at increasing offsets into the same block list.
        """
        n_tokens = k.shape[0]
        if n_tokens == 0:
            return

        for i in range(n_tokens):
            pos = start_pos + i
            logical_block = pos // self.block_size
            offset = pos % self.block_size
            if logical_block >= len(block_ids):
                raise IndexError(
                    f"position {pos} needs logical block {logical_block} but the "
                    f"sequence has only {len(block_ids)} blocks"
                )
            phys = block_ids[logical_block]
            self.k_cache[phys, :, offset, :] = k[i]
            self.v_cache[phys, :, offset, :] = v[i]

    def gather(self, block_ids: List[int], context_len: int) -> tuple:
        """Collect a sequence's KV into contiguous [context_len, n_kv_heads, head_dim].

        This gather is the overhead paging trades for its memory savings. A
        fused kernel would instead stream blocks directly into the attention
        accumulator and never materialise this tensor.
        """
        if context_len == 0:
            empty = torch.zeros(
                (0, self.n_kv_heads, self.head_dim), dtype=self.dtype, device=self.device
            )
            return empty, empty

        n_blocks = (context_len + self.block_size - 1) // self.block_size
        if n_blocks > len(block_ids):
            raise IndexError(
                f"context_len {context_len} needs {n_blocks} blocks, have {len(block_ids)}"
            )

        idx = torch.tensor(block_ids[:n_blocks], dtype=torch.long, device=self.device)
        # [n_blocks, n_kv_heads, block_size, head_dim] -> [n_blocks*block_size, ...]
        k = self.k_cache[idx].permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim)
        v = self.v_cache[idx].permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim)
        return k[:context_len], v[:context_len]

    def copy_block(self, src: int, dst: int) -> None:
        """Physical block copy, used by copy-on-write and the swap tier."""
        self.k_cache[dst].copy_(self.k_cache[src])
        self.v_cache[dst].copy_(self.v_cache[src])


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    GQA stores fewer KV heads than Q heads, which is what shrinks the KV cache;
    each KV head is shared by n_rep query heads.
    """
    if n_rep == 1:
        return x
    n_tokens, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, None, :]
        .expand(n_tokens, n_kv_heads, n_rep, head_dim)
        .reshape(n_tokens, n_kv_heads * n_rep, head_dim)
    )


def paged_attention_decode(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    block_ids: List[int],
    context_len: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Single-token attention over a paged cache.

    query: [n_q_heads, head_dim] -- one decode step for one sequence.
    Returns [n_q_heads, head_dim].

    Decode is memory-bandwidth-bound: one query token attends over the whole
    context, so the step reads the entire KV cache for the sequence and does
    O(context_len) work per head with essentially no data reuse. That low
    arithmetic intensity is exactly why decode and prefill contend badly when
    interleaved, and why the spec wants them on separate partitions.
    """
    n_q_heads, head_dim = query.shape
    scale = scale or 1.0 / math.sqrt(head_dim)

    k, v = kv_cache.gather(block_ids, context_len)
    n_rep = n_q_heads // kv_cache.n_kv_heads
    k = _repeat_kv(k, n_rep)      # [context_len, n_q_heads, head_dim]
    v = _repeat_kv(v, n_rep)

    # [n_q_heads, 1, head_dim] x [n_q_heads, head_dim, context_len]
    q = query.unsqueeze(1)
    scores = torch.matmul(q, k.permute(1, 2, 0)) * scale     # [n_q_heads, 1, ctx]
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.permute(1, 0, 2))            # [n_q_heads, 1, head_dim]
    return out.squeeze(1)


def paged_attention_prefill(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    block_ids: List[int],
    start_pos: int,
    context_len: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Causal attention for a prompt chunk over a paged cache.

    query: [n_tokens, n_q_heads, head_dim] for the chunk being processed.
    `start_pos` is the chunk's offset in the sequence and `context_len` the
    total valid KV length (start_pos + n_tokens), so a chunk correctly attends
    to all previously-cached tokens as well as to itself. That is what makes
    chunked prefill produce identical results to a single-shot prefill.

    Prefill is compute-bound: n_tokens queries each attend over up to
    context_len keys, so work grows quadratically in prompt length while the
    weight reads amortise across all tokens in the batch.
    """
    n_tokens, n_q_heads, head_dim = query.shape
    scale = scale or 1.0 / math.sqrt(head_dim)

    k, v = kv_cache.gather(block_ids, context_len)
    n_rep = n_q_heads // kv_cache.n_kv_heads
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    q = query.permute(1, 0, 2)                    # [n_q_heads, n_tokens, head_dim]
    scores = torch.matmul(q, k.permute(1, 2, 0)) * scale  # [n_q_heads, n_tok, ctx]

    # Causal mask: query token at absolute position start_pos + i may attend to
    # key positions <= start_pos + i.
    q_pos = torch.arange(n_tokens, device=query.device) + start_pos
    k_pos = torch.arange(context_len, device=query.device)
    mask = k_pos[None, :] > q_pos[:, None]
    scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.permute(1, 0, 2))  # [n_q_heads, n_tokens, head_dim]
    return out.permute(1, 0, 2)                    # [n_tokens, n_q_heads, head_dim]

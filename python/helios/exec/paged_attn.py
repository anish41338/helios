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


class QuantizedPagedKVCache(PagedKVCache):
    """INT8 paged KV storage with per-token, per-head scales (spec section 7.4).

    Drop-in for PagedKVCache: `write` quantizes, `gather` dequantizes, so every
    attention path above is unchanged and stays fp. That is the point of a
    "quantized shadow" -- the *storage* shrinks, the *math* does not move.

    Granularity is per (block, head, token): one fp scale for each token's
    head_dim-length key vector. Coarser (per-tensor, as vLLM's fp8 KV uses) is
    cheaper to store but attention is a dot product over head_dim, so a single
    outlier token's magnitude would set the scale for every token that shares it.
    Per-token costs 2 bytes per (token, head) against head_dim bytes of payload,
    which at head_dim=64 is a 3% overhead for a much tighter fit.

    Symmetric rather than asymmetric: keys and values are near-centred, and a
    zero point would have to be carried through the score matmul as an extra
    correction term rather than folding into a single multiply.

    Why bother: the KV cache, not the weights, is what limits concurrency at
    long context. Halving it doubles the number of resident sequences at the same
    byte budget, and the batched-decode win is proportional to that batch size.
    The cost is accuracy, bounded and measured by tests/quant/test_kv_quant.py.
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
        # Deliberately does NOT call super().__init__: that would allocate the fp
        # tensors this class exists to avoid.
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype          # the dtype gather() returns, not the storage
        self.device = device

        shape = (num_blocks, n_kv_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        self.v_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        # Scale storage is fp16 regardless of compute dtype: a per-token scale is
        # a magnitude, and fp16's 10-bit mantissa is far more precision than an
        # int8 range needs.
        sshape = (num_blocks, n_kv_heads, block_size)
        self.k_scale = torch.ones(sshape, dtype=torch.float16, device=device)
        self.v_scale = torch.ones(sshape, dtype=torch.float16, device=device)

    @property
    def nbytes(self) -> int:
        return 2 * (
            self.k_cache.numel() * self.k_cache.element_size()
            + self.k_scale.numel() * self.k_scale.element_size()
        )

    @staticmethod
    def _quantize(x: torch.Tensor):
        """x: [n_tokens, n_heads, head_dim] -> (int8, scale [n_tokens, n_heads])."""
        absmax = x.abs().amax(dim=-1).clamp(min=1e-8)
        scale = absmax / 127.0
        q = torch.round(x / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        return q, scale

    def write(
        self,
        block_ids: List[int],
        start_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        n_tokens = k.shape[0]
        if n_tokens == 0:
            return
        kq, ks = self._quantize(k.float())
        vq, vs = self._quantize(v.float())

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
            self.k_cache[phys, :, offset, :] = kq[i]
            self.v_cache[phys, :, offset, :] = vq[i]
            self.k_scale[phys, :, offset] = ks[i].to(torch.float16)
            self.v_scale[phys, :, offset] = vs[i].to(torch.float16)

    def gather(self, block_ids: List[int], context_len: int) -> tuple:
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

        def deq(cache, scales):
            q = cache[idx].permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim)
            s = scales[idx].permute(0, 2, 1).reshape(-1, self.n_kv_heads)
            return (q[:context_len].float() * s[:context_len].float().unsqueeze(-1)).to(
                self.dtype
            )

        return deq(self.k_cache, self.k_scale), deq(self.v_cache, self.v_scale)

    def copy_block(self, src: int, dst: int) -> None:
        self.k_cache[dst].copy_(self.k_cache[src])
        self.v_cache[dst].copy_(self.v_cache[src])
        # The scales are as much a part of the block as the payload. Forgetting
        # them would leave the copy holding the destination's old magnitudes
        # against the source's quantized values -- a copy-on-write that produces
        # numerically wrong attention rather than an obvious crash.
        self.k_scale[dst].copy_(self.k_scale[src])
        self.v_scale[dst].copy_(self.v_scale[src])


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


def paged_attention_decode_batched(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    block_tables: List[List[int]],
    context_lens: List[int],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Decode attention for a whole batch of sequences at once.

    query: [n_seqs, n_q_heads, head_dim] -- one query token per sequence.
    Returns [n_seqs, n_q_heads, head_dim].

    Sequences have different context lengths and scattered blocks, so their KV
    is gathered into one right-padded [n_seqs, max_ctx, ...] tensor and the pad
    positions are masked to -inf before softmax. That trades some wasted work on
    the padding for a single batched matmul instead of n_seqs small ones, which
    is the whole point: on CPU the small-matmul path is dominated by per-call
    overhead, and batching the GEMMs measures ~10x faster at n_seqs=32.

    A fused GPU kernel would instead loop blocks per (seq, head) program and
    skip the padding entirely (spec section 8.3); this is the vectorised
    equivalent available without one.
    """
    n_seqs, n_q_heads, head_dim = query.shape
    scale = scale or 1.0 / math.sqrt(head_dim)
    if n_seqs == 0:
        return query
    max_ctx = max(context_lens)
    n_rep = n_q_heads // kv_cache.n_kv_heads

    k_pad = torch.zeros(
        (n_seqs, max_ctx, kv_cache.n_kv_heads, head_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    v_pad = torch.zeros_like(k_pad)
    for i, (blocks, ctx) in enumerate(zip(block_tables, context_lens)):
        k, v = kv_cache.gather(blocks, ctx)
        k_pad[i, :ctx] = k
        v_pad[i, :ctx] = v

    # [n_seqs, max_ctx, n_q_heads, head_dim] -> [n_seqs, n_q_heads, max_ctx, hd]
    k_rep = _repeat_kv_batched(k_pad, n_rep).transpose(1, 2)
    v_rep = _repeat_kv_batched(v_pad, n_rep).transpose(1, 2)

    q = query.unsqueeze(2)                      # [n_seqs, n_q_heads, 1, hd]
    scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scale   # [...,1,max_ctx]

    # Mask padding beyond each sequence's real context length.
    ctx_t = torch.tensor(context_lens, device=query.device).view(n_seqs, 1, 1, 1)
    idx = torch.arange(max_ctx, device=query.device).view(1, 1, 1, max_ctx)
    scores = scores.masked_fill(idx >= ctx_t, float("-inf"))

    probs = torch.softmax(scores.float(), dim=-1).to(v_rep.dtype)
    out = torch.matmul(probs, v_rep)            # [n_seqs, n_q_heads, 1, hd]
    return out.squeeze(2)


def _repeat_kv_batched(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA head expansion for a batched [n_seqs, ctx, n_kv_heads, hd] tensor."""
    if n_rep == 1:
        return x
    n_seqs, ctx, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(n_seqs, ctx, n_kv_heads, n_rep, head_dim)
        .reshape(n_seqs, ctx, n_kv_heads * n_rep, head_dim)
    )


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

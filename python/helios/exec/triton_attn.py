"""Triton paged-attention kernel (spec section 8.3).

STATUS: VERIFIED on a Tesla T4 (sm_75, CUDA 12.8, Triton 3.6.0, torch 2.10.0).

    9/9 tests/parity/test_triton_parity.py passed
    paged decode @ 32 seqs x 512 ctx: pytorch 14.089 ms -> triton 2.676 ms
    speedup 5.26x

Authored on a machine with no NVIDIA GPU and marked unverified until that run;
the parity gate is what promoted it (spec section 19.3). Re-run it on any new
device before trusting the numbers -- they are specific to sm_75.

Why a kernel at all: the PyTorch fallback in paged_attn.py gathers a sequence's
scattered KV blocks into a contiguous tensor before computing scores. That
gather is pure overhead -- it materialises memory the math does not need. A fused
kernel instead loops over the block table and streams each block directly into
an online-softmax accumulator, so the score matrix is never written to memory at
all (FlashAttention's IO-aware idea, applied to paged storage).

The correctness oracle is the existing `paged_attention_decode` /
`paged_attention_decode_batched`, which the CPU suite already pins against a
dense reference. That is the whole reason the fallback was built first.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

try:                                  # pragma: no cover - depends on hardware
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:                   # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def triton_available() -> bool:
    """True only if Triton is importable AND a CUDA device exists.

    Both halves matter: Triton installs on CPU-only machines but cannot compile
    without a target, so importability alone is not enough to run the kernel.
    """
    return HAS_TRITON and torch.cuda.is_available()


if HAS_TRITON:                        # pragma: no cover - requires GPU

    @triton.jit
    def _paged_attn_decode_kernel(
        q_ptr,              # [n_seqs, n_q_heads, head_dim]
        k_cache_ptr,        # [n_blocks, n_kv_heads, block_size, head_dim]
        v_cache_ptr,
        out_ptr,            # [n_seqs, n_q_heads, head_dim]
        block_tables_ptr,   # [n_seqs, max_blocks]
        seq_lens_ptr,       # [n_seqs]
        scale,
        n_kv_heads: tl.constexpr,
        n_q_heads: tl.constexpr,
        max_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """One program per (sequence, query head).

        Streams the sequence's KV blocks through an online softmax, so neither
        the gathered KV nor the score vector is ever materialised in HBM. State
        is O(HEAD_DIM) per program instead of O(context_len).
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        if seq_len == 0:
            return

        # Grouped-query attention: several query heads share one KV head.
        kv_head = head_idx // (n_q_heads // n_kv_heads)

        d_offs = tl.arange(0, HEAD_DIM)
        q = tl.load(
            q_ptr + seq_idx * n_q_heads * HEAD_DIM + head_idx * HEAD_DIM + d_offs
        ).to(tl.float32)

        # Online-softmax running state (FlashAttention-style).
        m_i = float("-inf")           # running max of the scores
        l_i = 0.0                     # running sum of exp(score - m_i)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        t_offs = tl.arange(0, BLOCK_SIZE)
        n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        for b in range(0, n_blocks):
            phys = tl.load(block_tables_ptr + seq_idx * max_blocks + b)
            # Tokens in this block that are actually part of the sequence.
            tok_pos = b * BLOCK_SIZE + t_offs
            valid = tok_pos < seq_len

            base = (
                phys * n_kv_heads * BLOCK_SIZE * HEAD_DIM
                + kv_head * BLOCK_SIZE * HEAD_DIM
            )
            kv_offs = base + t_offs[:, None] * HEAD_DIM + d_offs[None, :]

            k = tl.load(k_cache_ptr + kv_offs, mask=valid[:, None], other=0.0)
            v = tl.load(v_cache_ptr + kv_offs, mask=valid[:, None], other=0.0)

            # scores: [BLOCK_SIZE]
            s = tl.sum(k.to(tl.float32) * q[None, :], axis=1) * scale
            s = tl.where(valid, s, float("-inf"))

            # Rescale the accumulator to the new max before adding this block.
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)

            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(
            out_ptr + seq_idx * n_q_heads * HEAD_DIM + head_idx * HEAD_DIM + d_offs,
            out.to(out_ptr.dtype.element_ty),
        )


def paged_attention_decode_triton(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: List[List[int]],
    context_lens: List[int],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused paged decode attention. query: [n_seqs, n_q_heads, head_dim].

    Raises RuntimeError rather than silently falling back, so a caller that
    believes it is measuring the kernel always is.
    """
    if not triton_available():
        raise RuntimeError(
            "Triton kernel requires an NVIDIA GPU; use "
            "paged_attention_decode_batched for the PyTorch path"
        )

    n_seqs, n_q_heads, head_dim = query.shape
    n_blocks_total, n_kv_heads, block_size, _ = k_cache.shape
    scale = scale or 1.0 / math.sqrt(head_dim)

    max_blocks = max((len(t) for t in block_tables), default=1)
    tables = torch.zeros(
        (n_seqs, max_blocks), dtype=torch.int32, device=query.device
    )
    for i, t in enumerate(block_tables):
        if t:
            tables[i, : len(t)] = torch.tensor(
                t, dtype=torch.int32, device=query.device
            )
    lens = torch.tensor(context_lens, dtype=torch.int32, device=query.device)

    out = torch.empty_like(query)
    grid = (n_seqs, n_q_heads)
    _paged_attn_decode_kernel[grid](
        query,
        k_cache,
        v_cache,
        out,
        tables,
        lens,
        scale,
        n_kv_heads=n_kv_heads,
        n_q_heads=n_q_heads,
        max_blocks=max_blocks,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
    return out

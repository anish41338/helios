"""Triton kernel parity tests -- the gate that promotes the kernel to 'verified'.

Every test here SKIPS on a machine without an NVIDIA GPU, so the CPU suite stays
green. Nothing about `triton_attn.py` may be claimed until these pass on real
hardware (spec section 19.3).

The oracle is `paged_attention_decode_batched`, the PyTorch path the CPU suite
already pins against a dense reference implementation. That chain matters: dense
reference -> PyTorch paged -> Triton paged. Each rung is checked against the one
below it, so a kernel bug cannot hide behind a shared assumption.
"""

from __future__ import annotations

import math

import pytest
import torch

from helios.exec.paged_attn import PagedKVCache, paged_attention_decode_batched
from helios.exec.triton_attn import paged_attention_decode_triton, triton_available

pytestmark = pytest.mark.skipif(
    not triton_available(),
    reason="Triton paged-attention kernel requires an NVIDIA GPU",
)

# fp16 on tensor cores accumulates differently from fp32 on CPU. These are the
# tolerances a fused kernel should hold; they are NOT to be loosened to make a
# test pass (spec section 19.2). If the kernel cannot hold them, the kernel is
# wrong.
ATOL = 2e-2
RTOL = 2e-2


def _make_cache(n_blocks, block_size, n_kv_heads, head_dim, dtype, device):
    cache = PagedKVCache(
        num_blocks=n_blocks,
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    cache.k_cache.normal_()
    cache.v_cache.normal_()
    return cache


@pytest.mark.parametrize("n_seqs", [1, 4, 17])
@pytest.mark.parametrize("block_size", [16, 32])
def test_triton_decode_matches_pytorch_paged(n_seqs, block_size):
    """The fused kernel must agree with the PyTorch paged path.

    Context lengths are deliberately not multiples of block_size, so the
    kernel's tail masking is exercised -- reading past a sequence's real length
    would pull in another sequence's KV and silently corrupt output.
    """
    torch.manual_seed(0)
    dev, dtype = "cuda", torch.float16
    n_kv_heads, n_q_heads, head_dim = 2, 8, 64
    n_blocks = 256

    cache = _make_cache(n_blocks, block_size, n_kv_heads, head_dim, dtype, dev)

    ctx_lens, tables = [], []
    used = 0
    for i in range(n_seqs):
        ctx = block_size * (i % 3 + 1) - (i % block_size)   # ragged on purpose
        ctx = max(1, ctx)
        need = (ctx + block_size - 1) // block_size
        # Non-contiguous physical blocks: a block-table indexing bug shows up
        # here and not with sequential ids.
        tables.append([(used + j * 7) % n_blocks for j in range(need)])
        used += need
        ctx_lens.append(ctx)

    q = torch.randn(n_seqs, n_q_heads, head_dim, dtype=dtype, device=dev)
    scale = 1.0 / math.sqrt(head_dim)

    want = paged_attention_decode_batched(q, cache, tables, ctx_lens, scale)
    got = paged_attention_decode_triton(
        q, cache.k_cache, cache.v_cache, tables, ctx_lens, scale
    )

    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_triton_decode_is_order_invariant():
    """A sequence's output must not depend on its row in the batch."""
    torch.manual_seed(1)
    dev, dtype = "cuda", torch.float16
    block_size, n_kv_heads, n_q_heads, head_dim = 16, 2, 4, 64
    cache = _make_cache(64, block_size, n_kv_heads, head_dim, dtype, dev)

    tables = [[0, 1], [5, 2, 9], [3]]
    ctx_lens = [20, 40, 7]
    q = torch.randn(3, n_q_heads, head_dim, dtype=dtype, device=dev)

    fwd = paged_attention_decode_triton(
        q, cache.k_cache, cache.v_cache, tables, ctx_lens
    )
    order = [2, 0, 1]
    rev = paged_attention_decode_triton(
        q[order],
        cache.k_cache,
        cache.v_cache,
        [tables[i] for i in order],
        [ctx_lens[i] for i in order],
    )
    for row, i in enumerate(order):
        torch.testing.assert_close(rev[row], fwd[i], atol=ATOL, rtol=RTOL)


def test_triton_single_token_context():
    """A context of exactly one token is the degenerate case most likely to break."""
    torch.manual_seed(2)
    dev, dtype = "cuda", torch.float16
    cache = _make_cache(16, 16, 1, 64, dtype, dev)
    q = torch.randn(1, 1, 64, dtype=dtype, device=dev)

    got = paged_attention_decode_triton(q, cache.k_cache, cache.v_cache, [[0]], [1])
    want = paged_attention_decode_batched(q, cache, [[0]], [1])
    torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL)


def test_triton_beats_pytorch_path():
    """The kernel must actually be faster, or it is not worth its complexity.

    Asserted rather than assumed: a fused kernel that loses to a gather+SDPA
    fallback would mean the fusion is wrong, and shipping it on the assumption
    that "fused is faster" is exactly the kind of unmeasured claim spec section
    19.3 forbids.
    """
    import time

    torch.manual_seed(3)
    dev, dtype = "cuda", torch.float16
    block_size, n_kv_heads, n_q_heads, head_dim = 16, 8, 32, 128
    n_seqs, ctx = 32, 512
    need = ctx // block_size

    cache = _make_cache(n_seqs * need + 8, block_size, n_kv_heads, head_dim, dtype, dev)
    tables = [[i * need + j for j in range(need)] for i in range(n_seqs)]
    ctx_lens = [ctx] * n_seqs
    q = torch.randn(n_seqs, n_q_heads, head_dim, dtype=dtype, device=dev)

    def bench(fn, iters=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    t_torch = bench(
        lambda: paged_attention_decode_batched(q, cache, tables, ctx_lens)
    )
    t_triton = bench(
        lambda: paged_attention_decode_triton(
            q, cache.k_cache, cache.v_cache, tables, ctx_lens
        )
    )
    print(
        f"\npaged decode @ {n_seqs} seqs x {ctx} ctx: "
        f"pytorch {t_torch * 1e3:.3f} ms, triton {t_triton * 1e3:.3f} ms, "
        f"speedup {t_torch / t_triton:.2f}x"
    )
    assert t_triton < t_torch, (
        f"kernel ({t_triton * 1e3:.3f} ms) is slower than the PyTorch path "
        f"({t_torch * 1e3:.3f} ms)"
    )

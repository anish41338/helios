"""INT8 paged KV cache tests (spec section 7.4).

A quantized KV cache is a dangerous optimisation: it is invisible when wrong.
Attention still produces a plausible distribution from slightly-wrong keys, so
the failure mode is degraded output quality rather than a crash -- which no
smoke test catches. These tests therefore pin the numerical claim explicitly, and
pin the accounting claims (bytes, block copies) exactly.

The specific hazard `test_copy_block_moves_the_scales` covers: copy-on-write
copies the payload and, if the scales are forgotten, leaves the destination
holding the source's int8 values against its own stale magnitudes. Every
invariant in the allocator still holds. The output is just wrong.
"""

from __future__ import annotations

import pytest
import torch

from helios.core.allocator import Allocator
from helios.exec.paged_attn import (
    PagedKVCache,
    QuantizedPagedKVCache,
    paged_attention_decode,
)

N_BLOCKS, BLOCK, HEADS, DIM = 8, 16, 2, 64


def _caches():
    kwargs = dict(
        num_blocks=N_BLOCKS, block_size=BLOCK, n_kv_heads=HEADS, head_dim=DIM,
        dtype=torch.float32, device="cpu",
    )
    return PagedKVCache(**kwargs), QuantizedPagedKVCache(**kwargs)


def test_quantized_cache_is_a_drop_in_for_the_fp_cache():
    """Same public surface, so no attention code has to know which it has."""
    fp, q = _caches()
    for name in ("write", "gather", "copy_block", "nbytes", "block_size",
                 "n_kv_heads", "head_dim", "num_blocks", "dtype"):
        assert hasattr(q, name), f"QuantizedPagedKVCache is missing {name}"
    assert isinstance(q, PagedKVCache), "must substitute wherever the base is used"


def test_quantized_cache_uses_roughly_half_the_bytes():
    fp, q = _caches()
    ratio = fp.nbytes / q.nbytes
    # int8 payload plus one fp16 scale per (token, head): at head_dim=64 that is
    # 66 bytes against 256 fp32 bytes. Against an fp16 cache it would be ~1.94x,
    # which is the number to quote on a GPU.
    assert ratio > 3.5, ratio
    fp16, q16 = (
        PagedKVCache(N_BLOCKS, BLOCK, HEADS, DIM, dtype=torch.float16),
        QuantizedPagedKVCache(N_BLOCKS, BLOCK, HEADS, DIM, dtype=torch.float16),
    )
    assert 1.8 < fp16.nbytes / q16.nbytes < 2.0


def test_write_then_gather_recovers_values_within_int8_resolution():
    """Round-trip error is bounded by the two terms it actually has.

        |error|  <=  s/2  +  |q| * |s - fp16(s)|,      s = absmax / 127

    The first term is int8 round-to-nearest. The second is easy to forget and is
    what makes an s/2-only bound fail: the scale is *stored* in fp16, so the
    reconstruction multiplies the quantized integer by a slightly different scale
    than the one it was divided by. That error scales with |q|, so it is largest
    exactly where the payload is largest -- at |q| = 127 it reaches ~6% of s,
    about 12% of the int8 term.

    Worth knowing rather than hiding behind a looser constant: it says the fp16
    scale is a real but minor contributor, so widening it to fp32 would buy ~10%
    accuracy for 2 extra bytes per (token, head). Not taken, but a deliberate
    choice rather than an unexamined one.
    """
    torch.manual_seed(0)
    _fp, q = _caches()
    k = torch.randn(BLOCK, HEADS, DIM)
    v = torch.randn(BLOCK, HEADS, DIM)
    q.write([0], 0, k, v)

    gk, gv = q.gather([0], BLOCK)
    for got, want in ((gk, k), (gv, v)):
        s = want.abs().amax(dim=-1, keepdim=True) / 127.0
        q_int = torch.round(want / s).clamp(-127, 127).abs()
        scale_err = (s - s.to(torch.float16).float()).abs()
        bound = s / 2 + q_int * scale_err + 1e-7
        assert torch.all((got - want).abs() <= bound)


def test_attention_output_tracks_the_fp_cache():
    """The claim that matters: quantized storage barely moves attention output."""
    torch.manual_seed(0)
    fp, q = _caches()
    ctx = 48
    k = torch.randn(ctx, HEADS, DIM)
    v = torch.randn(ctx, HEADS, DIM)
    blocks = [0, 1, 2]
    fp.write(blocks, 0, k, v)
    q.write(blocks, 0, k, v)

    query = torch.randn(4, DIM)   # 4 query heads over 2 KV heads (GQA, n_rep=2)
    out_fp = paged_attention_decode(query, fp, blocks, ctx)
    out_q = paged_attention_decode(query, q, blocks, ctx)

    rel = ((out_q - out_fp).pow(2).mean() / out_fp.pow(2).mean()).sqrt()
    assert rel < 0.02, f"relative RMS {rel:.4f} -- int8 KV should be near-lossless here"


def test_copy_block_moves_the_scales():
    """Copy-on-write must copy magnitudes, not just quantized payload.

    Without the scale copy the destination decodes the source's integers against
    whatever magnitudes it happened to hold -- wrong numbers, no error, and every
    allocator invariant still satisfied.
    """
    torch.manual_seed(0)
    _fp, q = _caches()
    small = torch.randn(BLOCK, HEADS, DIM) * 0.001
    large = torch.randn(BLOCK, HEADS, DIM) * 1000.0
    q.write([0], 0, large, large)
    q.write([1], 0, small, small)

    q.copy_block(0, 1)
    a, _ = q.gather([0], BLOCK)
    b, _ = q.gather([1], BLOCK)
    assert torch.allclose(a, b), "copy_block did not reproduce the source block"
    # And specifically: the magnitude came across, not just the integers.
    assert b.abs().max() > 100.0, "the destination kept its own (tiny) scales"


def test_partial_block_write_leaves_untouched_positions_alone():
    """Chunked prefill writes into the middle of a block; neighbours must not move."""
    torch.manual_seed(0)
    _fp, q = _caches()
    first = torch.randn(4, HEADS, DIM)
    q.write([0], 0, first, first)
    snapshot, _ = q.gather([0], 4)

    second = torch.randn(4, HEADS, DIM) * 50.0
    q.write([0], 4, second, second)
    again, _ = q.gather([0], 4)
    assert torch.equal(snapshot, again), "an earlier chunk's KV was disturbed"


def test_out_of_range_position_raises_like_the_fp_cache():
    _fp, q = _caches()
    k = torch.randn(BLOCK, HEADS, DIM)
    with pytest.raises(IndexError, match="logical block"):
        q.write([0], BLOCK * 2, k, k)


def test_allocator_sizing_accounts_for_the_scales():
    """The engine divides its budget by bytes_per_block; the divisor must be right.

    An understated divisor hands out more blocks than the cache physically has,
    which surfaces as an IndexError deep in a forward pass under load rather than
    as a sizing error at startup.
    """
    args = dict(block_size=BLOCK, n_kv_heads=HEADS, head_dim=DIM, n_layers=4)
    unquantized = Allocator.bytes_per_block(dtype_bytes=2, **args)
    quantized = Allocator.bytes_per_block(dtype_bytes=1, scale_bytes_per_token=2, **args)

    q = QuantizedPagedKVCache(N_BLOCKS, BLOCK, HEADS, DIM, dtype=torch.float16)
    assert quantized == q.nbytes // N_BLOCKS * 4, "formula disagrees with the cache"
    assert quantized < unquantized

    # And the default still reproduces the spec's formula exactly.
    assert Allocator.bytes_per_block(dtype_bytes=2, **args) == 2 * 4 * BLOCK * HEADS * DIM * 2


def test_engine_with_quantized_kv_gets_more_blocks(quant_toy_dir):
    """The point of the whole exercise: same budget, more resident sequences."""
    from helios.engine import EngineConfig, LLMEngine

    budget = 8 * 1024 * 1024
    plain = LLMEngine(EngineConfig(model_dir=quant_toy_dir, kv_cache_bytes=budget))
    quant = LLMEngine(
        EngineConfig(model_dir=quant_toy_dir, kv_cache_bytes=budget, quantize_kv=True)
    )
    assert quant.num_blocks > plain.num_blocks * 3.0
    assert isinstance(quant.runner.kv_caches[0], QuantizedPagedKVCache)
    assert not isinstance(plain.runner.kv_caches[0], QuantizedPagedKVCache)


def test_engine_with_quantized_kv_still_generates(quant_toy_dir):
    from helios.core.types import SamplingParams
    from helios.engine import EngineConfig, LLMEngine

    engine = LLMEngine(
        EngineConfig(
            model_dir=quant_toy_dir, kv_cache_bytes=8 * 1024 * 1024,
            quantize_kv=True, max_model_len=256,
        )
    )
    for i in range(3):
        engine.add_request(
            f"r{i}", [3 + (i * 5 + j) % 200 for j in range(24)],
            SamplingParams(max_tokens=12, temperature=0.0),
        )
    outs = engine.run_until_complete(max_steps=2000)
    assert len(outs) == 3
    assert all(o.completion_tokens == 12 for o in outs)

"""INT4 weight quantization tests (spec section 8.2).

The claims being pinned:

  * packing is exactly invertible -- a lossy pack would be indistinguishable
    from a lossy quantizer, so the two have to be separated;
  * the group-wise quantizer's error is bounded by half a step, which is the
    definition of round-to-nearest and the thing a broken scale silently breaks;
  * AWQ's activation-aware scaling reduces *output* error versus RTN on the same
    weights. This is the whole justification for the extra machinery, so it is
    asserted rather than assumed;
  * the identity (x/s) @ (W*s)^T == x @ W^T holds exactly, which is why scaling
    is free in exact arithmetic;
  * memory actually goes down, in bytes, measured.

Per spec section 19.2, no tolerance here is loosened to make a test pass.
"""

from __future__ import annotations

import pytest
import torch

from helios.exec.quant import (
    QuantConfig,
    QuantLinear,
    awq_channel_scales,
    dequantize_weight,
    pack_int4,
    quantization_error,
    quantize_model,
    quantize_weight,
    unpack_int4,
)


@pytest.fixture(scope="module")
def cfg():
    return QuantConfig(group_size=128)


# ------------------------------------------------------------------- packing


@pytest.mark.parametrize("shape", [(1, 2), (4, 8), (7, 128), (33, 256)])
def test_pack_roundtrip_is_exact(shape):
    torch.manual_seed(0)
    q = torch.randint(0, 16, shape, dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


def test_pack_uses_exactly_four_bits_per_value():
    q = torch.randint(0, 16, (64, 256), dtype=torch.uint8)
    packed = pack_int4(q)
    bits = packed.numel() * packed.element_size() * 8
    assert bits == q.numel() * 4


def test_pack_rejects_odd_last_dimension():
    with pytest.raises(ValueError, match="odd last dimension"):
        pack_int4(torch.zeros((2, 5), dtype=torch.uint8))


def test_pack_boundary_values_survive():
    """0 and 15 are the nibble extremes -- an off-by-one mask loses one of them."""
    q = torch.tensor([[0, 15, 15, 0, 8, 7, 1, 14]], dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


# -------------------------------------------------------------- quantization


def test_quantization_error_is_within_half_a_step(cfg):
    """Round-to-nearest means |error| <= scale/2 for every element.

    The bound is per group, and it is what distinguishes a correct quantizer from
    one whose scales are merely plausible: a scale that is too large still
    produces small-looking relative error while wasting levels.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    packed, scales, zeros = quantize_weight(w, cfg)
    deq = dequantize_weight(packed, scales, zeros, 256)

    gs = 256 // scales.shape[1]
    err = (deq - w).abs().reshape(64, scales.shape[1], gs)
    bound = scales.unsqueeze(-1) / 2
    # A small epsilon for the float rounding in the divide/round/multiply chain,
    # not a tolerance on the claim itself.
    assert torch.all(err <= bound + 1e-6)


def test_degenerate_group_does_not_produce_nan(cfg):
    """An all-equal group has zero range; the scale clamp must handle it."""
    w = torch.zeros(8, 128)
    w[3] = 1.5  # one constant-but-nonzero row
    packed, scales, zeros = quantize_weight(w, cfg)
    deq = dequantize_weight(packed, scales, zeros, 128)
    assert torch.isfinite(deq).all()
    assert torch.allclose(deq[0], torch.zeros(128))


def test_compression_beats_three_and_a_half_times_versus_fp16(cfg):
    """4 bits plus per-group scale/zero overhead, against fp16 weights.

    The honest ratio to quote: 4-bit against fp16 is what "W4A16" means. Against
    fp32 the number is twice as large and twice as misleading.
    """
    torch.manual_seed(0)
    w = torch.randn(512, 1024, dtype=torch.float16)
    r = quantization_error(w, cfg)
    assert r["compression"] > 3.5, r
    assert r["compression"] < 4.0, "cannot beat 4x -- the overhead is real"


def test_larger_groups_compress_more_and_cost_accuracy():
    """The group-size trade-off, stated as a test rather than as a comment."""
    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=torch.float16)
    small = quantization_error(w, QuantConfig(group_size=32))
    large = quantization_error(w, QuantConfig(group_size=256))
    assert large["compression"] > small["compression"]
    assert large["rel_rmse"] > small["rel_rmse"]


# --------------------------------------------------------------------- AWQ


def test_awq_scaling_identity_is_exact_in_full_precision():
    """(x / s) @ (W * s)^T == x @ W^T, so scaling only moves quantization error.

    If this identity did not hold, AWQ would be changing the function being
    computed rather than changing where its error lands.
    """
    torch.manual_seed(0)
    x = torch.randn(16, 128, dtype=torch.float64)
    w = torch.randn(32, 128, dtype=torch.float64)
    s = torch.rand(128, dtype=torch.float64) + 0.5

    lhs = (x / s) @ (w * s).T
    assert torch.allclose(lhs, x @ w.T, atol=1e-12, rtol=1e-12)


def test_awq_reduces_output_error_versus_rtn_with_salient_channels(cfg):
    """The claim AWQ exists to make.

    Constructed with a few high-magnitude activation channels, which is the
    regime AWQ was designed for and the one real transformers exhibit (a handful
    of channels carry outsized activations).
    """
    torch.manual_seed(0)
    w = torch.randn(128, 256) * 0.02
    x = torch.randn(64, 256)
    x[:, :8] *= 20.0

    scale, err_awq, err_rtn = awq_channel_scales(w, x, cfg)
    assert err_awq < err_rtn, f"AWQ {err_awq} did not beat RTN {err_rtn}"
    assert scale.shape == (256,)
    assert torch.all(scale > 0), "a non-positive scale would flip signs on divide"


def test_awq_never_loses_to_rtn(cfg):
    """Even with no salient channels, the search includes alpha=0 (plain RTN).

    So AWQ is never worse -- it can only decline to scale. A search that could
    return something worse than its own starting point would be a bug in the
    search, not a property of the method.
    """
    torch.manual_seed(1)
    w = torch.randn(64, 128) * 0.02
    x = torch.randn(32, 128)   # uniform magnitudes: nothing to exploit
    _scale, err_awq, err_rtn = awq_channel_scales(w, x, cfg)
    assert err_awq <= err_rtn + 1e-12


# ------------------------------------------------------------- QuantLinear


def test_quant_linear_matches_its_own_dequantized_weight(cfg):
    """The forward pass must use exactly the weights the layer claims to hold."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 64, bias=False)
    ql = QuantLinear.from_linear(lin, cfg)
    x = torch.randn(8, 256)
    assert torch.allclose(ql(x), x @ ql.dequantized().T, atol=1e-6)


def test_quant_linear_preserves_bias_exactly(cfg):
    """Biases are not quantized: they are one vector, and Qwen2 has them.

    Dropping them here would reproduce the exact GPU bug from docs/GPU.md in a
    new place.
    """
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 32, bias=True)
    ql = QuantLinear.from_linear(lin, cfg)
    assert torch.equal(ql.bias, lin.bias)


def test_quant_linear_output_tracks_full_precision(cfg):
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 128, bias=False)
    x = torch.randn(32, 256)
    ref = lin(x)
    got = QuantLinear.from_linear(lin, cfg)(x)
    rel = ((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()
    # 4-bit with 128-wide groups on Gaussian weights: a few percent RMS. Asserted
    # as an upper bound so a regression that doubles the error fails here.
    assert rel < 0.10, f"relative RMS error {rel:.4f} is too large for 4-bit"


def test_quant_linear_stores_fewer_bytes(cfg):
    lin = torch.nn.Linear(1024, 512, bias=False).to(torch.float16)
    ql = QuantLinear.from_linear(lin, cfg)
    fp = lin.weight.numel() * lin.weight.element_size()
    assert ql.stored_bytes() < fp / 3.5


def test_quantization_still_saves_memory_after_running(cfg):
    """The claim has to survive USE, not just construction.

    This is the regression test for a real bug: `dequantized()` memoised the fp
    weight, so after one forward pass the layer held the packed int4 weights AND a
    full fp copy -- 1.26x MORE memory than the fp16 Linear it replaced, while
    `stored_bytes()` cheerfully reported a 3.8x saving. Every quantization memory
    figure in the repo was false in practice.

    Asserts the invariant (resident memory stays below fp after real use) rather
    than the symptom (`_deq_cache is None`), because the invariant is what the
    documentation claims and it stays meaningful if the implementation changes.
    """
    lin = torch.nn.Linear(512, 512, bias=False).to(torch.float16)
    fp_bytes = lin.weight.numel() * lin.weight.element_size()
    ql = QuantLinear.from_linear(lin, cfg)

    x = torch.randn(4, 512, dtype=torch.float16)
    for _ in range(3):
        ql(x)

    assert ql.resident_bytes() < fp_bytes / 3.0, (
        f"after 3 forward passes the layer holds {ql.resident_bytes()} bytes "
        f"against {fp_bytes} for plain fp16 -- the memory saving was given back"
    )
    assert ql.resident_bytes() == ql.stored_bytes(), "something is being cached"


def test_opt_in_cache_is_reported_as_resident(cfg):
    """Caching is allowed, but it must show up in resident_bytes().

    The bug was not the cache; it was a cache that `stored_bytes()` did not
    account for. With caching on, resident must exceed stored -- otherwise the
    accounting is lying again, just in the other direction.
    """
    lin = torch.nn.Linear(512, 512, bias=False).to(torch.float16)
    ql = QuantLinear.from_linear(lin, cfg, cache_dequantized=True)
    assert ql.resident_bytes() == ql.stored_bytes()
    ql(torch.randn(2, 512, dtype=torch.float16))
    assert ql.resident_bytes() > ql.stored_bytes()


def test_caching_does_not_change_the_result(cfg):
    """Cached and uncached paths must be numerically identical."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 128, bias=True)
    x = torch.randn(8, 256)
    a = QuantLinear.from_linear(lin, cfg, cache_dequantized=False)
    b = QuantLinear.from_linear(lin, cfg, cache_dequantized=True)
    b(x)                                    # populate the cache
    assert torch.equal(a(x), b(x))


def test_quantized_model_report_counts_resident_not_packed(toy_model_factory):
    """A model-level compression figure must also survive use."""
    model = toy_model_factory()
    report = quantize_model(model, QuantConfig(group_size=64))
    before = report.total_quant_bytes

    from helios.exec.paged_attn import PagedKVCache

    ids = list(range(3, 35))
    caches = [
        PagedKVCache(4, 16, model.config.num_key_value_heads, model.config.head_dim)
        for _ in range(model.config.num_hidden_layers)
    ]
    with torch.inference_mode():
        model.forward(ids, list(range(len(ids))), caches, [0, 1, 2, 3], False, len(ids))

    after = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        m.resident_bytes() for m in model.modules() if isinstance(m, QuantLinear)
    )
    assert after == before, (
        f"resident bytes grew from {before} to {after} by running the model, so "
        "the reported compression was only true before the first forward pass"
    )


def test_device_move_invalidates_the_dequantized_cache(cfg):
    """A cached fp weight must not survive a .to() call.

    Asserts the invariant, not the symptom: the CPU-only equivalent of the GPU
    bug where a cached tensor stayed on the wrong device. Checked by dtype here
    because that is what is observable without a second device.
    """
    lin = torch.nn.Linear(128, 32, bias=False)
    # Caching is off by default, so it has to be requested for there to be a
    # cache to invalidate at all.
    ql = QuantLinear.from_linear(lin, cfg, cache_dequantized=True)
    ql(torch.randn(2, 128))                 # populate the cache
    assert ql._deq_cache is not None
    ql.to(torch.float64)
    assert ql._deq_cache is None, "the cache survived a dtype change"
    assert ql(torch.randn(2, 128, dtype=torch.float64)).dtype == torch.float64


# ------------------------------------------------------------ model surgery


def test_quantize_model_replaces_layers_and_keeps_lm_head(toy_model_factory):
    model = toy_model_factory()
    report = quantize_model(model, QuantConfig(group_size=64))

    assert report.n_layers_quantized > 0
    assert isinstance(model.lm_head, torch.nn.Linear), "lm_head must stay fp"
    assert not isinstance(model.lm_head, QuantLinear)
    quantized = [m for m in model.modules() if isinstance(m, QuantLinear)]
    assert len(quantized) == report.n_layers_quantized
    assert report.weight_compression > 3.5
    assert report.model_compression > 1.0


def test_quantized_model_still_runs_and_produces_finite_logits(toy_model_factory):
    """The end-to-end check that surgery did not break the forward pass."""
    from helios.exec.paged_attn import PagedKVCache

    model = toy_model_factory()
    quantize_model(model, QuantConfig(group_size=64))

    ids = list(range(3, 35))
    caches = [
        PagedKVCache(4, 16, model.config.num_key_value_heads, model.config.head_dim)
        for _ in range(model.config.num_hidden_layers)
    ]
    with torch.inference_mode():
        logits = model.forward(ids, list(range(len(ids))), caches, [0, 1, 2, 3],
                               False, len(ids))
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, model.config.vocab_size)


def test_awq_calibration_changes_the_result(toy_model_factory):
    """Calibrated and uncalibrated quantization must not produce the same model.

    A silent failure to apply the searched scales would leave every AWQ claim in
    this repo false while every other test still passed.
    """
    rtn = toy_model_factory()
    awq = toy_model_factory()
    calib = [[3 + (i * 7) % 200 for i in range(48)]]

    quantize_model(rtn, QuantConfig(group_size=64))
    report = quantize_model(awq, QuantConfig(group_size=64), calib_token_ids=calib)

    assert report.awq_used
    scales = [m.act_scale for m in awq.modules() if isinstance(m, QuantLinear)]
    assert any(s is not None for s in scales)
    a = next(m for m in rtn.modules() if isinstance(m, QuantLinear))
    b = next(m for m in awq.modules() if isinstance(m, QuantLinear))
    assert not torch.equal(a.qweight, b.qweight)

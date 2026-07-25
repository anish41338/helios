"""W4A16 weight quantization: INT4 packing, RTN, and AWQ activation-aware scaling.

Spec section 8.2. This is the piece docs/SCOPE.md previously listed as *not
built* on the grounds that "quantized kernels need a GPU to be worth anything".
That reasoning conflated two separable claims:

  * a *speed* claim -- W4A16 is faster because decode is memory-bandwidth-bound
    and 4-bit weights are a quarter of the bytes. This genuinely needs a GPU.
    On CPU, dequantize-then-GEMM is strictly SLOWER than an fp32 GEMM: the
    dequantization is real work and there is no HBM bottleneck to relieve.
  * a *numerical* claim -- that 4-bit weights preserve the model's output
    distribution well enough to be usable, and that AWQ's activation-aware
    scaling beats round-to-nearest. This is pure arithmetic. It is
    device-independent and fully measurable here.

The second claim is also the one the whole QASSD idea rests on (spec section 7):
if int4 drafting does not agree with fp16 verification often enough, the
speculation has no speedup available to it at any bandwidth. Section 14 makes
that a kill gate at alpha < 0.6. So this module exists to make that gate
measurable without a GPU -- see `bench/measure_alpha.py`.

Storage layout, chosen for simplicity over kernel-friendliness:

    qweight  uint8  [out_features, in_features // 2]   two nibbles per byte
    scales   fp     [out_features, n_groups]
    qzeros   uint8  [out_features, n_groups]
    act_scale fp    [in_features]  or None

`in_features` is grouped into `n_groups = in_features // group_size` runs that
each get their own scale/zero. Dequantization yields exactly nn.Linear's
[out, in] layout, so `F.linear` consumes it with no transpose.

uint8 rather than the int32-packed layout AWQ ships: packing eight nibbles into
a *signed* int32 puts a value in the sign bit, and the conversion is then
relying on wrap-around semantics that PyTorch does not promise. Two nibbles in a
byte is exactly 4 bits per weight with no representational cliff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class QuantConfig:
    """How to quantize. Defaults match AWQ's published 4-bit setup."""

    bits: int = 4
    group_size: int = 128
    # Modules whose name ends with one of these is left in full precision.
    # lm_head is excluded by default and this is not a shortcut: it is the only
    # layer whose output IS the sampled distribution, so its error is not
    # attenuated by any downstream layer, and it is a single matrix rather than
    # one per block. AWQ, GPTQ and llama.cpp all keep it (or most of it) high
    # precision for the same reason.
    skip_suffixes: Tuple[str, ...] = ("lm_head",)
    # AWQ grid resolution for the per-channel scale exponent search.
    awq_grid: int = 20

    def __post_init__(self) -> None:
        if self.bits != 4:
            raise ValueError(f"only 4-bit is implemented, got bits={self.bits}")
        if self.group_size <= 0 or self.group_size % 2 != 0:
            raise ValueError(f"group_size must be a positive even number, got {self.group_size}")

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1     # 15 for 4-bit


# ------------------------------------------------------------------- packing


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack a [..., n] tensor of 0..15 values into [..., n//2] uint8.

    Even indices go to the low nibble, odd to the high nibble. `n` must be even,
    which the group_size constraint guarantees.
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"cannot pack an odd last dimension: {q.shape[-1]}")
    q = q.to(torch.uint8)
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4. Returns [..., n*2] uint8 values in 0..15."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


# -------------------------------------------------------------- quantization


def quantize_weight(
    weight: torch.Tensor,
    cfg: QuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric group-wise round-to-nearest quantization of [out, in] weights.

    Returns (packed_qweight, scales, qzeros).

    Asymmetric (with a zero point) rather than symmetric: at 4 bits there are
    only 16 levels, and weight distributions per group are not centred, so a
    symmetric range wastes levels on a tail that is not there. The zero point
    costs one extra byte per group per output channel -- 0.06 bits per weight.
    """
    out_features, in_features = weight.shape
    gs = cfg.group_size
    if in_features % gs != 0:
        # Fall back to a single group rather than silently mis-grouping. Real
        # checkpoints have power-of-two hidden sizes, so this is for toy models.
        gs = in_features
    n_groups = in_features // gs

    w = weight.detach().float().reshape(out_features, n_groups, gs)

    w_min = w.amin(dim=-1, keepdim=True)
    w_max = w.amax(dim=-1, keepdim=True)
    # Guard a degenerate group (all-equal weights, e.g. a zero-initialised row):
    # scale 0 would produce NaN on the divide.
    scales = ((w_max - w_min) / cfg.qmax).clamp(min=1e-8)
    zeros = torch.round(-w_min / scales).clamp(0, cfg.qmax)

    q = torch.round(w / scales + zeros).clamp(0, cfg.qmax)

    q = q.reshape(out_features, in_features)
    return (
        pack_int4(q),
        scales.reshape(out_features, n_groups).to(weight.dtype),
        zeros.reshape(out_features, n_groups).to(torch.uint8),
    )


def dequantize_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    in_features: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct [out, in] weights from the packed form."""
    out_features, n_groups = scales.shape
    gs = in_features // n_groups

    q = unpack_int4(packed).reshape(out_features, n_groups, gs).to(torch.float32)
    s = scales.float().unsqueeze(-1)
    z = qzeros.float().unsqueeze(-1)
    w = (q - z) * s
    return w.reshape(out_features, in_features).to(dtype)


def quantization_error(weight: torch.Tensor, cfg: QuantConfig) -> Dict[str, float]:
    """Round-trip error for one weight matrix. Used by tests and reports."""
    packed, scales, zeros = quantize_weight(weight, cfg)
    deq = dequantize_weight(packed, scales, zeros, weight.shape[1], weight.dtype)
    err = (deq.float() - weight.float())
    denom = weight.float().pow(2).mean().clamp(min=1e-12)
    return {
        "rel_rmse": float((err.pow(2).mean() / denom).sqrt()),
        "max_abs_err": float(err.abs().max()),
        "compression": weight.numel() * weight.element_size() / _packed_bytes(
            packed, scales, zeros
        ),
    }


def _packed_bytes(packed, scales, qzeros) -> int:
    return (
        packed.numel() * packed.element_size()
        + scales.numel() * scales.element_size()
        + qzeros.numel() * qzeros.element_size()
    )


# ------------------------------------------------------------------- modules


class QuantLinear(torch.nn.Module):
    """A 4-bit linear layer. Dequantizes, then calls F.linear.

    Dequantizing on every forward is the honest CPU implementation: there is no
    int4 GEMM here, and materialising the fp weight is what a fused GPU kernel
    would do in registers instead of in memory. The consequence is that this
    layer is *slower* than the fp32 Linear it replaces on CPU while using ~4x
    less parameter memory. Both halves of that are measured, not assumed --
    see tests/quant/test_quant.py.

    `act_scale` implements AWQ's channel scaling without rewriting the
    surrounding architecture. AWQ folds 1/s into the preceding op; keeping it
    explicit here costs one elementwise divide on the activation (negligible
    next to the GEMM) and keeps the layer a drop-in replacement.

    The identity being exploited:  (x / s) @ (W * s)^T == x @ W^T
    exactly, for any positive s -- so scaling is free in exact arithmetic and
    only changes where the quantization error lands.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: QuantConfig,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg
        gs = cfg.group_size if in_features % cfg.group_size == 0 else in_features
        n_groups = in_features // gs

        self.register_buffer(
            "qweight",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales", torch.ones((out_features, n_groups), dtype=dtype, device=device)
        )
        self.register_buffer(
            "qzeros", torch.zeros((out_features, n_groups), dtype=torch.uint8, device=device)
        )
        self.register_buffer("act_scale", None)
        self.bias = (
            torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
            if bias
            else None
        )
        self._deq_cache: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        cfg: QuantConfig,
        act_scale: Optional[torch.Tensor] = None,
    ) -> "QuantLinear":
        """Quantize an existing fp Linear, optionally with AWQ channel scales."""
        out_features, in_features = linear.weight.shape
        q = cls(
            in_features,
            out_features,
            cfg,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=str(linear.weight.device),
        )
        w = linear.weight.detach()
        dtype = w.dtype
        if act_scale is not None:
            act_scale = act_scale.to(device=w.device, dtype=torch.float32)
            w = (w.float() * act_scale).to(dtype)
            q.act_scale = act_scale.to(dtype)

        packed, scales, zeros = quantize_weight(w, cfg)
        q.qweight.copy_(packed)
        q.scales.copy_(scales.to(q.scales.dtype))
        q.qzeros.copy_(zeros)
        if linear.bias is not None:
            with torch.no_grad():
                q.bias.copy_(linear.bias.detach())
        return q

    def dequantized(self) -> torch.Tensor:
        """The fp weight this layer represents, including the AWQ scaling.

        Cached: the buffers are frozen after load, so recomputing per forward is
        pure waste. `torch.inference_mode` means no autograd graph is held.
        """
        if self._deq_cache is None:
            self._deq_cache = dequantize_weight(
                self.qweight, self.scales, self.qzeros, self.in_features,
                self.scales.dtype,
            )
        return self._deq_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_scale is not None:
            x = x / self.act_scale
        return F.linear(x, self.dequantized(), self.bias)

    def _apply(self, *args, **kwargs):
        # Any device/dtype move invalidates the cached fp weight. Without this,
        # `model.to("cuda")` would leave a CPU tensor cached and the next forward
        # would fail with a device mismatch -- exactly the bug class that cost a
        # GPU session already (see docs/GPU.md).
        self._deq_cache = None
        return super()._apply(*args, **kwargs)

    def stored_bytes(self) -> int:
        n = _packed_bytes(self.qweight, self.scales, self.qzeros)
        if self.act_scale is not None:
            n += self.act_scale.numel() * self.act_scale.element_size()
        if self.bias is not None:
            n += self.bias.numel() * self.bias.element_size()
        return n

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, bits=4, "
            f"group={self.cfg.group_size}, awq={self.act_scale is not None}"
        )


# ----------------------------------------------------------------- AWQ search


def awq_channel_scales(
    weight: torch.Tensor,
    x: torch.Tensor,
    cfg: QuantConfig,
) -> Tuple[torch.Tensor, float, float]:
    """Search AWQ's per-input-channel scale exponent.

    weight: [out, in]. x: [n_samples, in] calibration activations.
    Returns (best_scale, err_with_scale, err_rtn).

    The AWQ observation: quantization error on an input channel is weighted by
    that channel's activation magnitude when it reaches the output. Scaling a
    salient channel's weights UP before quantizing gives it proportionally more
    of the 16 available levels; the matching divide on the activation cancels it
    exactly. Channels that get scaled down are the ones whose activations were
    small, so their inflated weight error contributes little to the output.

    Only the exponent is searched, over s = act_absmean ** alpha for alpha in a
    grid on [0, 1] -- alpha=0 is plain RTN, alpha=1 is full activation scaling.
    This is AWQ's own parameterisation; the point is that the optimum is usually
    interior, which is why a search beats either endpoint.
    """
    w = weight.detach().float()
    x = x.detach().float()
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    ref = x @ w.T
    denom = ref.pow(2).mean().clamp(min=1e-12)

    act_absmean = x.abs().mean(dim=0).clamp(min=1e-5)

    def loss_for(scale: Optional[torch.Tensor]) -> float:
        ws = w * scale if scale is not None else w
        packed, scales, zeros = quantize_weight(ws, cfg)
        deq = dequantize_weight(packed, scales, zeros, w.shape[1], torch.float32)
        xs = x / scale if scale is not None else x
        return float(((xs @ deq.T - ref).pow(2).mean() / denom))

    err_rtn = loss_for(None)
    best_scale: Optional[torch.Tensor] = None
    best_err = err_rtn

    for i in range(1, cfg.awq_grid + 1):
        alpha = i / cfg.awq_grid
        s = act_absmean.pow(alpha)
        # Normalise so the geometric mean of s is ~1. Without this, alpha near 1
        # can push every scale far from unity, which shifts error into the
        # activation divide and into fp16 range rather than reducing it.
        s = s / (s.max() * s.min()).sqrt().clamp(min=1e-8)
        err = loss_for(s)
        if err < best_err:
            best_err, best_scale = err, s

    if best_scale is None:
        best_scale = torch.ones_like(act_absmean)
    return best_scale, best_err, err_rtn


# ------------------------------------------------------------- model surgery


def _target_linears(model: torch.nn.Module, cfg: QuantConfig) -> List[Tuple[str, torch.nn.Linear]]:
    out = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name.endswith(sfx) for sfx in cfg.skip_suffixes):
            continue
        out.append((name, module))
    return out


def _set_submodule(model: torch.nn.Module, name: str, new: torch.nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def collect_activations(
    model,
    calib_token_ids: List[List[int]],
    cfg: QuantConfig,
    max_samples_per_layer: int = 512,
) -> Dict[str, torch.Tensor]:
    """Run calibration prompts and capture each target Linear's input.

    Forward hooks rather than a rewritten forward pass: the activation a layer
    actually sees depends on every layer before it, including the paged KV path,
    so reconstructing it analytically would be a second implementation to keep
    in sync. Samples are subsampled per layer to bound memory -- AWQ's scale
    search only needs the per-channel magnitude profile, not every token.
    """
    from .paged_attn import PagedKVCache

    captured: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, inputs, _out):
            x = inputs[0].detach()
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            if x.shape[0] > max_samples_per_layer:
                # Deterministic stride, not a random sample: calibration must be
                # reproducible or the resulting quantized model is not.
                stride = x.shape[0] // max_samples_per_layer
                x = x[::stride][:max_samples_per_layer]
            buf = captured.setdefault(name, [])
            if sum(t.shape[0] for t in buf) < max_samples_per_layer:
                buf.append(x.float().cpu())
        return hook

    for name, module in _target_linears(model, cfg):
        handles.append(module.register_forward_hook(make_hook(name)))

    cfg_m = model.config
    block_size = 16
    try:
        with torch.inference_mode():
            for ids in calib_token_ids:
                n_blocks = (len(ids) + block_size - 1) // block_size
                caches = [
                    PagedKVCache(
                        num_blocks=n_blocks,
                        block_size=block_size,
                        n_kv_heads=cfg_m.num_key_value_heads,
                        head_dim=cfg_m.head_dim,
                        dtype=next(model.parameters()).dtype,
                        device=model.device,
                    )
                    for _ in range(cfg_m.num_hidden_layers)
                ]
                model.forward(
                    token_ids=list(ids),
                    positions=list(range(len(ids))),
                    kv_caches=caches,
                    block_ids=list(range(n_blocks)),
                    is_decode=False,
                    context_len=len(ids),
                )
    finally:
        for h in handles:
            h.remove()

    return {k: torch.cat(v, dim=0)[:max_samples_per_layer] for k, v in captured.items()}


@dataclass
class QuantReport:
    """What quantization actually did, per layer and in total."""

    n_layers_quantized: int
    fp_bytes: int
    quant_bytes: int
    total_fp_bytes: int
    total_quant_bytes: int
    awq_used: bool
    group_size: int
    per_layer: Dict[str, Dict[str, float]]

    @property
    def weight_compression(self) -> float:
        """Compression of the layers that were actually quantized."""
        return self.fp_bytes / max(1, self.quant_bytes)

    @property
    def model_compression(self) -> float:
        """Compression of the whole model, including the layers left in fp."""
        return self.total_fp_bytes / max(1, self.total_quant_bytes)

    def summary(self) -> str:
        return (
            f"quantized {self.n_layers_quantized} linears to 4-bit "
            f"(group={self.group_size}, awq={self.awq_used}): "
            f"{self.fp_bytes / 2**20:.1f} MiB -> {self.quant_bytes / 2**20:.1f} MiB "
            f"= {self.weight_compression:.2f}x on those layers, "
            f"{self.model_compression:.2f}x on the full model"
        )


def quantize_model(
    model,
    cfg: Optional[QuantConfig] = None,
    calib_token_ids: Optional[List[List[int]]] = None,
    verbose: bool = False,
) -> QuantReport:
    """Replace every eligible Linear in `model` with a QuantLinear, in place.

    With `calib_token_ids`, AWQ scales are searched per layer; without, this is
    plain RTN. Both are supported because the difference between them is one of
    the things worth measuring -- see tests/quant/test_quant.py, which asserts
    AWQ's output error is no worse than RTN's on the same weights.

    Mutates `model`. Callers that need both precisions (the QASSD draft path)
    should quantize a *copy*; see qassd.DualPrecisionModel.
    """
    cfg = cfg or QuantConfig()
    targets = _target_linears(model, cfg)

    acts: Dict[str, torch.Tensor] = {}
    if calib_token_ids:
        acts = collect_activations(model, calib_token_ids, cfg)

    total_fp = sum(p.numel() * p.element_size() for p in model.parameters())
    fp_bytes = 0
    quant_bytes = 0
    per_layer: Dict[str, Dict[str, float]] = {}

    for name, linear in targets:
        fp_bytes += linear.weight.numel() * linear.weight.element_size()

        scale = None
        entry: Dict[str, float] = {}
        if name in acts:
            scale, err_awq, err_rtn = awq_channel_scales(linear.weight, acts[name], cfg)
            entry["err_awq"] = err_awq
            entry["err_rtn"] = err_rtn

        q = QuantLinear.from_linear(linear, cfg, act_scale=scale)
        quant_bytes += q.stored_bytes()
        _set_submodule(model, name, q)
        per_layer[name] = entry
        if verbose:
            extra = (
                f"  rel_err rtn={entry['err_rtn']:.5f} awq={entry['err_awq']:.5f}"
                if entry else ""
            )
            print(f"  quantized {name}{extra}", flush=True)

    total_quant = sum(p.numel() * p.element_size() for p in model.parameters())
    total_quant += sum(
        m.stored_bytes() for m in model.modules() if isinstance(m, QuantLinear)
    )

    return QuantReport(
        n_layers_quantized=len(targets),
        fp_bytes=fp_bytes,
        quant_bytes=quant_bytes,
        total_fp_bytes=total_fp,
        total_quant_bytes=total_quant,
        awq_used=bool(acts),
        group_size=cfg.group_size,
        per_layer=per_layer,
    )

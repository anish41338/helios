"""Llama-family transformer with paged KV attention.

Spec section 8.1: GQA, RoPE, SwiGLU, RMSNorm -- the Llama-3.x / Qwen2.5
architecture. Weights load from HuggingFace safetensors via loader.py.

This is a from-scratch forward pass rather than a call into
transformers.LlamaModel, because the whole point is to own the KV cache: the
model must read and write paged blocks chosen by our allocator, which no stock
implementation exposes. Correctness is pinned by tests/parity, which compares
greedy output against HF `generate()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .paged_attn import (
    PagedKVCache,
    paged_attention_decode,
    paged_attention_prefill,
)


@dataclass
class ModelConfig:
    """Architecture hyperparameters, read from HF config.json."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_rep(self) -> int:
        """Query heads per KV head (GQA group size)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, cfg: dict) -> "ModelConfig":
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg.get(
                "num_key_value_heads", cfg["num_attention_heads"]
            ),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", 10000.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            torch_dtype=cfg.get("torch_dtype", "float32"),
        )


class RMSNorm(torch.nn.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias.

    Computed in fp32 regardless of parameter dtype: the reciprocal square root
    of a sum over hidden_size terms loses too much precision in fp16, and a
    norm discrepancy compounds through every layer. Matches HF's implementation
    so parity tests can hold to tight tolerances.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(dtype) * self.weight).to(dtype)


def build_rope_cache(
    max_pos: int, head_dim: int, theta: float, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary cos/sin tables, [max_pos, head_dim/2].

    RoPE encodes absolute position as a rotation, so the dot product between a
    query at position m and a key at position n depends only on (m - n) --
    relative position falls out of the geometry rather than being added as a
    bias term.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    pos = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Rotate query/key vectors by their position's angle.

    x: [n_tokens, n_heads, head_dim]. positions: [n_tokens].

    Uses the HF "rotate_half" convention -- the first and second halves of the
    head dimension form the (x, y) pairs -- NOT the interleaved convention from
    the original RoPE paper. Getting this wrong produces a model that still
    generates fluent-looking text while being subtly wrong, so it is pinned by
    parity tests rather than eyeballed.
    """
    c = cos[positions].unsqueeze(1)   # [n_tokens, 1, head_dim/2]
    s = sin[positions].unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    rot_x1 = x1 * c - x2 * s
    rot_x2 = x2 * c + x1 * s
    return torch.cat((rot_x1, rot_x2), dim=-1)


class Attention(torch.nn.Module):
    """Grouped-query attention reading and writing a paged KV cache."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = torch.nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden: torch.Tensor,          # [n_tokens, hidden_size]
        positions: torch.Tensor,       # [n_tokens]
        kv_cache: PagedKVCache,
        block_ids: List[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_decode: bool,
        context_len: int,
    ) -> torch.Tensor:
        n_tokens = hidden.shape[0]

        q = self.q_proj(hidden).view(n_tokens, self.n_heads, self.head_dim)
        k = self.k_proj(hidden).view(n_tokens, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(n_tokens, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        # Write this chunk's KV into the cache BEFORE attending, so a prefill
        # chunk can attend to itself and a decode step sees its own token.
        start_pos = int(positions[0].item())
        kv_cache.write(block_ids, start_pos, k, v)

        if is_decode:
            out = paged_attention_decode(
                q[0], kv_cache, block_ids, context_len, self.scale
            ).unsqueeze(0)
        else:
            out = paged_attention_prefill(
                q, kv_cache, block_ids, start_pos, context_len, self.scale
            )

        out = out.reshape(n_tokens, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MLP(torch.nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = torch.nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(torch.nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        block_ids: List[int],
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_decode: bool,
        context_len: int,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(
            hidden, positions, kv_cache, block_ids, cos, sin, is_decode, context_len
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class HeliosModel(torch.nn.Module):
    """The full causal LM, driven one sequence at a time by the runner.

    Sequences are processed individually rather than as a padded batch. That is
    a deliberate simplification of this build: correct paged KV handling per
    sequence comes first, and batched paged attention is a kernel-level
    optimisation the CPU fallback cannot exploit anyway. The scheduler already
    produces properly batched ExecSteps, so batching the executor later requires
    no scheduler change.
    """

    def __init__(self, config: ModelConfig, device: str = "cpu") -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        cos, sin = build_rope_cache(
            config.max_position_embeddings, config.head_dim, config.rope_theta, device
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        token_ids: List[int],
        positions: List[int],
        kv_caches: List[PagedKVCache],
        block_ids: List[int],
        is_decode: bool,
        context_len: int,
    ) -> torch.Tensor:
        """Run one sequence's chunk. Returns logits for the final position only.

        Only the last row of logits is returned because that is all sampling
        needs -- materialising [n_tokens, vocab_size] for a long prefill chunk
        is a large, pointless allocation (vocab is often 128k+).
        """
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        pos = torch.tensor(positions, dtype=torch.long, device=self.device)

        hidden = self.embed_tokens(tokens)
        for layer, kv_cache in zip(self.layers, kv_caches):
            hidden = layer(
                hidden,
                pos,
                kv_cache,
                block_ids,
                self.rope_cos,
                self.rope_sin,
                is_decode,
                context_len,
            )
        hidden = self.norm(hidden)
        return self.lm_head(hidden[-1:])   # [1, vocab_size]

    def forward_all_logits(
        self,
        token_ids: List[int],
        positions: List[int],
        kv_caches: List[PagedKVCache],
        block_ids: List[int],
        context_len: int,
    ) -> torch.Tensor:
        """Like forward() but returns every position's logits.

        Needed by speculative verification, which must score all gamma+1
        candidate positions in one pass.
        """
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        pos = torch.tensor(positions, dtype=torch.long, device=self.device)

        hidden = self.embed_tokens(tokens)
        for layer, kv_cache in zip(self.layers, kv_caches):
            hidden = layer(
                hidden, pos, kv_cache, block_ids, self.rope_cos, self.rope_sin,
                False, context_len,
            )
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

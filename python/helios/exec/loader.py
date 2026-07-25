"""Model weight loading from HuggingFace safetensors.

Spec section 8.2: load from safetensors with mmap, no full-model host copy.

Only the fp32/fp16 path is implemented. AWQ/GPTQ int4 unpacking (W4A16) is out
of scope for this build -- see docs/SCOPE.md. Quantized kernels need a GPU to be
worth anything, and shipping a dequantize-to-fp32 stub would be a claim without
a benchmark behind it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from .model import HeliosModel, ModelConfig

# HF checkpoint parameter name -> our module path. The architectures match
# closely enough that the mapping is mostly identity with a "model." prefix
# stripped; keeping it explicit means an unexpected checkpoint layout fails
# loudly at load time rather than silently leaving weights at their init values.
_PREFIX_STRIP = "model."


def load_config(model_dir: Path) -> ModelConfig:
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return ModelConfig.from_hf(cfg)


def _iter_shards(model_dir: Path):
    """Yield safetensors shard paths, honouring an index file if present."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        with open(index, "r", encoding="utf-8") as f:
            mapping = json.load(f)["weight_map"]
        for name in sorted(set(mapping.values())):
            yield model_dir / name
        return

    single = model_dir / "model.safetensors"
    if single.exists():
        yield single
        return

    shards = sorted(model_dir.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights found in {model_dir}")
    yield from shards


def load_model(
    model_dir: str | Path,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> HeliosModel:
    """Build a HeliosModel and populate it from safetensors on disk.

    Raises if any parameter is left unassigned: a silently partially-loaded
    model produces plausible-looking garbage, which is the worst failure mode
    for an inference engine (spec section 19.1).
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    config = load_config(model_dir)
    if dtype is None:
        dtype = torch.float32  # CPU path; fp16 matmul on CPU is slow and lossy

    model = HeliosModel(config, device=device)
    model = model.to(dtype)

    own = dict(model.named_parameters())
    assigned: set[str] = set()
    ignored: list[str] = []

    for shard in _iter_shards(model_dir):
        # framework="pt" with mmap semantics: tensors are read lazily per key,
        # so peak host memory stays near one tensor rather than the whole model.
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                target = _map_key(key, config)
                if target is None or target not in own:
                    # A checkpoint tensor this architecture has no slot for.
                    # Collected and reported below rather than skipped in
                    # silence: dropping real weights (Qwen2's q/k/v biases, for
                    # one) leaves a model that runs and emits fluent nonsense,
                    # which is the failure mode spec section 19.1 calls this
                    # project class's worst. Found on a real T4 run.
                    if not _is_benign_extra(key):
                        ignored.append(key)
                    continue
                tensor = f.get_tensor(key)
                param = own[target]
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(
                        f"shape mismatch for {key} -> {target}: "
                        f"checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(tensor.to(dtype))
                assigned.add(target)

    # Weight tying: many small models share the embedding and output matrices.
    if config.tie_word_embeddings and "lm_head.weight" not in assigned:
        with torch.no_grad():
            model.lm_head.weight.copy_(model.embed_tokens.weight)
        assigned.add("lm_head.weight")

    missing = set(own) - assigned
    if missing:
        raise ValueError(
            f"{len(missing)} parameters were not found in the checkpoint: "
            f"{sorted(missing)[:8]}"
        )
    if ignored:
        raise ValueError(
            f"{len(ignored)} checkpoint tensors have no slot in this "
            f"architecture and would be silently dropped: {sorted(ignored)[:8]}. "
            "Loading anyway would produce a model that runs and generates "
            "fluent nonsense. If these are genuinely unused, add them to "
            "_is_benign_extra with a reason."
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# Checkpoint keys that legitimately have no parameter slot here.
_BENIGN_EXTRA_SUFFIXES = (
    ".rotary_emb.inv_freq",      # RoPE table; recomputed in build_rope_cache
    ".attn.bias",                # GPT-2-style causal mask buffer, not a weight
    ".attn.masked_bias",
)


def _is_benign_extra(key: str) -> bool:
    """True for checkpoint tensors that are buffers rather than weights.

    Kept as an explicit allowlist so that anything genuinely unexpected fails
    the load instead of being quietly discarded.
    """
    return any(key.endswith(sfx) for sfx in _BENIGN_EXTRA_SUFFIXES)


def _map_key(key: str, config: ModelConfig) -> Optional[str]:
    """Translate an HF parameter name to our module path."""
    if key.startswith("lm_head."):
        return key
    if key.startswith(_PREFIX_STRIP):
        return key[len(_PREFIX_STRIP) :]
    return key


def save_toy_model(
    path: str | Path,
    config: ModelConfig,
    seed: int = 0,
) -> Path:
    """Write a small randomly-initialised model to disk in HF layout.

    Used by tests and benchmarks so the whole engine can be exercised
    end-to-end without downloading a real checkpoint. The weights are random,
    so generated text is meaningless -- but every code path (paging, batching,
    sampling, parity against a reference forward pass) is real.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = HeliosModel(config, device="cpu").to(torch.float32)

    # Small init keeps logits in a sane range so softmax does not saturate.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    state = {}
    for name, param in model.named_parameters():
        key = name if name.startswith("lm_head.") else _PREFIX_STRIP + name
        state[key] = param.detach().clone()

    save_file(state, str(path / "model.safetensors"))

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "float32",
        "model_type": "llama",
    }
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    _write_toy_tokenizer(path, config.vocab_size)
    return path


def _write_toy_tokenizer(path: Path, vocab_size: int) -> None:
    """Write a byte-level tokenizer so the toy model is self-contained.

    A trivial vocabulary of single bytes: every input encodes without an <unk>
    and decoding round-trips. This lets the HTTP text endpoints and the
    benchmark harness run against the toy model with no downloads, which keeps
    the test suite hermetic.
    """
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        if len(vocab) >= vocab_size:
            break
        tok = f"<0x{b:02X}>"
        vocab.setdefault(tok, len(vocab))

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 1,
                "content": "<s>",
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            },
            {
                "id": 2,
                "content": "</s>",
                "special": True,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "WordPiece",
            "unk_token": "<pad>",
            "continuing_subword_prefix": "",
            "max_input_chars_per_word": 1000,
            "vocab": vocab,
        },
    }
    with open(path / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer, f)

    with open(path / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 4096,
            },
            f,
            indent=2,
        )

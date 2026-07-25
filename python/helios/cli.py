"""HELIOS command line interface.

    python -m helios.cli serve   --model DIR [--port 8000]
    python -m helios.cli generate --model DIR --prompt "..."
    python -m helios.cli vopr    --seeds 10000 [--seed N --replay]
    python -m helios.cli make-toy-model --out DIR
    python -m helios.cli info    --model DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.server import create_app
    from .engine import EngineConfig

    app = create_app(
        EngineConfig(
            model_dir=args.model,
            kv_cache_bytes=args.kv_mb * 1024 * 1024,
            block_size=args.block_size,
            max_num_seqs=args.max_seqs,
            max_num_batched_tokens=args.max_batched_tokens,
            max_model_len=args.max_model_len,
            enable_prefix_cache=not args.no_prefix_cache,
            enable_chunked_prefill=not args.no_chunked_prefill,
            enable_spec_decode=args.spec_decode,
            spec_gamma=args.spec_gamma,
            quantize_kv=args.quantize_kv,
            quantized_draft=args.quantized_draft,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .core.types import SamplingParams
    from .engine import EngineConfig, LLMEngine

    engine = LLMEngine(
        EngineConfig(
            model_dir=args.model,
            kv_cache_bytes=args.kv_mb * 1024 * 1024,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            enable_spec_decode=args.spec_decode,
            spec_gamma=args.spec_gamma,
            quantize_kv=args.quantize_kv,
            quantized_draft=args.quantized_draft,
        )
    )
    if engine.dual is not None:
        print(engine.dual.report.summary())
        print(engine.dual.summary())
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        stop_token_ids=engine.eos_token_ids,
    )

    prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]
    for out in engine.generate(prompts, params):
        print(f"--- {out.request_id} ---")
        print(out.text or f"(token ids: {out.token_ids})")
        reason = out.finish_reason.value if out.finish_reason else "none"
        print(f"[{out.completion_tokens} tokens, finish={reason}]")
    return 0


def cmd_vopr(args: argparse.Namespace) -> int:
    """Run the deterministic simulation harness (spec section 10)."""
    from .core.vopr import replay, sweep

    if args.replay:
        result = replay(args.seed, max_steps=args.max_steps)
        return 0 if result.ok else 1

    print(
        f"running seeds [{args.start}, {args.start + args.seeds}) "
        f"max_steps={args.max_steps}"
    )
    passed, failures = sweep(
        start=args.start,
        count=args.seeds,
        max_steps=args.max_steps,
        stop_on_fail=args.stop_on_fail,
        progress_every=args.progress_every,
    )
    print(f"\n{passed}/{args.seeds} seeds passed, {len(failures)} failed")
    for f in failures[:20]:
        print(f.summary())
        for v in f.violations:
            print(f"    {v}")
        print(
            f"    reproduce: python -m helios.cli vopr --seed {f.seed} --replay"
        )

    if args.json_out:
        _write_vopr_artifact(args, passed, failures)
    return 0 if not failures else 1


def _write_vopr_artifact(args, passed: int, failures) -> None:
    """Record the sweep result with its provenance.

    Spec section 19.3: no claim without a reproducing command and a recorded
    result. Writing this from the CLI (rather than an ad-hoc script) is what
    makes the commit SHA reliably present -- a hand-rolled one-liner silently
    recorded "unknown" once, which is exactly the kind of gap that turns a
    documented number into an unverifiable one.
    """
    import json
    import platform
    import subprocess
    import time
    from pathlib import Path

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[2]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    record = {
        "seeds": args.seeds,
        "start_seed": args.start,
        "passed": passed,
        "failures": [f.summary() for f in failures],
        "max_steps": args.max_steps,
        "commit": commit,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": (
            f"python -m helios.cli vopr --seeds {args.seeds} "
            f"--start {args.start} --max-steps {args.max_steps}"
        ),
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def cmd_make_toy_model(args: argparse.Namespace) -> int:
    """Write a small random-weight model, for tests and benchmarks."""
    from .exec.loader import save_toy_model
    from .exec.model import ModelConfig

    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 2,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.max_pos,
    )
    path = save_toy_model(args.out, config, seed=args.seed)
    print(f"wrote toy model to {path}")
    print("  NOTE: random weights -- output is meaningless, but every engine")
    print("  code path (paging, batching, sampling, parity) is real.")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Report derived capacity for a model + KV budget."""
    from .core.allocator import Allocator
    from .exec.loader import load_config

    config = load_config(Path(args.model))
    per_block = Allocator.bytes_per_block(
        block_size=args.block_size,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        n_layers=config.num_hidden_layers,
        dtype_bytes=4,
    )
    budget = args.kv_mb * 1024 * 1024
    blocks = budget // per_block

    print(f"model: {args.model}")
    print(f"  layers={config.num_hidden_layers} hidden={config.hidden_size}")
    print(f"  q_heads={config.num_attention_heads} kv_heads={config.num_key_value_heads}"
          f" head_dim={config.head_dim} (GQA group={config.n_rep})")
    print(f"  vocab={config.vocab_size} max_pos={config.max_position_embeddings}")
    print(f"KV cache (fp32, block_size={args.block_size}):")
    print(f"  bytes/block = {per_block:,}")
    print(f"  budget {args.kv_mb} MiB -> {blocks:,} blocks"
          f" -> {blocks * args.block_size:,} tokens of KV")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="helios", description="HELIOS LLM serving engine")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_engine_args(p):
        p.add_argument("--model", required=True)
        p.add_argument("--kv-mb", type=int, default=512, help="KV cache budget in MiB")
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--max-model-len", type=int, default=None)
        p.add_argument(
            "--quantize-kv",
            action="store_true",
            help="INT8 paged KV cache: ~2x smaller, so ~2x the resident sequences "
                 "at the same budget (spec 7.4)",
        )
        p.add_argument(
            "--quantized-draft",
            action="store_true",
            help="QASSD: draft speculation from a 4-bit view of the same weights. "
                 "Requires --spec-decode; costs ~+50%% weight memory (spec 7.1)",
        )

    p = sub.add_parser("serve", help="run the OpenAI-compatible HTTP server")
    add_engine_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-seqs", type=int, default=64)
    p.add_argument("--max-batched-tokens", type=int, default=2048)
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--spec-decode", action="store_true")
    p.add_argument("--spec-gamma", type=int, default=4)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("generate", help="offline generation")
    add_engine_args(p)
    p.add_argument("--prompt", nargs="+", required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--spec-decode", action="store_true")
    p.add_argument("--spec-gamma", type=int, default=4)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("vopr", help="deterministic simulation testing")
    p.add_argument("--seeds", type=int, default=1000, help="how many seeds to run")
    p.add_argument("--start", type=int, default=0, help="first seed")
    p.add_argument("--seed", type=int, default=0, help="seed to replay")
    p.add_argument("--replay", action="store_true", help="replay one seed verbosely")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument(
        "--json-out",
        default=None,
        help="write the sweep result + provenance to this JSON path",
    )
    p.set_defaults(func=cmd_vopr)

    p = sub.add_parser("make-toy-model", help="write a small random-weight model")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=259)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--max-pos", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_make_toy_model)

    p = sub.add_parser("info", help="report derived KV capacity for a model")
    add_engine_args(p)
    p.set_defaults(func=cmd_info)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        # Configuration errors are the user's problem to fix, not a crash to
        # debug. A traceback here buries a one-line message under twenty lines of
        # frames from inside the engine.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

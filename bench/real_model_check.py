"""Run a real HuggingFace model through the engine and print what it generates.

Separated from kaggle_gpu_run.py because it was originally inlined as a
`python -c` one-liner, which swallowed the traceback and reported only
`IndexError: list index out of range` from indexing an empty result. A real
script surfaces the actual failure.

Coherent output here is the strongest single signal that the whole stack is
correct: paging, block tables, RoPE, GQA, RMSNorm, SwiGLU, and sampling all have
to be right simultaneously for a pretrained model to produce sensible text. The
toy model has random weights, so it can only ever demonstrate self-consistency.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from helios.core.types import SamplingParams  # noqa: E402
from helios.engine import EngineConfig, LLMEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv-mb", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="repeatable; defaults to a small factual set",
    )
    args = ap.parse_args()

    prompts = args.prompt or [
        "The capital of France is",
        "Water boils at",
        "def add(a, b):",
    ]

    dtype_note = "float32" if args.device == "cpu" else "float32 (see note)"
    print(f"loading {args.model} on {args.device} ({dtype_note}) ...", flush=True)
    t0 = time.perf_counter()
    engine = LLMEngine(
        EngineConfig(
            model_dir=args.model,
            device=args.device,
            kv_cache_bytes=args.kv_mb * 1024 * 1024,
            block_size=16,
            max_model_len=args.max_model_len,
        )
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  kv blocks: {engine.num_blocks}  max_model_len: {engine.max_model_len}")
    print(f"  params: {engine.runner.model.num_parameters():,}")

    eos = engine.eos_token_ids
    params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0, stop_token_ids=eos
    )

    failures = 0
    for text in prompts:
        ids = engine.tokenizer.encode(text)
        engine.add_request(f"p{len(ids)}-{abs(hash(text)) % 9999}", ids, params)

    t0 = time.perf_counter()
    outputs = engine.run_until_complete()
    elapsed = time.perf_counter() - t0

    if not outputs:
        print(
            "\nFAILED: the engine returned no completions. "
            "That is an engine bug, not a harness bug -- investigate before "
            "trusting any number from this device."
        )
        return 1

    total_out = 0
    print()
    for out in outputs:
        total_out += out.completion_tokens
        status = "ok" if out.completion_tokens else "EMPTY"
        print(f"[{status}] {out.completion_tokens} tok, finish={out.finish_reason.value}")
        print(f"   text: {out.text!r}")
        if not out.completion_tokens:
            failures += 1

    print(
        f"\n{len(outputs)} completions, {total_out} output tokens in "
        f"{elapsed:.2f}s = {total_out / max(1e-9, elapsed):.1f} tok/s"
    )
    if failures:
        print(f"FAILED: {failures} completion(s) were empty")
        return 1
    print(
        "\nRead the text above: if it is coherent, the paged KV path, RoPE, GQA, "
        "and sampling are jointly correct on this device."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

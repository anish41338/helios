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


def _smoke(engine, prompt: str) -> None:
    """Drive one prefill + one decode straight at the runner, unguarded.

    Deliberately bypasses the scheduler so that a broken forward pass raises
    here with its real traceback instead of being converted into an ExecFault
    and retried.
    """
    import torch

    from helios.core.execstep import DecodeItem, ExecStep, PrefillItem
    from helios.core.types import SamplingParams as SP

    ids = engine.tokenizer.encode(prompt)
    n_blocks = len(ids) // engine.config.block_size + 2
    blocks = list(range(n_blocks))
    sp = SP(max_tokens=2, temperature=0.0)

    step = ExecStep(
        step_id=1,
        prefills=[
            PrefillItem(
                seq_id=0, token_ids=list(ids), start_pos=0,
                block_ids=blocks, params=sp, is_last_chunk=True,
            )
        ],
    )
    out = engine.runner.run(step)
    if not out.outputs or not out.outputs[0].token_ids:
        raise RuntimeError("prefill produced no token")
    first = out.outputs[0].token_ids[0]
    print(f"    prefill ok -> first token id {first}")

    step2 = ExecStep(
        step_id=2,
        decodes=[
            DecodeItem(
                seq_id=0, last_token_id=first, position=len(ids),
                block_ids=blocks, params=sp, context_len=len(ids) + 1,
            )
        ],
    )
    out2 = engine.runner.run(step2)
    if not out2.outputs or not out2.outputs[0].token_ids:
        raise RuntimeError("decode produced no token")
    print(f"    decode ok  -> next token id {out2.outputs[0].token_ids[0]}")

    logits_finite = True
    with torch.inference_mode():
        lg = engine.runner.model.forward(
            list(ids), list(range(len(ids))), engine.runner.kv_caches,
            blocks, False, len(ids),
        )
        logits_finite = bool(torch.isfinite(lg).all().item())
    print(f"    logits all finite: {logits_finite}")
    if not logits_finite:
        raise RuntimeError(
            "logits contain NaN/Inf -- the model computed garbage, so sampling "
            "cannot produce a sensible token"
        )


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
    print(f"  eos_token_ids: {eos}")
    params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0, stop_token_ids=eos
    )

    # SMOKE TEST FIRST, with the fault handler bypassed.
    #
    # ModelRunner.run() converts RuntimeError/IndexError into an ExecFault so the
    # scheduler can preempt and retry rather than commit tokens computed from bad
    # KV. That is right for production and terrible for diagnosis: a forward pass
    # that always fails becomes an infinite preempt/retry loop that reports
    # nothing but "no completions". So the raw executor is driven once here and
    # the exception is allowed to propagate.
    banner_smoke = "  -- smoke test: one raw forward pass, exceptions unhandled --"
    print(banner_smoke)
    try:
        _smoke(engine, prompts[0])
        print("  smoke test OK")
    except Exception:
        import traceback

        print("  smoke test FAILED -- this is the real error:\n")
        traceback.print_exc()
        print(
            "\nThe scheduler would have swallowed this as an ExecFault and "
            "retried forever. Fix it before reading any number from this device."
        )
        return 1

    failures = 0
    for i, text in enumerate(prompts):
        ids = engine.tokenizer.encode(text)
        engine.add_request(f"req-{i}", ids, params)

    t0 = time.perf_counter()
    # Bounded so a stuck engine fails in seconds rather than grinding through
    # 100k no-op steps and reporting an empty list.
    outputs = engine.run_until_complete(max_steps=2000)
    elapsed = time.perf_counter() - t0

    if not outputs:
        stats = engine.scheduler.stats
        print(
            f"\nFAILED: no completions after {stats.step} steps.\n"
            f"  exec_faults        : {stats.exec_faults}\n"
            f"  preempt (recompute): {stats.preemptions_recompute}\n"
            f"  prefill tokens     : {stats.prefill_tokens}\n"
            f"  decode tokens      : {stats.decode_tokens}\n"
            f"  running / waiting  : {engine.scheduler.num_running()} / "
            f"{engine.scheduler.num_waiting()}\n"
        )
        if stats.exec_faults:
            print(
                "  exec_faults > 0: the executor is failing and the scheduler is "
                "retrying. The smoke test above should have caught it -- if it "
                "passed, the fault is specific to a batched or multi-sequence "
                "step."
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

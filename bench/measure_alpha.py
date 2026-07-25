"""Measure alpha for quantization-asymmetric speculation (spec section 14 gate).

This is a decision procedure, not a benchmark. Spec section 14 says: abandon
QASSD if the acceptance rate alpha < 0.6. Until now that gate was unmeasurable
here, because no quantized draft path existed and the symmetric self-speculation
in the runner has alpha = 1.0 by construction. It is measurable now.

Why the measurement is valid on a CPU with no int4 kernels
----------------------------------------------------------
alpha is the probability that the argmax of the int4 model's logits equals the
argmax of the fp model's logits, given the same context. That is a property of
two weight matrices and a prompt. It does not depend on how fast either forward
pass runs, so it transfers to a GPU unchanged.

What does NOT transfer is the speedup, which needs the draft's forward pass to
actually be cheaper -- that requires an int4 GEMM reading a quarter of the bytes,
and this build dequantizes to fp before a normal GEMM. So this script reports the
*modelled* speedup from measured alpha and a stated cost assumption, clearly
labelled as modelled. Spec section 19.6 forbids presenting it as measured.

Usage:

    python bench/measure_alpha.py --model artifacts/qwen05b --gamma 4
    python bench/measure_alpha.py --model artifacts/bench_model --toy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Prompts chosen to span the register the model will actually see: factual
# recall, arithmetic, code, and open-ended prose. Acceptance is strongly
# context-dependent -- a low-entropy continuation ("the capital of France is")
# is easy for a degraded model to match, while an open-ended one is not -- so a
# single prompt family would produce a number that does not generalise.
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Water boils at 100 degrees Celsius, which is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The three primary colors are",
    "In 1969, humans first landed on",
    "The derivative of x squared with respect to x is",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):",
    "Photosynthesis is the process by which plants",
]

# Calibration prompts for AWQ. Deliberately disjoint from the evaluation set:
# calibrating on the text you then measure on is how a quantization method gets
# reported as better than it is.
CALIB_PROMPTS = [
    "The history of computing begins with mechanical calculators.",
    "import numpy as np\n\ndef solve(a, b):\n    return np.linalg.solve(a, b)",
    "Machine learning models are trained on large datasets to recognise patterns.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--no-awq", action="store_true", help="plain RTN instead of AWQ")
    ap.add_argument(
        "--toy",
        action="store_true",
        help="model has random weights; report but do not treat alpha as meaningful",
    )
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args()

    import torch

    from helios.exec.loader import load_model
    from helios.exec.qassd import (
        DualPrecisionModel,
        acceptance_verdict,
        measure_acceptance,
        speculation_speedup_model,
    )
    from helios.exec.quant import QuantConfig

    model_dir = Path(args.model)
    print(f"loading {model_dir} on {args.device} ...", flush=True)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = load_model(model_dir, device=args.device, dtype=dtype)

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as exc:
        print(f"  no usable tokenizer ({exc}); falling back to synthetic token ids")

    def encode(text: str):
        if tokenizer is not None:
            return tokenizer.encode(text)
        # Byte ids offset past the specials, for the toy tokenizer.
        return [3 + (b % 200) for b in text.encode("utf-8")][:64]

    prompts = [encode(p) for p in DEFAULT_PROMPTS]
    calib = None if args.no_awq else [encode(p) for p in CALIB_PROMPTS]

    cfg = QuantConfig(group_size=args.group_size)
    print(
        f"building the int4 draft (group_size={args.group_size}, "
        f"awq={'off' if args.no_awq else 'on'}) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    dual = DualPrecisionModel(model, cfg, calib_token_ids=calib)
    print(f"  built in {time.perf_counter() - t0:.1f}s")
    print("  " + dual.report.summary())
    print("  " + dual.summary())

    # AWQ's own effect, aggregated: per-layer relative output error, RTN vs AWQ.
    per = [v for v in dual.report.per_layer.values() if "err_rtn" in v]
    awq_effect = None
    if per:
        mean_rtn = sum(v["err_rtn"] for v in per) / len(per)
        mean_awq = sum(v["err_awq"] for v in per) / len(per)
        awq_effect = {
            "mean_rel_output_err_rtn": mean_rtn,
            "mean_rel_output_err_awq": mean_awq,
            "error_reduction": mean_rtn / max(mean_awq, 1e-12),
        }
        print(
            f"  AWQ vs RTN mean per-layer relative output error: "
            f"{mean_rtn:.5f} -> {mean_awq:.5f} ({mean_rtn / max(mean_awq, 1e-12):.2f}x lower)"
        )

    print(f"\nmeasuring acceptance over {len(prompts)} prompts, gamma={args.gamma} ...",
          flush=True)
    t0 = time.perf_counter()
    stats = measure_acceptance(
        dual, prompts, gamma=args.gamma, max_new_tokens=args.max_new_tokens
    )
    elapsed = time.perf_counter() - t0

    verdict = acceptance_verdict(stats)
    print(f"  done in {elapsed:.1f}s over {stats.verify_passes} verify passes\n")
    print(f"  drafted            : {stats.drafted}")
    print(f"  accepted           : {stats.accepted}")
    print(f"  ALPHA              : {stats.alpha:.4f}")
    print(f"  tokens/verify pass : {stats.tokens_per_pass:.2f} (measured)")
    print(f"  accepted-run histogram: {verdict['run_lengths']}")

    print("\n  MODELLED speedup (not measured -- there is no int4 GEMM here):")
    for g in (1, 2, 4, 8):
        m = speculation_speedup_model(stats.alpha, g)
        print(
            f"    gamma={g}: E[tokens/pass]={m['expected_tokens_per_pass']:.2f} "
            f"cost={m['cost_per_pass']:.2f} -> {m['speedup']:.2f}x"
        )
    print(f"  best gamma at this alpha: {verdict['best_gamma']}")

    print(f"\n  SPEC SECTION 14 GATE (alpha >= {verdict['kill_threshold']}): ", end="")
    if args.toy:
        print("N/A")
        print(
            "  --toy was passed: the weights are random, so the fp and int4 models\n"
            "  agree or disagree for no reason connected to language. This run\n"
            "  exercises the code path only. Do not quote this alpha."
        )
    elif verdict["passes_gate"]:
        print(f"PASS ({stats.alpha:.4f})")
        print(
            "  Speculation is worth building on this model. The remaining\n"
            "  question is whether an int4 GEMM can realise the modelled cost\n"
            "  ratio, which needs a GPU to answer."
        )
    else:
        print(f"FAIL ({stats.alpha:.4f})")
        print(
            "  Spec section 14 says abandon the feature at this alpha rather than\n"
            "  tune it. Recording that decision, with the measurement behind it,\n"
            "  is the deliverable -- a negative result that was actually measured\n"
            "  is worth more than an unmeasured positive one."
        )

    record = {
        "model": str(model_dir),
        "device": args.device,
        "dtype": str(dtype),
        "gamma": args.gamma,
        "group_size": args.group_size,
        "awq": not args.no_awq,
        "toy": args.toy,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "elapsed_s": round(elapsed, 2),
        "quant_report": {
            "n_layers_quantized": dual.report.n_layers_quantized,
            "weight_compression": round(dual.report.weight_compression, 3),
            "model_compression": round(dual.report.model_compression, 3),
        },
        "memory": {k: round(v, 4) for k, v in dual.memory_overhead().items()},
        "awq_effect": awq_effect,
        "verdict": verdict,
        "modelled_speedup": {
            str(g): speculation_speedup_model(stats.alpha, g) for g in (1, 2, 4, 8)
        },
    }

    out_path = Path(args.out) if args.out else ROOT / "artifacts" / "alpha.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

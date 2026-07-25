"""Is the AWQ-versus-RTN difference in alpha real, or is it noise?

Exists because the naive comparison points the wrong way and it would be easy to
either bury that or over-read it:

    AWQ  alpha = 0.6548   (220 / 336)
    RTN  alpha = 0.6860   (225 / 328)

AWQ reduces mean per-layer output error by 1.58x, and yet drafts slightly *worse*.
Before drawing any conclusion from a 0.03 gap, it has to be established whether a
gap that size means anything at this sample size.

Why a plain binomial test would be wrong
----------------------------------------
Draft acceptances are **not independent**. The measured run-length histograms are
strongly bimodal -- about half of all verify passes accept the full draft and a
sixth accept nothing -- because a context the draft finds easy stays easy for
several tokens. Treating 336 tokens as 336 independent Bernoulli trials would
understate the variance, and the understatement is exactly in the direction that
manufactures significance.

The unit of independence is the **verify pass**, not the token. So this
bootstraps over passes, resampling whole accepted-run outcomes.

Run:
    python bench/alpha_significance.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GAMMA = 4
N_BOOT = 20_000
SEED = 12345


def load(path: Path):
    """Read a measure_alpha.py artifact into a list of per-pass accepted counts."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    hist = rec["verdict"]["run_lengths"]
    passes = []
    for k, count in hist.items():
        passes.extend([int(k)] * int(count))
    return rec, passes


def alpha_of(passes, gamma=GAMMA):
    if not passes:
        return 0.0
    return sum(passes) / (len(passes) * gamma)


def bootstrap_ci(passes, n_boot=N_BOOT, seed=SEED, level=0.95):
    """Percentile bootstrap CI for alpha, resampling verify passes."""
    rng = random.Random(seed)
    n = len(passes)
    draws = []
    for _ in range(n_boot):
        sample = [passes[rng.randrange(n)] for _ in range(n)]
        draws.append(alpha_of(sample))
    draws.sort()
    lo = draws[int(n_boot * (1 - level) / 2)]
    hi = draws[int(n_boot * (1 + level) / 2) - 1]
    return lo, hi


def permutation_p(a, b, n_boot=N_BOOT, seed=SEED):
    """Two-sided permutation test on the difference in alpha.

    Pools both sets of per-pass outcomes and reshuffles the group labels. Makes no
    distributional assumption, which is what is wanted when the within-group
    distribution is bimodal.
    """
    rng = random.Random(seed)
    observed = abs(alpha_of(a) - alpha_of(b))
    pool = list(a) + list(b)
    na = len(a)
    hits = 0
    for _ in range(n_boot):
        rng.shuffle(pool)
        if abs(alpha_of(pool[:na]) - alpha_of(pool[na:])) >= observed:
            hits += 1
    # +1 smoothing so a p of exactly 0 is never reported from a finite resample.
    return (hits + 1) / (n_boot + 1)


def main() -> int:
    awq_path = ROOT / "artifacts" / "alpha.json"
    rtn_path = ROOT / "artifacts" / "alpha_rtn.json"
    for p in (awq_path, rtn_path):
        if not p.exists():
            print(f"missing {p}. Run:\n"
                  f"  python bench/measure_alpha.py --model artifacts/qwen05b\n"
                  f"  python bench/measure_alpha.py --model artifacts/qwen05b "
                  f"--no-awq --out artifacts/alpha_rtn.json")
            return 1

    awq_rec, awq = load(awq_path)
    rtn_rec, rtn = load(rtn_path)

    print(f"gamma = {GAMMA}, bootstrap resamples = {N_BOOT:,}, seed = {SEED}\n")
    rows = []
    for name, passes, rec in (("AWQ", awq, awq_rec), ("RTN", rtn, rtn_rec)):
        a = alpha_of(passes)
        lo, hi = bootstrap_ci(passes)
        rows.append((name, len(passes), a, lo, hi))
        print(f"  {name}: {len(passes)} verify passes, alpha = {a:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"       run lengths: {rec['verdict']['run_lengths']}")

    diff = alpha_of(awq) - alpha_of(rtn)
    p = permutation_p(awq, rtn)
    print(f"\n  difference (AWQ - RTN) = {diff:+.4f}")
    print(f"  permutation test p     = {p:.3f}")

    lo_a, hi_a = rows[0][3], rows[0][4]
    lo_r, hi_r = rows[1][3], rows[1][4]
    overlap = not (hi_a < lo_r or hi_r < lo_a)

    print("\n  VERDICT: ", end="")
    if p >= 0.05:
        print("no significant difference.")
        print(
            "  The confidence intervals overlap"
            f"{'' if overlap else ' (unexpectedly not)'} and p >= 0.05, so at this\n"
            "  sample size AWQ and RTN draft equally well. AWQ's 1.58x reduction in\n"
            "  per-layer output error does NOT show up as better acceptance.\n"
            "\n"
            "  The likely reason is an objective mismatch. AWQ minimises\n"
            "  ||(W_q - W)x||^2 -- output fidelity. Acceptance depends only on\n"
            "  whether the ARGMAX survives, which is a far coarser property: a\n"
            "  method can cut MSE substantially while flipping near-ties in the\n"
            "  logits either way, and only the flips cost acceptance.\n"
            "\n"
            "  This does not make AWQ the wrong choice for standalone W4A16\n"
            "  serving, where output fidelity IS the objective. It says the\n"
            "  calibration target for a *speculative draft* should be top-1\n"
            "  agreement with the target model, which is not what AWQ optimises.\n"
            "  That is a concrete, testable follow-up rather than a conclusion."
        )
    elif diff > 0:
        print(f"AWQ drafts significantly better (p = {p:.3f}).")
    else:
        print(f"RTN drafts significantly better (p = {p:.3f}).")
        print(
            "  A real inversion, not noise. Worth investigating before AWQ is\n"
            "  recommended for the draft path."
        )

    out = {
        "gamma": GAMMA,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "awq": {"passes": len(awq), "alpha": alpha_of(awq),
                "ci95": list(bootstrap_ci(awq))},
        "rtn": {"passes": len(rtn), "alpha": alpha_of(rtn),
                "ci95": list(bootstrap_ci(rtn))},
        "difference_awq_minus_rtn": diff,
        "permutation_p": p,
        "significant_at_0.05": p < 0.05,
    }
    path = ROOT / "artifacts" / "alpha_significance.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

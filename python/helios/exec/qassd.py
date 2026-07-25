"""Quantization-asymmetric self-speculative decoding (spec section 7).

This is the "QA" in QASSD -- the part docs/SCOPE.md described as *the core
novelty claim, and it is not implemented*. It is implemented now, and the number
that decides whether it was worth implementing is measurable without a GPU.

The idea
--------
Speculative decoding needs a draft model that is cheap and agrees with the
target often. The usual answer is a separate small model, which costs extra
memory and has a different tokenizer/distribution. QASSD's answer: draft with a
*4-bit quantization of the same weights*. Then

  * the draft's distribution is derived from the target's, not a different
    model's, so agreement should be high;
  * decode is memory-bandwidth-bound, so a draft step reading 4-bit weights
    costs roughly a quarter of the bytes of a verify step;
  * no second model has to be trained, distilled, or shipped.

What it costs
-------------
More memory, not less, and on TWO axes:

  1. both weight precisions stay resident -- the fp master plus a packed int4
     copy for drafting;
  2. the draft needs its OWN KV cache. Its keys and values are computed from int4
     weights, so they are numerically not the target's and cannot share storage.

`LLMEngine.memory_report()` returns both together, deliberately: quoting the
weight figure alone understates the real cost, and an earlier version of this
docstring did exactly that. `DualPrecisionModel.memory_overhead()` covers only
weights and says so.

Stated plainly because the opposite is easy to imply -- "4-bit quantization"
sounds like a saving. A deployment that only ever wants W4A16 serving keeps the
int4 copy alone and does get a real reduction, but that configuration contains no
speculation; it is `quantize_model` on its own.

The gate
--------
Spec section 14: abandon this if the acceptance rate alpha < 0.6. The expected
tokens per verify pass for greedy speculation with acceptance probability alpha
and draft length gamma is

    E[tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)

and the cost of a verify step plus gamma draft steps, in units of a full-precision
forward pass, is approximately 1 + gamma/4 when the draft reads a quarter of the
bytes. So the speedup is E[tokens] / (1 + gamma/4). That expression is what
`speculation_speedup_model` computes, and it is what makes alpha a *decision*
rather than a curiosity: below some alpha the arithmetic says stop.

alpha is a numerical property of two weight matrices. It does not depend on
having fast int4 kernels, so measuring it here is legitimate even though the
*speedup* is not realisable on this CPU. bench/measure_alpha.py does the
measurement; docs/QASSD.md records the verdict.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .model import HeliosModel
from .quant import QuantConfig, QuantLinear, quantize_model
from .sampler import greedy_token


@dataclass
class AcceptanceStats:
    """Draft/accept bookkeeping, aggregated over a measurement run."""

    drafted: int = 0
    accepted: int = 0
    verify_passes: int = 0
    committed: int = 0
    # Histogram: how many of a gamma-token draft were accepted, per pass.
    run_lengths: Dict[int, int] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        """Per-token acceptance rate: accepted / drafted."""
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_pass(self) -> float:
        """Mean tokens committed per verify pass -- the quantity that pays."""
        return self.committed / self.verify_passes if self.verify_passes else 0.0

    def record(self, n_drafted: int, n_accepted: int, n_committed: int) -> None:
        self.drafted += n_drafted
        self.accepted += n_accepted
        self.committed += n_committed
        self.verify_passes += 1
        self.run_lengths[n_accepted] = self.run_lengths.get(n_accepted, 0) + 1


def speculation_speedup_model(
    alpha: float, gamma: int, draft_cost_ratio: float = 0.25
) -> Dict[str, float]:
    """Expected speedup from greedy speculation, given alpha.

    `draft_cost_ratio` is the cost of one draft forward pass relative to one
    verify pass. 0.25 is the bandwidth-bound ideal for 4-bit vs 16-bit weights;
    it is optimistic, since attention over the KV cache is unquantized and does
    not shrink, and kernel overheads do not either.

    The expected number of tokens committed per verify pass, for a draft of
    length gamma where each draft token is independently accepted with
    probability alpha (and one bonus token is always emitted):

        E[tokens] = sum_{i=0..gamma} alpha^i = (1 - alpha^(gamma+1)) / (1 - alpha)

    Independence is an approximation -- acceptances are correlated, because a
    context the draft finds easy tends to stay easy -- and it is the same
    approximation the speculative-decoding literature uses. Correlation makes the
    real distribution more bimodal without changing the mean much.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if alpha == 1.0:
        e_tokens = float(gamma + 1)
    else:
        e_tokens = (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)
    cost = 1.0 + gamma * draft_cost_ratio
    return {
        "alpha": alpha,
        "gamma": gamma,
        "expected_tokens_per_pass": e_tokens,
        "cost_per_pass": cost,
        "speedup": e_tokens / cost,
    }


def best_gamma(alpha: float, draft_cost_ratio: float = 0.25, max_gamma: int = 16) -> int:
    """The gamma that maximises modelled speedup at this alpha.

    Longer drafts win more when acceptance is high and lose more when it is low,
    because a rejection throws away every draft token after it. The optimum is
    interior, which is the justification for the scheduler's adaptive gating
    (spec section 7.3) rather than a fixed gamma.
    """
    scored = [
        (speculation_speedup_model(alpha, g, draft_cost_ratio)["speedup"], g)
        for g in range(1, max_gamma + 1)
    ]
    return max(scored)[1]


class DualPrecisionModel:
    """An fp target model plus an int4 draft view of the same weights.

    Deliberately not an nn.Module: it owns two models and must never be treated
    as one, because `.to()` or `.parameters()` on a merged object would silently
    include the draft copy in a size calculation. Keeping the seam explicit is
    the point -- the runner asks for `target` or `draft` by name.

    The draft is a deep copy that then has its Linears replaced in place. That
    copy is the extra weight memory, and it is made once at load. The draft's
    KV shadow is allocated separately by ModelRunner; see LLMEngine.memory_report()
    for the combined figure.
    """

    def __init__(
        self,
        target: HeliosModel,
        cfg: Optional[QuantConfig] = None,
        calib_token_ids: Optional[List[List[int]]] = None,
        verbose: bool = False,
    ) -> None:
        self.target = target
        self.cfg = cfg or QuantConfig()

        self.draft = copy.deepcopy(target)
        self.report = quantize_model(
            self.draft, self.cfg, calib_token_ids=calib_token_ids, verbose=verbose
        )
        self.draft.eval()

    @property
    def config(self):
        return self.target.config

    @property
    def device(self) -> str:
        return self.target.device

    def memory_overhead(self) -> Dict[str, float]:
        """What holding both precisions costs, in bytes and as a ratio.

        Uses `resident_bytes()`, not `stored_bytes()`: the difference is any
        dequantization cache a QuantLinear is holding, and reporting the packed
        size while the process holds an fp copy is how a memory regression gets
        published as a saving. See QuantLinear.dequantized().
        """
        target_bytes = sum(p.numel() * p.element_size() for p in self.target.parameters())
        draft_bytes = sum(
            p.numel() * p.element_size() for p in self.draft.parameters()
        ) + sum(
            m.resident_bytes() for m in self.draft.modules() if isinstance(m, QuantLinear)
        )
        return {
            "target_bytes": target_bytes,
            "draft_bytes": draft_bytes,
            "total_bytes": target_bytes + draft_bytes,
            "overhead_ratio": (target_bytes + draft_bytes) / max(1, target_bytes),
            "draft_alone_compression": target_bytes / max(1, draft_bytes),
        }

    def summary(self) -> str:
        m = self.memory_overhead()
        return (
            f"QASSD dual-precision: target {m['target_bytes'] / 2**20:.1f} MiB fp + "
            f"draft {m['draft_bytes'] / 2**20:.1f} MiB int4 "
            f"= {m['overhead_ratio']:.2f}x the target alone "
            f"(the int4 view alone would be {m['draft_alone_compression']:.2f}x smaller)"
        )


def measure_acceptance(
    dual: DualPrecisionModel,
    prompt_token_ids: List[List[int]],
    gamma: int = 4,
    max_new_tokens: int = 64,
    block_size: int = 16,
    stats: Optional[AcceptanceStats] = None,
) -> AcceptanceStats:
    """Run true quantization-asymmetric speculation and measure alpha.

    This is the measurement, not a serving path: it uses a private KV cache per
    prompt so that the number is not entangled with the scheduler's paging
    decisions. The *serving* path is ModelRunner._run_speculative_quantized.

    Structure per iteration:
      1. draft gamma tokens with the int4 model, sequentially (each needs the
         previous token), writing into the DRAFT's own KV cache;
      2. verify all gamma in one fp forward pass over the target's KV cache;
      3. accept the longest prefix where the draft matches the target's argmax;
      4. emit one bonus token from the target at the first mismatch;
      5. roll both KV caches back to the committed length.

    Two separate KV caches is the honest cost of asymmetric drafting: the draft
    model's keys and values are computed from int4 weights and are therefore
    *not* the target's, so they cannot share storage. Spec section 7.4's
    "quantized KV shadow" is exactly this second cache, and its size is why the
    spec wanted it quantized too -- see paged_attn.QuantizedPagedKVCache.
    """
    from .paged_attn import PagedKVCache

    stats = stats or AcceptanceStats()
    cfg = dual.config
    dtype = next(dual.target.parameters()).dtype

    for ids in prompt_token_ids:
        total_len = len(ids) + max_new_tokens + gamma + 2
        n_blocks = (total_len + block_size - 1) // block_size

        def fresh():
            return [
                PagedKVCache(
                    num_blocks=n_blocks,
                    block_size=block_size,
                    n_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=dtype,
                    device=dual.device,
                )
                for _ in range(cfg.num_hidden_layers)
            ]

        target_kv, draft_kv = fresh(), fresh()
        blocks = list(range(n_blocks))

        with torch.inference_mode():
            # Prefill both models over the prompt.
            positions = list(range(len(ids)))
            t_logits = dual.target.forward(
                list(ids), positions, target_kv, blocks, False, len(ids)
            )
            dual.draft.forward(
                list(ids), positions, draft_kv, blocks, False, len(ids)
            )

            committed = [greedy_token(t_logits[-1])]
            ctx = len(ids) + 1

            while len(committed) < max_new_tokens:
                # ---- 1. DRAFT with int4 weights
                drafted: List[int] = []
                cur_tok = committed[-1]
                cur_pos = ctx - 1
                d_ctx = ctx
                for _ in range(gamma):
                    lg = dual.draft.forward(
                        [cur_tok], [cur_pos], draft_kv, blocks, True, d_ctx
                    )
                    tok = greedy_token(lg[-1])
                    drafted.append(tok)
                    cur_tok = tok
                    cur_pos += 1
                    d_ctx += 1
                    if d_ctx >= n_blocks * block_size:
                        break
                if not drafted:
                    break

                # ---- 2. VERIFY in one fp pass
                verify_tokens = [committed[-1]] + drafted[:-1]
                verify_pos = list(range(ctx - 1, ctx - 1 + len(verify_tokens)))
                all_logits = dual.target.forward_all_logits(
                    verify_tokens, verify_pos, target_kv, blocks,
                    ctx - 1 + len(verify_tokens),
                )

                # ---- 3/4. ACCEPT the matching prefix, then one bonus token
                n_acc = 0
                for i, tok in enumerate(drafted):
                    if greedy_token(all_logits[i]) == tok:
                        n_acc += 1
                    else:
                        break
                new_tokens = drafted[:n_acc]
                if n_acc < len(drafted):
                    new_tokens = new_tokens + [greedy_token(all_logits[n_acc])]

                stats.record(len(drafted), n_acc, len(new_tokens))
                committed.extend(new_tokens)
                ctx += len(new_tokens)

                # ---- 5. Re-sync the draft's KV to the committed prefix.
                #
                # The draft ran ahead by `gamma` and some of that was rejected,
                # so its cache holds keys for tokens that were never committed.
                # Nothing truncates paged storage -- the next write to those
                # positions overwrites them -- but the draft's next step must
                # start from the committed token, and the KV at the accepted
                # positions was computed from the DRAFT's own (wrong) prediction
                # for anything past the first rejection. Recomputing the
                # committed suffix through the draft is what keeps the two
                # models' contexts consistent.
                if new_tokens:
                    resync_from = ctx - len(new_tokens) - 1
                    dual.draft.forward(
                        [committed[-len(new_tokens) - 1]] + new_tokens[:-1],
                        list(range(resync_from, resync_from + len(new_tokens))),
                        draft_kv, blocks, False, ctx - 1,
                    )
                if ctx >= n_blocks * block_size - gamma - 1:
                    break

    return stats


def acceptance_verdict(stats: AcceptanceStats, kill_threshold: float = 0.6) -> Dict[str, object]:
    """Apply spec section 14's kill gate to a measured alpha.

    Returns the verdict rather than printing it, so the caller decides whether
    this is a test assertion or a report line.
    """
    alpha = stats.alpha
    at_best = best_gamma(alpha)
    model = speculation_speedup_model(alpha, at_best)
    return {
        "alpha": alpha,
        "drafted": stats.drafted,
        "accepted": stats.accepted,
        "measured_tokens_per_pass": stats.tokens_per_pass,
        "kill_threshold": kill_threshold,
        "passes_gate": alpha >= kill_threshold,
        "best_gamma": at_best,
        "modelled_speedup_at_best_gamma": model["speedup"],
        "run_lengths": dict(sorted(stats.run_lengths.items())),
    }

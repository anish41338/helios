"""Quantization-asymmetric self-speculative decoding tests (spec section 7).

The load-bearing claim is a *safety* one, and it is the reason speculative
decoding is usable at all: speculation must not change what the engine emits. The
draft only proposes; every committed token comes from the target's distribution.
If that holds, a bad draft costs throughput and nothing else -- which is what
makes it safe to ship a draft whose quality is unknown.

`test_asymmetric_speculation_matches_the_target_exactly` is that claim. It is the
single most important test in this file, because a violation would mean the
engine silently serves the int4 model's output while reporting the fp model's.

The other claims here are about the arithmetic of the section 14 gate, which is a
decision procedure and therefore has to be right in the directions it is used:
speedup must rise with alpha, and the optimal gamma must rise with alpha.
"""

from __future__ import annotations

import pytest
import torch

from helios.exec.qassd import (
    AcceptanceStats,
    DualPrecisionModel,
    acceptance_verdict,
    best_gamma,
    measure_acceptance,
    speculation_speedup_model,
)
from helios.exec.quant import QuantConfig, QuantLinear


@pytest.fixture(scope="module")
def dual(quant_toy_dir):
    from helios.exec.loader import load_model

    return DualPrecisionModel(load_model(quant_toy_dir), QuantConfig(group_size=64))


# -------------------------------------------------------------- the FSM math


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_perfect_acceptance_commits_gamma_plus_one(gamma):
    m = speculation_speedup_model(1.0, gamma)
    assert m["expected_tokens_per_pass"] == pytest.approx(gamma + 1)


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_zero_acceptance_still_commits_one_token(gamma):
    """The bonus token is what stops a rejection from stalling the sequence."""
    m = speculation_speedup_model(0.0, gamma)
    assert m["expected_tokens_per_pass"] == pytest.approx(1.0)


def test_speedup_is_monotonic_in_alpha():
    prev = 0.0
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0):
        s = speculation_speedup_model(a, 4)["speedup"]
        assert s > prev, f"speedup fell as alpha rose to {a}"
        prev = s


def test_best_gamma_rises_with_alpha():
    """Longer drafts only pay when acceptance is high -- the basis for section 7.3.

    If this were flat, the scheduler's adaptive gating would be pointless
    machinery.
    """
    gammas = [best_gamma(a) for a in (0.1, 0.5, 0.8, 0.95, 0.99)]
    assert gammas == sorted(gammas)
    assert gammas[0] < gammas[-1], gammas


def test_speculation_loses_below_the_break_even_point():
    """At low alpha, drafting costs more than it returns -- hence a kill gate."""
    assert speculation_speedup_model(0.1, 8)["speedup"] < 1.0


def test_alpha_out_of_range_is_rejected():
    with pytest.raises(ValueError, match=r"alpha must be in"):
        speculation_speedup_model(1.5, 4)


def test_verdict_applies_the_section_14_threshold():
    low, high = AcceptanceStats(), AcceptanceStats()
    low.record(4, 1, 2)      # 1 of 4 accepted
    high.record(4, 4, 4)
    assert not acceptance_verdict(low)["passes_gate"]
    assert acceptance_verdict(high)["passes_gate"]
    assert acceptance_verdict(low)["kill_threshold"] == 0.6


def test_acceptance_stats_bookkeeping():
    s = AcceptanceStats()
    s.record(4, 2, 3)
    s.record(4, 4, 4)
    assert s.drafted == 8 and s.accepted == 6
    assert s.alpha == pytest.approx(0.75)
    assert s.tokens_per_pass == pytest.approx(3.5)
    assert s.run_lengths == {2: 1, 4: 1}


# --------------------------------------------------------- dual-precision model


def test_draft_is_quantized_and_target_is_not(dual):
    assert any(isinstance(m, QuantLinear) for m in dual.draft.modules())
    assert not any(isinstance(m, QuantLinear) for m in dual.target.modules())


def test_draft_weights_differ_from_the_target(dual):
    """A deep copy that silently shared storage would make alpha exactly 1.0.

    That failure mode is invisible in every other test -- the engine would work
    perfectly and just report a fictional acceptance rate.
    """
    t = dual.target.layers[0].self_attn.q_proj
    d = dual.draft.layers[0].self_attn.q_proj
    assert isinstance(d, QuantLinear)
    assert not torch.allclose(d.dequantized(), t.weight, atol=1e-8)


def test_memory_overhead_is_reported_as_an_increase(dual):
    """Holding both precisions costs MORE than the target alone. Say so."""
    m = dual.memory_overhead()
    assert m["overhead_ratio"] > 1.0
    assert m["total_bytes"] == m["target_bytes"] + m["draft_bytes"]
    assert m["draft_alone_compression"] > 1.0
    assert "x the target alone" in dual.summary()


def test_measure_acceptance_produces_consistent_counts(dual):
    stats = measure_acceptance(dual, [[3 + i for i in range(16)]], gamma=4,
                               max_new_tokens=12)
    assert stats.verify_passes > 0
    assert stats.drafted >= stats.accepted
    assert 0.0 <= stats.alpha <= 1.0
    # Every pass commits at least one token (accepted prefix, or the bonus).
    assert stats.committed >= stats.verify_passes


# ------------------------------------------------------ the safety property


def test_asymmetric_speculation_matches_the_target_exactly(quant_toy_dir):
    """QASSD must not change the engine's output. The critical test.

    Greedy decoding with and without a quantized draft, same prompts, same
    engine config otherwise. Token ids must be identical -- not close.

    Why it holds: an accepted token is one the target's argmax agreed with, and a
    rejection emits the target's own argmax. So the committed sequence is the
    target's greedy sequence by construction, whatever the draft proposes. A
    failure here means the accept/rollback bookkeeping is wrong, and the engine
    would be quietly serving degraded output.
    """
    from helios.core.types import SamplingParams
    from helios.engine import EngineConfig, LLMEngine

    prompts = [[3 + (i * 7 + j) % 200 for j in range(20)] for i in range(3)]
    params = SamplingParams(max_tokens=24, temperature=0.0)

    def run(**kwargs):
        engine = LLMEngine(
            EngineConfig(
                model_dir=quant_toy_dir, kv_cache_bytes=16 * 1024 * 1024,
                max_model_len=256, **kwargs,
            )
        )
        for i, p in enumerate(prompts):
            engine.add_request(f"r{i}", p, params)
        outs = engine.run_until_complete(max_steps=4000)
        return {o.request_id: o.token_ids for o in outs}, engine

    baseline, _ = run()
    qassd, engine = run(
        enable_spec_decode=True, spec_gamma=4, quantized_draft=True,
        quant_group_size=64,
    )

    assert set(baseline) == set(qassd)
    for rid in baseline:
        assert qassd[rid] == baseline[rid], (
            f"{rid}: quantization-asymmetric speculation changed the output.\n"
            f"  baseline: {baseline[rid][:12]}\n"
            f"  qassd   : {qassd[rid][:12]}"
        )
    # And the draft was genuinely exercised, not silently skipped.
    assert engine.runner.spec_drafted > 0
    assert engine.dual is not None


def test_quantized_draft_without_speculation_is_a_config_error(quant_toy_dir):
    """An int4 copy nothing reads is pure memory overhead. Fail loudly."""
    from helios.engine import EngineConfig, LLMEngine

    with pytest.raises(ValueError, match="requires enable_spec_decode"):
        LLMEngine(
            EngineConfig(model_dir=quant_toy_dir, quantized_draft=True,
                         enable_spec_decode=False)
        )


def test_memory_report_includes_the_draft_kv_shadow(quant_toy_dir):
    """The complete cost, not just the weights.

    Regression test for an incomplete accounting: `memory_overhead()` covers only
    weights, and enabling a quantized draft also allocates an entire second KV
    cache. Quoting the weight-only figure as "the cost of QASSD" understated it.
    """
    from helios.engine import EngineConfig, LLMEngine

    base = dict(model_dir=quant_toy_dir, kv_cache_bytes=16 * 1024 * 1024,
                max_num_seqs=8, max_model_len=256)
    plain = LLMEngine(EngineConfig(**base))
    qassd = LLMEngine(EngineConfig(
        **base, enable_spec_decode=True, spec_gamma=2,
        quantized_draft=True, quant_group_size=64,
    ))

    p, q = plain.memory_report(), qassd.memory_report()
    assert p["draft_weight_bytes"] == 0 and p["draft_kv_bytes"] == 0
    assert p["total_overhead_ratio"] == pytest.approx(1.0)

    # Both extra costs must be present and counted.
    assert q["draft_weight_bytes"] > 0, "the int4 draft weights are not counted"
    assert q["draft_kv_bytes"] > 0, "the draft KV shadow is not counted"
    assert q["total_bytes"] == (
        q["target_weight_bytes"] + q["main_kv_bytes"]
        + q["draft_weight_bytes"] + q["draft_kv_bytes"]
    )
    assert q["total_overhead_ratio"] > 1.0


def test_gate_disables_speculation_when_the_draft_is_bad(quant_toy_dir):
    """The adaptive gate must react to a genuinely bad draft (spec section 7.3).

    On a random-weight toy model an int4 draft agrees almost never, so acceptance
    collapses and speculation becomes pure cost. The gate has to notice. Without
    it, QASSD on a bad draft is ~3x SLOWER than not speculating at all -- measured.

    This is what makes shipping a draft of uncharacterised quality safe: not that
    the draft is good, but that the engine stops using it when it is not.
    """
    from helios.core.types import SamplingParams
    from helios.engine import EngineConfig, LLMEngine

    engine = LLMEngine(EngineConfig(
        model_dir=quant_toy_dir, kv_cache_bytes=16 * 1024 * 1024,
        max_num_seqs=8, max_model_len=256,
        enable_spec_decode=True, spec_gamma=2,
        quantized_draft=True, quant_group_size=64,
    ))

    gammas = []
    original = engine.scheduler._build_exec_step

    def spy(*a, **k):
        step = original(*a, **k)
        if step.decodes:
            gammas.append(step.spec_gamma)
        return step

    engine.scheduler._build_exec_step = spy

    for i in range(12):
        engine.add_request(
            f"r{i}", [3 + (i * 11 + j) % 200 for j in range(32)],
            SamplingParams(max_tokens=20, temperature=0.0),
        )
    engine.run_until_complete(max_steps=6000)

    assert gammas, "no decode steps ran"
    speculated = sum(1 for g in gammas if g > 0)
    assert speculated >= 1, "speculation never ran, so the gate proves nothing"
    assert speculated < len(gammas), (
        "the gate never fired despite a draft that agrees "
        f"{engine.runner.measured_acceptance:.1%} of the time"
    )
    assert engine.runner.measured_acceptance < 0.6, (
        "this test assumes a bad draft; if the toy draft became good, the "
        "assertion above no longer tests the gate"
    )


def test_draft_uses_a_separate_quantized_kv_cache(quant_toy_dir):
    """The draft's KV comes from int4 weights and cannot share the target's.

    Sharing would corrupt the target's context with the draft's approximate keys
    -- and would look like a working optimisation right up to the point where
    output quality was measured.
    """
    from helios.exec.paged_attn import QuantizedPagedKVCache
    from helios.engine import EngineConfig, LLMEngine

    engine = LLMEngine(
        EngineConfig(
            model_dir=quant_toy_dir, kv_cache_bytes=16 * 1024 * 1024,
            enable_spec_decode=True, quantized_draft=True, quant_group_size=64,
        )
    )
    assert engine.runner.draft_kv_caches is not None
    assert len(engine.runner.draft_kv_caches) == len(engine.runner.kv_caches)
    for d, t in zip(engine.runner.draft_kv_caches, engine.runner.kv_caches):
        assert d is not t
        assert isinstance(d, QuantizedPagedKVCache)

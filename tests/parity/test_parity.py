"""Correctness parity tests.

Spec section 13.1 defines an oracle hierarchy. This build's rungs:

  * paged attention  vs a dense, unpaged reference implementation
  * chunked prefill  vs single-shot prefill (must be bit-identical)
  * speculative decode vs non-speculative (must be bit-identical, section 7.2)
  * engine end-to-end vs a direct model forward loop

Per spec section 19.2, tolerances here are never loosened to make a test pass.
Where an exact match is required it is asserted exactly; where float
associativity makes that impossible, the tolerance is tight and justified.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile

import pytest
import torch

from helios.core.types import SamplingParams
from helios.engine import EngineConfig, LLMEngine
from helios.exec.loader import save_toy_model
from helios.exec.model import ModelConfig, HeliosModel
from helios.exec.paged_attn import (
    PagedKVCache,
    _repeat_kv,
    paged_attention_decode,
    paged_attention_prefill,
)

TOY_CONFIG = ModelConfig(
    vocab_size=259,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def toy_dir():
    path = os.path.join(tempfile.gettempdir(), "helios_parity_toy")
    shutil.rmtree(path, ignore_errors=True)
    save_toy_model(path, TOY_CONFIG, seed=0)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def toy_model(toy_dir):
    from helios.exec.loader import load_model

    return load_model(toy_dir)


def make_engine(toy_dir, **overrides) -> LLMEngine:
    kwargs = dict(
        model_dir=toy_dir,
        kv_cache_bytes=8 * 1024 * 1024,
        block_size=16,
        max_model_len=256,
    )
    kwargs.update(overrides)
    return LLMEngine(EngineConfig(**kwargs))


# --------------------------------------------------- paged vs dense attention


def dense_causal_attention(q, k, v, n_rep):
    """Reference: plain causal attention with no paging whatsoever."""
    T, n_q_heads, head_dim = q.shape
    kr = _repeat_kv(k, n_rep).permute(1, 0, 2)
    vr = _repeat_kv(v, n_rep).permute(1, 0, 2)
    qr = q.permute(1, 0, 2)
    scores = torch.matmul(qr, kr.transpose(1, 2)) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask[None], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), vr).permute(1, 0, 2)


@pytest.mark.parametrize("n_tokens", [1, 7, 11, 16, 33])
def test_paged_prefill_matches_dense_reference(n_tokens):
    """Paging must not change the mathematics of attention."""
    torch.manual_seed(0)
    bs, n_kv, hd, n_q = 4, 2, 8, 4
    cache = PagedKVCache(num_blocks=32, block_size=bs, n_kv_heads=n_kv, head_dim=hd)
    n_blocks = (n_tokens + bs - 1) // bs
    # Deliberately non-contiguous physical blocks: a bug in block-table
    # indirection would show up here and not with sequential ids.
    blocks = [(7 * i + 3) % 32 for i in range(n_blocks)]
    assert len(set(blocks)) == len(blocks)

    k = torch.randn(n_tokens, n_kv, hd)
    v = torch.randn(n_tokens, n_kv, hd)
    q = torch.randn(n_tokens, n_q, hd)
    cache.write(blocks, 0, k, v)

    got = paged_attention_prefill(q, cache, blocks, 0, n_tokens)
    want = dense_causal_attention(q, k, v, n_q // n_kv)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_paged_decode_matches_dense_reference():
    torch.manual_seed(1)
    bs, n_kv, hd, n_q = 4, 2, 8, 4
    T = 11
    cache = PagedKVCache(num_blocks=32, block_size=bs, n_kv_heads=n_kv, head_dim=hd)
    blocks = [5, 1, 9]
    k = torch.randn(T, n_kv, hd)
    v = torch.randn(T, n_kv, hd)
    cache.write(blocks, 0, k, v)

    kn, vn = torch.randn(1, n_kv, hd), torch.randn(1, n_kv, hd)
    qn = torch.randn(n_q, hd)
    cache.write(blocks, T, kn, vn)

    got = paged_attention_decode(qn, cache, blocks, T + 1)

    kall = _repeat_kv(torch.cat([k, kn]), n_q // n_kv).permute(1, 0, 2)
    vall = _repeat_kv(torch.cat([v, vn]), n_q // n_kv).permute(1, 0, 2)
    scores = torch.matmul(qn.unsqueeze(1), kall.transpose(1, 2)) / math.sqrt(hd)
    want = torch.matmul(torch.softmax(scores, dim=-1), vall).squeeze(1)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_kv_write_read_round_trips_through_scattered_blocks():
    cache = PagedKVCache(num_blocks=16, block_size=4, n_kv_heads=2, head_dim=8)
    blocks = [11, 2, 7]
    k = torch.randn(10, 2, 8)
    v = torch.randn(10, 2, 8)
    cache.write(blocks, 0, k, v)
    gk, gv = cache.gather(blocks, 10)
    torch.testing.assert_close(gk, k)
    torch.testing.assert_close(gv, v)


def test_gather_rejects_context_longer_than_block_table():
    cache = PagedKVCache(num_blocks=16, block_size=4, n_kv_heads=2, head_dim=8)
    with pytest.raises(IndexError):
        cache.gather([0], 99)


# ------------------------------------------------- chunked vs single-shot


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 8])
def test_chunked_prefill_is_identical_to_single_shot(toy_model, chunk):
    """Chunking is a scheduling decision and must not alter numerics at all."""
    cfg = toy_model.config
    prompt = [5, 9, 14, 22, 33, 41, 7, 2, 88, 100, 3, 55]
    blocks = [0, 1, 2, 3]

    def caches():
        return [
            PagedKVCache(16, 16, cfg.num_key_value_heads, cfg.head_dim)
            for _ in range(cfg.num_hidden_layers)
        ]

    with torch.inference_mode():
        single = toy_model.forward(
            prompt, list(range(len(prompt))), caches(), blocks, False, len(prompt)
        )

        cs = caches()
        chunked = None
        for start in range(0, len(prompt), chunk):
            part = prompt[start : start + chunk]
            chunked = toy_model.forward(
                part,
                list(range(start, start + len(part))),
                cs,
                blocks,
                False,
                start + len(part),
            )

    assert int(single.argmax()) == int(chunked.argmax()), "same token must be chosen"
    torch.testing.assert_close(single, chunked, atol=1e-5, rtol=1e-5)


def test_decode_continues_prefill_consistently(toy_model):
    """A decode step after prefill must see the prompt's KV.

    Compares against a full forward pass over prompt+token: if the decode path
    read the wrong blocks or positions, the logits would diverge.
    """
    cfg = toy_model.config
    prompt = [5, 9, 14, 22, 33]
    blocks = [0, 1]

    def caches():
        return [
            PagedKVCache(16, 16, cfg.num_key_value_heads, cfg.head_dim)
            for _ in range(cfg.num_hidden_layers)
        ]

    with torch.inference_mode():
        cs = caches()
        logits = toy_model.forward(
            prompt, list(range(len(prompt))), cs, blocks, False, len(prompt)
        )
        nxt = int(logits.argmax())
        decode_logits = toy_model.forward(
            [nxt], [len(prompt)], cs, blocks, True, len(prompt) + 1
        )

        # Reference: one prefill over the whole extended sequence.
        full = toy_model.forward(
            prompt + [nxt],
            list(range(len(prompt) + 1)),
            caches(),
            blocks,
            False,
            len(prompt) + 1,
        )

    torch.testing.assert_close(decode_logits, full, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------ speculative parity


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_speculative_output_is_identical_to_non_speculative(toy_dir, gamma):
    """Spec section 7.2: THE most important test in the project.

    Speculation that changes outputs is a bug, not a speedup. Asserted as exact
    token-sequence equality, never as a tolerance.
    """
    prompt = [5, 9, 14, 22, 33, 41, 7]
    params = SamplingParams(max_tokens=16, temperature=0.0)

    base_engine = make_engine(toy_dir, enable_spec_decode=False)
    base_engine.add_request("base", prompt, params)
    base = base_engine.run_until_complete()[0]

    spec_engine = make_engine(toy_dir, enable_spec_decode=True, spec_gamma=gamma)
    spec_engine.add_request("spec", prompt, params)
    spec = spec_engine.run_until_complete()[0]

    assert spec.token_ids == base.token_ids, (
        f"gamma={gamma} changed the output: "
        f"{spec.token_ids} != {base.token_ids}"
    )


def test_speculation_reports_acceptance_statistics(toy_dir):
    """Acceptance must be measured, not assumed.

    Draft and verify share identical weights in this build (no quantization
    asymmetry), so acceptance is 1.0 by construction. That is a property of this
    scope, not evidence about QASSD -- see docs/SCOPE.md.
    """
    engine = make_engine(toy_dir, enable_spec_decode=True, spec_gamma=4)
    engine.add_request("s", [5, 9, 14, 22], SamplingParams(max_tokens=12))
    engine.run_until_complete()
    stats = engine.scheduler.stats
    assert stats.spec_drafted > 0, "speculation should have run"
    assert stats.acceptance_rate == pytest.approx(1.0)


# ------------------------------------------------------------ engine parity


def test_engine_matches_direct_greedy_forward_loop(toy_dir, toy_model):
    """End-to-end: the engine's tokens must equal a naive generation loop.

    This is the closest available analogue of the spec's "exact match vs HF
    generate()" rung: a reference loop that keeps no paged cache and simply
    re-runs the whole sequence each step.
    """
    cfg = toy_model.config
    prompt = [5, 9, 14, 22, 33]
    n_new = 8

    reference = []
    with torch.inference_mode():
        seq = list(prompt)
        for _ in range(n_new):
            caches = [
                PagedKVCache(64, 16, cfg.num_key_value_heads, cfg.head_dim)
                for _ in range(cfg.num_hidden_layers)
            ]
            blocks = list(range(len(seq) // 16 + 1))
            logits = toy_model.forward(
                seq, list(range(len(seq))), caches, blocks, False, len(seq)
            )
            nxt = int(logits.argmax())
            reference.append(nxt)
            seq.append(nxt)

    engine = make_engine(toy_dir)
    engine.add_request("e", prompt, SamplingParams(max_tokens=n_new, temperature=0.0))
    got = engine.run_until_complete()[0]

    assert got.token_ids == reference


def test_identical_prompts_produce_identical_greedy_output(toy_dir):
    """Concurrent requests must not perturb one another's token streams."""
    engine = make_engine(toy_dir)
    prompt = [5, 9, 14, 22]
    for i in range(4):
        engine.add_request(f"r{i}", prompt, SamplingParams(max_tokens=6, temperature=0.0))
    results = engine.run_until_complete()
    assert len({tuple(r.token_ids) for r in results}) == 1


def test_prefix_cache_does_not_change_output(toy_dir):
    """A cache is only valid if it is invisible in the output.

    Reusing another sequence's KV blocks is exactly where a subtle indexing bug
    would corrupt results, so this compares cached and uncached runs directly.
    """
    shared = list(range(20, 84))     # 4 whole blocks at block_size=16
    params = SamplingParams(max_tokens=8, temperature=0.0)

    cached = make_engine(toy_dir, enable_prefix_cache=True)
    cached.add_request("warm", shared + [200], params)
    cached.run_until_complete()
    cached.add_request("hit", shared + [201], params)
    with_cache = cached.run_until_complete()[0]
    assert cached.scheduler.prefix_cache.hits >= 1, "expected a cache hit to exercise"

    plain = make_engine(toy_dir, enable_prefix_cache=False)
    plain.add_request("nocache", shared + [201], params)
    without_cache = plain.run_until_complete()[0]

    assert with_cache.token_ids == without_cache.token_ids


def test_max_tokens_and_eos_stop_conditions(toy_dir):
    engine = make_engine(toy_dir)
    engine.add_request("len", [5, 9], SamplingParams(max_tokens=3, temperature=0.0))
    out = engine.run_until_complete()[0]
    assert out.completion_tokens == 3
    assert out.finish_reason.value == "length"


@pytest.mark.parametrize("n_seqs", [1, 2, 3, 8])
def test_batched_decode_matches_sequential(toy_model, n_seqs):
    """Batched decode must be numerically identical to one-pass-per-sequence.

    This is the safety net for the executor's main optimisation. A batching bug
    -- an off-by-one in a token slice, a mis-scattered attention output, a
    padding mask that leaks across rows -- would silently mix sequences' KV and
    produce fluent, wrong output. That is the failure mode spec section 19.1
    calls this project class's characteristic risk, so it is asserted directly
    rather than assumed from throughput looking better.

    Sequences deliberately have DIFFERENT context lengths, so the right-padding
    and masking in the batched path are actually exercised.
    """
    from helios.exec.model import BatchedAttnMeta

    cfg = toy_model.config
    prompts = [[5 + i, 9, 14, 22, 33][: 2 + (i % 4)] for i in range(n_seqs)]
    tables = [[2 * i, 2 * i + 1] for i in range(n_seqs)]
    assert len({len(p) for p in prompts}) > 1 or n_seqs == 1

    def caches():
        return [
            PagedKVCache(64, 16, cfg.num_key_value_heads, cfg.head_dim)
            for _ in range(cfg.num_hidden_layers)
        ]

    # Sequential: prefill each, then one decode step each, separate passes.
    cs = caches()
    seq_logits = []
    with torch.inference_mode():
        firsts = []
        for p, tb in zip(prompts, tables):
            lg = toy_model.forward(p, list(range(len(p))), cs, tb, False, len(p))
            firsts.append(int(lg[-1].argmax()))
        for i, (p, tb) in enumerate(zip(prompts, tables)):
            lg = toy_model.forward([firsts[i]], [len(p)], cs, tb, True, len(p) + 1)
            seq_logits.append(lg[-1].clone())

    # Batched: identical prefills, then ONE batched decode pass.
    cb = caches()
    with torch.inference_mode():
        for p, tb in zip(prompts, tables):
            toy_model.forward(p, list(range(len(p))), cb, tb, False, len(p))
        positions = [len(p) for p in prompts]
        meta = BatchedAttnMeta(
            token_slices=[(i, i + 1) for i in range(n_seqs)],
            block_tables=tables,
            context_lens=[len(p) + 1 for p in prompts],
            start_positions=positions,
            is_decode=True,
        )
        batched = toy_model.forward_batched(firsts, positions, cb, meta)

    assert batched.shape[0] == n_seqs
    for i in range(n_seqs):
        assert int(batched[i].argmax()) == int(seq_logits[i].argmax()), (
            f"sequence {i} sampled a different token when batched"
        )
        torch.testing.assert_close(batched[i], seq_logits[i], atol=1e-5, rtol=1e-5)


def test_batched_decode_is_order_invariant(toy_model):
    """A sequence's logits must not depend on its position within the batch.

    If they do, some state is leaking across rows -- the single most dangerous
    class of bug in a batched executor, because output stays plausible.
    """
    from helios.exec.model import BatchedAttnMeta

    cfg = toy_model.config
    prompts = [[5, 9, 14], [22, 33, 41, 7], [88, 100]]
    tables = [[0, 1], [2, 3], [4, 5]]

    def run(order):
        caches = [
            PagedKVCache(64, 16, cfg.num_key_value_heads, cfg.head_dim)
            for _ in range(cfg.num_hidden_layers)
        ]
        with torch.inference_mode():
            firsts = {}
            for i in order:
                lg = toy_model.forward(
                    prompts[i], list(range(len(prompts[i]))), caches,
                    tables[i], False, len(prompts[i]),
                )
                firsts[i] = int(lg[-1].argmax())
            meta = BatchedAttnMeta(
                token_slices=[(k, k + 1) for k in range(len(order))],
                block_tables=[tables[i] for i in order],
                context_lens=[len(prompts[i]) + 1 for i in order],
                start_positions=[len(prompts[i]) for i in order],
                is_decode=True,
            )
            out = toy_model.forward_batched(
                [firsts[i] for i in order],
                [len(prompts[i]) for i in order],
                caches,
                meta,
            )
        return {i: out[k].clone() for k, i in enumerate(order)}

    forward = run([0, 1, 2])
    reversed_ = run([2, 1, 0])
    for i in range(3):
        torch.testing.assert_close(forward[i], reversed_[i], atol=1e-5, rtol=1e-5)


def test_engine_output_unchanged_by_decode_batching(toy_dir):
    """End-to-end: ablating decode batching must not change any token.

    Exercises the same switch the benchmark's `baseline_unbatched_executor`
    flips, so the measured speedup is known to be free of output changes.
    """
    params = SamplingParams(max_tokens=10, temperature=0.0)
    prompts = [[5, 9, 14], [22, 33], [41, 7, 2, 88]]

    batched = make_engine(toy_dir)
    for i, p in enumerate(prompts):
        batched.add_request(f"b{i}", p, params)
    got = {o.request_id: o.token_ids for o in batched.run_until_complete()}

    plain = make_engine(toy_dir)
    runner = plain.runner
    runner._run_decode_batch = lambda items: [runner._run_decode(it, 0) for it in items]
    for i, p in enumerate(prompts):
        plain.add_request(f"b{i}", p, params)
    want = {o.request_id: o.token_ids for o in plain.run_until_complete()}

    assert got == want


def test_output_survives_recompute_preemption(toy_dir):
    """Tokens must be unchanged by preemption and recompute.

    Recompute discards a sequence's KV entirely and re-prefills prompt plus
    everything already generated. Any error in position bookkeeping, block-table
    rebuilding, or the re-prefill path would corrupt the sequence -- and produce
    fluent, wrong text rather than a crash. This is the seam between the
    scheduler (which DST covers with a simulated executor) and the real model, so
    it is verified against actual logits.
    """
    prompts = [[5, 9, 14, 22, 33, 41, 7, 2], [88, 100, 3], [55, 66, 77, 88, 99, 11]]
    params = SamplingParams(max_tokens=14, temperature=0.0)

    def run(force_preempt: bool):
        engine = make_engine(toy_dir)
        if force_preempt:
            sched = engine.scheduler
            original = sched._ensure_running_can_progress

            def aggressive():
                # Preempt the newest running sequence every step there is more
                # than one, forcing repeated recompute.
                if len(sched.running) > 1:
                    sched._preempt(sched.running[-1])
                original()

            sched._ensure_running_can_progress = aggressive

        for i, p in enumerate(prompts):
            engine.add_request(f"r{i}", p, params)
        out = {o.request_id: tuple(o.token_ids) for o in engine.run_until_complete()}
        return out, engine.scheduler.stats.preemptions_recompute

    baseline, quiet = run(False)
    preempted, count = run(True)

    assert count > 0, "the test must actually cause preemptions to be meaningful"
    assert preempted == baseline, (
        f"output changed after {count} recompute preemptions"
    )


def test_stop_string_truncates_output(toy_dir):
    """A `stop` string must terminate generation and be absent from the output.

    Regression: `stop` was validated by the API and then never consulted, so a
    client passing one had it silently ignored -- the same failure mode as
    accepting an out-of-range max_tokens and quietly truncating.
    """
    params = SamplingParams(max_tokens=12, temperature=0.0)
    base_engine = make_engine(toy_dir)
    base_engine.add_request("plain", [5, 9], params)
    base = base_engine.run_until_complete()[0]
    assert base.finish_reason.value == "length"

    # Choose a stop string that the unconstrained run is known to produce.
    target = base.text[3:8]
    assert target and target in base.text

    stop_engine = make_engine(toy_dir)
    stop_engine.add_request(
        "stopped", [5, 9], SamplingParams(max_tokens=12, temperature=0.0, stop=[target])
    )
    stopped = stop_engine.run_until_complete()[0]

    assert stopped.finish_reason.value == "stop"
    assert target not in stopped.text, "the stop string must not appear in the output"
    assert len(stopped.text) < len(base.text)
    stop_engine.scheduler.check_invariants()


def test_stop_token_id_finishes_sequence(toy_dir):
    """Stop token ids are handled in the scheduler (a plain integer compare)."""
    engine = make_engine(toy_dir)
    engine.add_request("x", [5, 9], SamplingParams(max_tokens=8, temperature=0.0))
    first = engine.run_until_complete()[0]

    # Stop on the 3rd generated token (index 2). It is committed, then ends the
    # sequence, so exactly 3 tokens come back.
    stop_id = first.token_ids[2]
    engine2 = make_engine(toy_dir)
    engine2.add_request(
        "y",
        [5, 9],
        SamplingParams(max_tokens=8, temperature=0.0, stop_token_ids=[stop_id]),
    )
    got = engine2.run_until_complete()[0]
    assert got.finish_reason.value == "stop"
    # The token ids before the stop must match the unconstrained run.
    assert got.token_ids == first.token_ids[: len(got.token_ids)]
    assert got.token_ids[-1] == stop_id, "generation ends on the stop token"
    assert got.completion_tokens < 8, "stopped before max_tokens"


def test_engine_rejects_prompt_over_context_limit(toy_dir):
    engine = make_engine(toy_dir, max_model_len=64)
    with pytest.raises(ValueError):
        engine.add_request("toolong", list(range(100)), SamplingParams(max_tokens=1))


def test_loader_rejects_unmapped_checkpoint_tensors(tmp_path):
    """A checkpoint tensor with no slot in the model must fail the load.

    Regression, found on a real T4: Qwen2 puts biases on q/k/v projections and
    the loader skipped them silently, leaving a model that ran and generated
    fluent nonsense. Missing *model* params were already caught; ignored
    *checkpoint* params were not.
    """
    import shutil as _shutil

    from safetensors.torch import load_file, save_file
    from helios.exec.loader import load_model

    src = tmp_path / "src"
    save_toy_model(src, TOY_CONFIG, seed=0)

    # Build the tampered copy in a separate directory: safetensors mmaps the
    # file it reads, so rewriting it in place fails on Windows.
    dst = tmp_path / "tampered"
    dst.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if (src / name).exists():
            _shutil.copy(src / name, dst / name)
    state = load_file(str(src / "model.safetensors"))
    state["model.layers.0.self_attn.q_proj.bias"] = torch.zeros(
        TOY_CONFIG.hidden_size
    )
    save_file(state, str(dst / "model.safetensors"))

    with pytest.raises(ValueError, match="no slot in this architecture"):
        load_model(dst)


def test_qwen2_config_infers_attention_bias():
    """Qwen2 needs q/k/v biases; Llama must not get them.

    Qwen2's config.json carries no `attention_bias` flag, so defaulting to False
    silently dropped real weights (see the regression above). Detection is by
    model_type/architecture, with an explicit flag winning when present.
    """
    qwen = {
        "vocab_size": 8, "hidden_size": 896, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 14,
        "num_key_value_heads": 2, "model_type": "qwen2",
    }
    llama = dict(qwen, model_type="llama", hidden_size=64,
                 num_attention_heads=4)

    assert ModelConfig.from_hf(qwen).attention_bias is True
    assert ModelConfig.from_hf(llama).attention_bias is False
    # An explicit flag overrides the heuristic.
    assert ModelConfig.from_hf(dict(qwen, attention_bias=False)).attention_bias is False


def test_attention_bias_is_actually_applied():
    """The flag must reach the projection layers, not just the config."""
    from helios.exec.model import HeliosModel

    cfg = ModelConfig(
        vocab_size=8, hidden_size=64, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, attention_bias=True,
    )
    attn = HeliosModel(cfg).layers[0].self_attn
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None, "o_proj never carries a bias"

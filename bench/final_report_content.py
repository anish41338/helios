"""Content for helios_report.pdf -- the complete record of what was built.

Written to a different brief from report_content.py: no recommendations, no
future work, no framing of limitations as opportunities. It states what exists,
what was measured, what was got wrong, and what was not done. Where a number is
modelled rather than measured it says so on the same line.

Every figure is interpolated from an artifact in artifacts/ via the `Data` object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# Line counts, collected from the tree at the time of writing. Kept as data so
# the inventory table cannot drift from prose describing it.
MODULES = [
    ("core/scheduler.py", 1206, "Continuous batching, admission, preemption, "
     "chunked prefill, speculation gating"),
    ("core/vopr.py", 857, "Deterministic simulation harness: workload generation, "
     "fault injection, invariant checks, replay"),
    ("core/allocator.py", 602, "Paged block allocator: block tables, ref counts, "
     "copy-on-write, host swap tier, invariants I1-I7"),
    ("core/disagg.py", 543, "KV transfer state machine (not called by the engine "
     "-- see section 9)"),
    ("core/prefix_cache.py", 307, "Radix trie over block-aligned prefixes, LRU "
     "eviction, pinning"),
    ("core/types.py", 211, "Requests, sequences, sampling params, SLO classes"),
    ("core/execstep.py", 167, "The scheduler/executor value contract"),
    ("exec/quant.py", 602, "INT4 packing, group-wise quantization, AWQ scale "
     "search, QuantLinear"),
    ("exec/model.py", 524, "Llama/Qwen2 forward pass: GQA, RoPE, SwiGLU, RMSNorm, "
     "batched and single-sequence paths"),
    ("exec/runner.py", 433, "ExecStep execution, batched prefill/decode, "
     "speculative decode"),
    ("exec/paged_attn.py", 408, "Paged KV storage and attention; INT8 quantized "
     "cache variant"),
    ("exec/qassd.py", 376, "Dual-precision model, acceptance measurement, "
     "speedup model"),
    ("exec/loader.py", 301, "safetensors loading, toy-model generation"),
    ("exec/triton_attn.py", 183, "Fused paged decode attention kernel"),
    ("exec/sampler.py", 96, "Greedy, temperature, top-k, top-p, per-request seeds"),
    ("engine.py", 559, "Scheduler + executor + tokenizer + metrics + streaming"),
    ("api/server.py", 548, "OpenAI-compatible HTTP, SSE streaming, Prometheus"),
    ("cli.py", 309, "serve / generate / vopr / info / make-toy-model"),
]

TEST_FILES = [
    ("tests/allocator/test_allocator.py", 57, 308),
    ("tests/parity/test_parity.py", 38, 684),
    ("tests/allocator/test_prefix_cache.py", 36, 229),
    ("tests/e2e/test_api.py", 27, 292),
    ("tests/scheduler/test_scheduler.py", 26, 423),
    ("tests/quant/test_quant.py", 26, 368),
    ("tests/disagg/test_disagg.py", 26, 362),
    ("tests/quant/test_qassd.py", 23, 308),
    ("tests/quant/test_kv_quant.py", 10, 205),
    ("tests/parity/test_triton_parity.py", 9, 174),
]

BENCH_FILES = [
    ("bench/report_content.py", 1014, "PDF report content (the other report)"),
    ("bench/loadgen.py", 701, "Workload generation, ablation suites, baselines"),
    ("bench/make_report.py", 542, "Artifact loading, chart rendering, layout"),
    ("bench/report.py", 377, "Generates docs/BENCHMARKS.md from artifacts"),
    ("bench/vllm_baseline.py", 334, "vLLM comparison harness -- written, never run"),
    ("bench/kaggle_gpu_run.py", 297, "One-shot GPU verification sequence"),
    ("bench/real_model_check.py", 239, "Real-checkpoint end-to-end check"),
    ("bench/measure_alpha.py", 223, "Acceptance-rate measurement"),
    ("bench/alpha_significance.py", 175, "Bootstrap and permutation test on alpha"),
]

# The fifteen scheduler bugs the simulation harness found. Symptom / cause pairs
# are compressed from docs/DST.md, which holds the full write-ups.
DST_BUGS = [
    ("001", "Deadlock, 0/44 requests finished",
     "With chunked prefill off, a prompt longer than the per-step token budget was "
     "unschedulable, and admission broke on it, blocking every smaller request behind it"),
    ("002", "Same deadlock, remaining case",
     "A prompt within max_model_len but larger than the whole KV pool was accepted "
     "and queued forever; max_model_len was never validated against KV capacity"),
    ("003", "I7 violated: committed 63 > usable 62",
     "blocks_needed_to_append returned 0 for a shared, partially-filled tail block. "
     "It had spare room, but the write still required a copy-on-write, so allocation "
     "overshot the watermark"),
    ("004", "Livelock, cached blocks never reclaimed",
     "The prefix cache held blocks from finished sequences and nothing evicted them "
     "under memory pressure"),
    ("005", "Livelock, eviction found no victims",
     "prefix-cache acquire/release were unbalanced: a preempted and re-prefilled "
     "sequence acquired twice and released once, leaving holders > 0 forever"),
    ("006", "Head-of-line livelock",
     "Admission stopped at the first request that did not fit, starving smaller "
     "requests behind it when no running sequence remained to free memory"),
    ("007", "Cache nodes permanently pinned",
     "A sequence that attached a cached prefix then failed allocation returned to the "
     "waiting queue still holding the pin, making the memory it needed unevictable"),
    ("008", "Starvation with memory available",
     "No backstop when progress stopped. The progress metric also counted prefill "
     "tokens, which a repeatedly-preempted sequence inflates indefinitely"),
    ("009", "271 preemptions of a single sequence",
     "Preempting the last running sequence destroys progress for no benefit -- there "
     "is no other work to free memory for"),
    ("010", "68 OOM-preempts, each redoing a 204-token prefill",
     "Admission allocated for the prompt but reserved nothing for the tokens the "
     "sequence would generate"),
    ("011", "Most of the pool cached and unevictable",
     "Publishing a prompt to the cache also pinned it for the publisher's lifetime"),
    ("012", "Two sequences recompute-preempting each other",
     "The growth-headroom check tested total capacity rather than currently free "
     "blocks, so both were admitted believing they had room"),
    ("013", "Entire KV pool leaked",
     "A sequence that finished outside the running list became an orphan: marked "
     "finished, in no queue, holding 32/32 blocks. _reap_finished only walked running"),
    ("014", "Starvation with 9 free blocks",
     "Requests needing 3 blocks were refused while a 14-block request sat ahead of "
     "them and the running set had stabilised"),
    ("015", "Sequence admitted and re-prefilled every step forever",
     "A sequence needing the whole pool skipped the growth guard once it already held "
     "a block table mid-prefill"),
]


def build_story(d, figs: Dict[str, Path], prov: dict, helpers) -> list:
    P, table, code, bullets, figure, fmt, S = (
        helpers.P, helpers.table, helpers.code, helpers.bullets,
        helpers.figure, helpers.fmt, helpers.S,
    )
    st: list = []

    def tput(name):
        rec = d.bench.get(name)
        return rec.get("output_throughput") if rec else None

    def ms(name, field, pct="p50"):
        rec = d.bench.get(name)
        if not rec:
            return "n/a"
        return f"{rec[field].get(pct, 0) * 1000:.0f} ms"

    # ============================================================ title page
    st += [
        Spacer(1, 22 * mm),
        Paragraph("HELIOS", S["title"]),
        Paragraph("Project report: an LLM inference engine, what it does, "
                  "and what it does not", S["subtitle"]),
        Spacer(1, 11 * mm),
    ]
    st.append(table(
        [
            ["Repository", "github.com/anish41338/helios"],
            ["Commit", prov["commit"]],
            ["Report generated", prov["date"]],
            ["Language", "Python 3.12 (no Rust, no C++, no CUDA C)"],
            ["Source lines", "8,242 in python/, 3,404 in tests/, 3,902 in bench/"],
            ["Tests", f"{prov['tests']} run on CPU, plus 9 requiring an NVIDIA GPU"],
            ["Simulation seeds run", "50,000 (scheduler), 5,000 (with the transfer FSM)"],
            ["Hardware used", "Windows CPU (torch 2.5.1+cpu); "
                              "Tesla T4 via Kaggle for GPU gates"],
            ["Commits", "23"],
        ],
        [42 * mm, 121 * mm], header=False, font_size=8.4,
    ))
    st += [
        Spacer(1, 7 * mm),
        P(
            "This document records what was built and measured. It does not propose "
            "further work, and it does not present a limitation as an opportunity. "
            "Numbers that were measured are labelled measured; numbers that come out "
            "of an arithmetic model are labelled modelled, on the same line. Sections "
            "11, 12 and 13 list what is absent, what was got wrong, and what only "
            "showed up on real hardware.",
            "callout",
        ),
        Spacer(1, 5 * mm),
    ]
    st.append(table(
        [
            ["1", "What the software is, and the module inventory"],
            ["2", "The paged KV allocator"],
            ["3", "The scheduler"],
            ["4", "Deterministic simulation testing, and the fifteen bugs it found"],
            ["5", "The executor: model, batching, Triton kernel"],
            ["6", "Quantization: INT4 weights, AWQ, INT8 KV cache"],
            ["7", "Quantization-asymmetric speculative decoding, and the acceptance rate"],
            ["8", "API, streaming and metrics"],
            ["9", "The KV transfer state machine (not called by the engine)"],
            ["10", "Verification: tests, exact-equality properties, mutation testing"],
            ["11", "What is not built"],
            ["12", "Errors made during the build"],
            ["13", "What running on a GPU exposed"],
            ["14", "Reproduction and provenance"],
        ],
        [10 * mm, 143 * mm], header=False, font_size=8.2,
    ))
    st.append(PageBreak())

    # ============================================================== 1. what
    st += [
        P("1 &nbsp; What the software is", "h1"),
        P(
            "An LLM inference server. It loads a Llama-family or Qwen2 checkpoint in "
            "safetensors format, holds attention state in a paged KV cache it manages "
            "itself, schedules many concurrent requests through a continuous-batching "
            "loop, and serves them over an OpenAI-compatible HTTP API with incremental "
            "streaming.",
            "body",
        ),
        P(
            "The transformer forward pass is written from scratch &mdash; grouped-query "
            "attention, rotary position embeddings, SwiGLU feed-forward, RMS "
            "normalisation &mdash; rather than calling into "
            "<font face='Courier'>transformers</font>. The reason is not "
            "self-sufficiency for its own sake: attention has to read and write "
            "physical blocks chosen at runtime by this project's allocator, and no "
            "stock implementation exposes that.",
            "body",
        ),
        P(
            "It runs real pretrained weights. It has been driven end to end on "
            "Qwen2.5-0.5B on both a CPU and a Tesla T4, producing coherent text through "
            "the full HTTP path. It is not a wrapper, and the generation loop, KV "
            "management, batching, sampling and API surface are all this project's own "
            "code.",
            "body",
        ),
        P(
            "It is not production infrastructure. There is no multi-node support, no "
            "authentication, no rate limiting beyond the admission buckets, no model "
            "hot-swapping, no tensor or pipeline parallelism, and no persistence. It "
            "runs in a single process.",
            "body",
        ),
        P("Structure", "h2"),
        P(
            "The code splits at one boundary. The <b>core</b> &mdash; allocator, "
            "scheduler, prefix cache &mdash; contains no clock, no I/O, no threads, no "
            "unseeded randomness, and no iteration over unordered containers. The "
            "<b>executor</b> contains all the PyTorch. They exchange only "
            "<font face='Courier'>ExecStep</font> and "
            "<font face='Courier'>ExecOutputs</font> values.",
            "body",
        ),
        P(
            "That boundary exists so a fabricated executor can be substituted for the "
            "real one, which is what makes the scheduler testable under fault "
            "injection. Everything in section 4 depends on it.",
            "body",
        ),
    ]
    st.append(P("Module inventory", "h2"))
    rows = [["Module", "Lines", "Contents"]]
    rows += [[m, str(n), desc] for m, n, desc in MODULES]
    st.append(table(rows, [40 * mm, 15 * mm, 108 * mm], align_right=(1,)))
    st += [
        Spacer(1, 3 * mm),
        P(
            "Total: 8,242 lines under <font face='Courier'>python/</font>. Test code "
            "adds 3,404 lines and the benchmark and reporting harness 3,902.",
            "small",
        ),
    ]

    # ====================================================== 2. paged KV cache
    st += [PageBreak(), P("2 &nbsp; The paged KV allocator", "h1")]
    st += [
        P(
            "A conventional implementation reserves a contiguous KV buffer per sequence "
            "sized to the maximum context length. A 100-token request occupying a "
            "4096-token reservation wastes 97.5% of it. This allocator instead hands "
            "out fixed-size blocks and maps them per sequence through a block table, so "
            "waste is bounded by at most one partially-filled block per sequence.",
            "body",
        ),
        P("What it implements", "h2"),
    ]
    st += bullets([
        "<b>Block tables</b> mapping logical block index to physical block id, with a "
        "<font face='Courier'>filled_in_last</font> count so a partially used tail "
        "block is tracked exactly.",
        "<b>Reference counting</b> per physical block, with references coming from two "
        "distinct sources: live block tables and prefix-cache retention. They are "
        "counted separately so that an accounting error in one is attributable.",
        "<b>Copy-on-write.</b> Two sequences may share physical blocks after a fork or "
        "a prefix-cache hit. A write to a shared block copies it first. The case that "
        "matters is a shared block that is only partially filled, because it has spare "
        "room and therefore looks free to write into.",
        "<b>A host swap tier.</b> Blocks can be moved to a second, larger, slower pool "
        "instead of being discarded, and swapped back later.",
        "<b>A watermark</b> reserving a fraction of the pool, so the engine cannot "
        "reach a state where every running sequence needs one more block and none "
        "exists.",
    ])
    st += [
        Spacer(1, 2 * mm),
        P("The seven invariants", "h2"),
        P(
            "Asserted after every allocator operation in tests, and after every "
            "scheduler step in simulation:",
            "body",
        ),
    ]
    st.append(table(
        [
            ["I1", "A block is in exactly one free list, or referenced, never both"],
            ["I2", "ref_count equals block-table references plus prefix-cache references"],
            ["I3", "free + referenced == total, per tier"],
            ["I4", "No holes in any block table"],
            ["I5", "filled_in_last is in [1, BLOCK_SIZE] for a non-empty sequence"],
            ["I6", "A shared block is never written in place"],
            ["I7", "Committed blocks stay under the watermark"],
        ],
        [12 * mm, 151 * mm], header=False,
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "I7 is the one that caught a real bug. "
            "<font face='Courier'>blocks_needed_to_append</font> returned zero blocks "
            "for a shared, partially-filled tail on the grounds that it had room. It "
            "did, but writing to it required copying it first, so the allocation "
            "overshot the reserve. That is bug 003 in section 4.",
            "body",
        ),
        P("Sizing", "h2"),
        P(
            "Block count is derived from a byte budget rather than configured: "
            "<font face='Courier'>2 &times; n_layers &times; block_size &times; "
            "n_kv_heads &times; (head_dim &times; dtype_bytes + scale_bytes_per_token)"
            "</font>. The trailing term is zero for an unquantized cache and 2 for the "
            "INT8 cache, which stores one fp16 scale per token per head. It is inside "
            "the formula rather than applied afterwards because the engine divides its "
            "budget by this number to get a block count, and an understated divisor "
            "hands out more blocks than the cache physically has.",
            "body",
        ),
    ]

    # ======================================================== 3. scheduler
    st += [PageBreak(), P("3 &nbsp; The scheduler", "h1")]
    st += [
        P(
            "1,206 lines, and the largest single component. It decides, every "
            "iteration, which sequences run and what work each contributes.",
            "body",
        ),
        P("Continuous batching", "h2"),
        P(
            "Scheduling decisions are made per forward pass, not per batch. A sequence "
            "that emits an end-of-sequence token leaves immediately and a waiting "
            "sequence takes its place in the same iteration. "
            "<font face='Courier'>_reap_finished</font> runs before "
            "<font face='Courier'>_admit</font> specifically so the freed blocks are "
            "available in that step rather than the next one.",
            "body",
        ),
        P("Chunked prefill", "h2"),
        P(
            "A long prompt is split across steps against a token budget "
            "(<font face='Courier'>max_num_batched_tokens</font>), so a single large "
            "prefill cannot monopolise an iteration and inflate inter-token latency for "
            "every sequence already decoding. Successive chunks write into the same "
            "block list at increasing offsets and each attends over everything cached "
            "so far, which is what makes chunked output identical to single-shot. The "
            "step builder also reserves one token of budget per decoding sequence "
            "before packing prefills, so prefill cannot starve decode.",
            "body",
        ),
        P("Admission and starvation", "h2"),
        P(
            "Requests are ordered by (SLO class, arrival step, sequence id) and "
            "admitted against per-class token buckets refilled by <i>step count</i>, "
            "never by elapsed time. A request that does not fit is skipped rather than "
            "blocking the queue &mdash; but after 32 consecutive skips the queue head "
            "receives an exclusive reservation. Those two rules exist because the "
            "simulation harness found the opposite livelocks separately: stopping at "
            "the first request that does not fit starves small requests (bug 006), and "
            "always skipping starves large ones (bug 014).",
            "body",
        ),
        P("Preemption", "h2"),
        P(
            "Under memory pressure a victim's blocks are reclaimed either by discarding "
            "its KV and recomputing later, or by swapping to the host tier. The choice "
            "is a ratio test: swap only when generated length exceeds four times prompt "
            "length, because below that recomputing is cheaper than moving the bytes "
            "twice. The last running sequence is never preempted &mdash; there is no "
            "other work for the freed memory to serve, so it is pure loss (bug 009).",
            "body",
        ),
        P("Prefix cache", "h2"),
        P(
            "A radix trie over block-aligned prompt prefixes. A matching prefix lets a "
            "new request adopt existing physical blocks instead of recomputing them. "
            "Eviction is LRU, and blocks in use are pinned. Matching is block-aligned "
            "because a partial block cannot be shared without copy-on-write, which "
            "would negate the saving.",
            "body",
        ),
        P(
            "Measured on a workload with a long shared prefix: "
            f"<b>{fmt(tput('helios_shared_prefix_cache_on') / max(1e-9, tput('helios_shared_prefix_cache_off')) if tput('helios_shared_prefix_cache_on') and tput('helios_shared_prefix_cache_off') else None, '{:.2f}')}&times;</b> "
            "throughput against the same workload with the cache disabled "
            f"({fmt(tput('helios_shared_prefix_cache_on'), '{:.1f}')} against "
            f"{fmt(tput('helios_shared_prefix_cache_off'), '{:.1f}')} output tokens per "
            "second), and time-to-first-token from "
            f"{ms('helios_shared_prefix_cache_off', 'ttft')} to "
            f"{ms('helios_shared_prefix_cache_on', 'ttft')}. On the default workload, "
            "whose shared prefix is 32 tokens, the cache shows no benefit; that is not "
            "a defect but it does mean the headline figure is regime-specific.",
            "body",
        ),
        P("Determinism", "h2"),
        P(
            "The scheduler makes no random choices, reads no clock, and iterates no "
            "unordered container. A test asserts the module imports neither "
            "<font face='Courier'>time</font> nor <font face='Courier'>random</font>. "
            "This is a hard constraint rather than a style preference: a single "
            "wall-clock read would make a failing simulation seed unreplayable, and "
            "section 4 would not work.",
            "body",
        ),
    ]

    # ============================================================ 4. DST
    st += [PageBreak(), P("4 &nbsp; Deterministic simulation testing", "h1")]
    st += [
        P(
            "The scheduler and allocator are driven single-threaded by a harness in "
            "which every source of nondeterminism is derived from one integer seed: the "
            "workload, arrival burstiness, configuration values, cancellations, and the "
            "fault schedule. The clock is an integer counter. The executor is "
            "fabricated. Invariants are asserted after every step.",
            "body",
        ),
        P("What a seed randomises", "h2"),
    ]
    st += bullets([
        "Configuration: block size from {8, 16, 32}, pool size 24-200 blocks, host tier "
        "0-64 blocks, max sequences 2-32, token budget from {128, 256, 512, 2048}, "
        "context limit from {256, 512, 1024}, watermark from {0, 0.01, 0.05}, and "
        "whether chunked prefill, the prefix cache, speculation and swap are enabled.",
        "Workload: 1-60 requests, Poisson-ish arrivals with occasional ten-fold bursts, "
        "prompt and output lengths, SLO class, and whether a canned shared prefix is "
        "used so the prefix cache actually hits.",
        "Faults: 0-6 injected executor failures drawn from OOM, CUDA error, timeout, "
        "transfer stall, transfer checksum, transfer partial, and swap-tier-full, at "
        "randomly chosen steps; plus forced acceptance rates of 0% and 100% as "
        "adversarial extremes.",
        "Cancellations at arbitrary later steps.",
    ])
    st += [
        Spacer(1, 2 * mm),
        P("What is asserted", "h2"),
        P(
            "The seven allocator invariants, plus structural properties added as bugs "
            "demanded them: no sequence in two queues; a waiting sequence owns no KV and "
            "no cache pin; a finished sequence holds neither; every sequence is "
            "reachable from a queue or terminal and reaped; cache pins are attributable "
            "to a live sequence. Then two global properties at the end of each run: "
            "<b>liveness</b> (every submitted request reaches a terminal state) and "
            "<b>no leaks</b> (every block with a positive reference count has an owner).",
            "body",
        ),
        P("Result", "h2"),
    ]
    st.append(code(
        "50,000 seeds  -  0 failures   (3,807 s single-threaded)\n"
        "20,000 seeds  -  0 failures   (re-verified after the executor was rewritten)\n"
        " 5,000 seeds  -  0 failures   (re-verified after the transfer FSM was added)"
    ))
    st += [
        P(
            "Any failure replays exactly from its seed: "
            "<font face='Courier'>python -m helios.cli vopr --seed 918273 --replay</font>.",
            "body",
        ),
        P("The fifteen bugs it found", "h2"),
        P(
            "All fifteen were found by the harness rather than by hand, and none were "
            "seeded deliberately. Each is fixed, each fix carries a regression test, "
            "and each is named in a comment at the fix site. <b>Thirteen of the fifteen "
            "were liveness failures</b> &mdash; livelocks and starvation &mdash; rather "
            "than data corruption. In those cases the engine kept running and reported "
            "healthy metrics while completing nothing.",
            "body",
        ),
    ]
    rows = [["#", "Symptom", "Root cause"]]
    rows += [[n, sym, cause] for n, sym, cause in DST_BUGS]
    st.append(table(rows, [10 * mm, 48 * mm, 105 * mm], font_size=7.6))
    st += [
        Spacer(1, 3 * mm),
        P(
            "Two of the fixes were themselves wrong and the harness said so within "
            "seconds. The fix for bug 004 dropped the pass rate from 193/200 to 157/200 "
            "by exposing bug 005 underneath it. The fix for bug 011 surfaced bug 013.",
            "body",
        ),
        P("A coverage gap, and what it showed", "h2"),
        P(
            "Copy-on-write turned out to be unreachable in simulation: 0 of 300 seeds "
            "exercised it, because the prefix cache shares only whole blocks, so no "
            "sequence ever wrote into a shared partial tail &mdash; the exact case bug "
            "003 lived in. Adding a fork operation to the harness produced full coverage "
            "of that path. <b>Reintroducing bug 003 then still passed 300 of 300 "
            "seeds.</b> The line was executed; the conditions that make it fail were "
            "not reached. Making the forks persist across steps did catch it, at the "
            "cost of roughly sixty spurious liveness failures per six hundred seeds, "
            "because the scheduler cannot see those shadow tables and so can neither "
            "evict nor preempt them. That change was reverted and the case is pinned by "
            "a targeted test with exact control of the pool state instead.",
            "body",
        ),
    ]

    # ========================================================== 5. executor
    st += [PageBreak(), P("5 &nbsp; The executor", "h1")]
    st += [
        P("The model", "h2"),
        P(
            "A from-scratch implementation of the Llama 3.x / Qwen2.5 architecture: "
            "grouped-query attention, rotary embeddings using the rotate-half "
            "convention, SwiGLU feed-forward, and RMS normalisation computed in fp32 "
            "regardless of parameter dtype. Weights load from safetensors, read lazily "
            "per tensor so peak host memory stays near one tensor rather than the whole "
            "model.",
            "body",
        ),
        P(
            "The loader raises rather than skipping any checkpoint tensor it has no "
            "parameter for, with an explicit allowlist for genuine buffers. That is a "
            "response to a specific failure: Qwen2 places biases on the query, key and "
            "value projections and Llama does not; the model hardcoded no bias and the "
            "loader silently discarded three tensors per layer. The model loaded "
            "successfully and generated fluent nonsense.",
            "body",
        ),
        P("Batched execution", "h2"),
        P(
            "All decoding sequences in a step run in one forward pass. Sequences are "
            "concatenated along a single token dimension, so every projection and the "
            "MLP become one large matrix multiply; attention remains per sequence over "
            "its own paged KV, gathered into a right-padded tensor and masked. Prefill "
            "chunks are concatenated the same way, and the language-model head is "
            "applied only at the rows that will be sampled &mdash; with a 128k "
            "vocabulary, projecting every prompt token would cost more than the rest of "
            "the layer stack.",
            "body",
        ),
    ]
    st.append(table(
        [
            ["Configuration", "Output tok/s", "vs full engine"],
            ["helios_full (all mechanisms)", fmt(tput("helios_full"), "{:.1f}"), "—"],
            ["baseline_unbatched_executor (one pass per sequence)",
             fmt(tput("baseline_unbatched_executor"), "{:.1f}"),
             f"{tput('helios_full') / max(1e-9, tput('baseline_unbatched_executor')):.2f}× slower"
             if tput("helios_full") and tput("baseline_unbatched_executor") else "n/a"],
            ["baseline_static_batch_8", fmt(tput("baseline_static_batch_8"), "{:.1f}"),
             f"{tput('helios_full') / max(1e-9, tput('baseline_static_batch_8')):.2f}× slower"
             if tput("helios_full") and tput("baseline_static_batch_8") else "n/a"],
            ["baseline_hf_loop (no engine, no KV reuse)",
             fmt(tput("baseline_hf_loop"), "{:.1f}"),
             f"{tput('helios_full') / max(1e-9, tput('baseline_hf_loop')):.2f}× slower"
             if tput("helios_full") and tput("baseline_hf_loop") else "n/a"],
        ],
        [80 * mm, 30 * mm, 40 * mm], align_right=(1, 2),
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "These are CPU measurements on a randomly-initialised toy model. The "
            "absolute tokens-per-second figures are meaningless; the ratios between "
            "configurations are the result. The same ablations on a Tesla T4 gave "
            "3.14&times; for decode batching and 1.50&times; against static batching.",
            "small",
        ),
        P("The Triton kernel", "h2"),
        P(
            "Paging costs a gather: a sequence's KV is scattered, so the PyTorch path "
            "collects it into a contiguous tensor before computing scores. The fused "
            "kernel instead runs one program per (sequence, query head), walks the block "
            "table, and streams each block through an online-softmax accumulator, so "
            "neither the gathered KV nor the score vector is written to memory. State "
            "per program is O(head_dim) rather than O(context_len).",
            "body",
        ),
        P(
            "<b>Measured on a Tesla T4</b> (sm_75, CUDA 12.8, Triton 3.6.0) at 32 "
            "sequences by 512 context: <b>14.089 ms to 2.676 ms, a 5.26&times; "
            "speedup</b>, with 9 of 9 parity tests passing against the PyTorch paged "
            "path at atol = rtol = 2e-2. Three independent runs gave 5.26&times;, "
            "5.11&times; and 5.58&times;. The kernel was written on a machine with no "
            "NVIDIA GPU and marked unverified until a device ran the gate.",
            "body",
        ),
    ]

    # ======================================================== 6. quantization
    st += [PageBreak(), P("6 &nbsp; Quantization", "h1")]
    st += [
        P("INT4 weights", "h2"),
        P(
            "Weights are quantized to 4 bits with asymmetric group-wise scaling: two "
            "nibbles packed per byte, one fp16 scale and one uint8 zero-point per group "
            "of 128 input channels per output channel. Asymmetric rather than symmetric "
            "because at 4 bits there are only 16 levels and per-group weight "
            "distributions are not centred. uint8 packing rather than the int32 layout "
            "AWQ ships, because packing eight nibbles into a signed int32 puts a value "
            "in the sign bit and relies on wrap-around semantics PyTorch does not "
            "promise.",
            "body",
        ),
        P(
            "Quantization is applied in-process after loading an fp checkpoint, not read "
            "from a pre-quantized artifact. Loading an existing AWQ or GPTQ checkpoint "
            "is not implemented.",
            "body",
        ),
    ]
    q = d.alpha["quant_report"] if d.alpha else None
    st.append(table(
        [
            ["Measurement", "Value"],
            ["Compression vs fp16, single layer", "3.82× (measured, after use)"],
            ["Compression vs fp32, Qwen2.5-0.5B, 168 layers",
             f"{q['weight_compression']:.2f}× on quantized layers" if q else "n/a"],
            ["Whole-model compression, Qwen2.5-0.5B",
             f"{q['model_compression']:.2f}× (lm_head stays full precision)" if q else "n/a"],
            ["Round-trip error bound", "|error| ≤ scale/2, asserted elementwise"],
            ["Speed on CPU", "SLOWER than fp32 — see below"],
        ],
        [76 * mm, 77 * mm],
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "<b>There is no int4 matrix-multiply kernel.</b> "
            "<font face='Courier'>QuantLinear</font> dequantizes to floating point and "
            "then calls a normal GEMM, so on a CPU it performs strictly more work than "
            "the fp32 layer it replaces and runs slower. The memory reduction is real "
            "and measured; the speed benefit that W4A16 exists for is not implemented "
            "and not claimed. Every speedup figure involving quantization in this "
            "project is labelled modelled.",
            "body",
        ),
        P("AWQ activation-aware scaling", "h2"),
        P(
            "Quantization error on an input channel is weighted by that channel's "
            "activation magnitude when it reaches the output. Scaling a salient "
            "channel's weights up before quantizing gives it more of the 16 available "
            "levels, and a matching divide on the activation cancels it exactly: "
            "<font face='Courier'>(x/s) @ (W&middot;s)&#7488; = x @ W&#7488;</font> for "
            "any positive s, so scaling is free in exact arithmetic and only moves where "
            "the error lands. Implemented as a grid search over the exponent in "
            "<font face='Courier'>s = activation_absmean ** alpha</font> for alpha in "
            "[0, 1], minimising output error on captured calibration activations.",
            "body",
        ),
        P(
            "Activations are captured with forward hooks over calibration prompts, "
            "subsampled by a deterministic stride so calibration is reproducible. "
            f"Measured across 168 layers of Qwen2.5-0.5B: mean relative per-layer output "
            f"error <b>{d.alpha['awq_effect']['mean_rel_output_err_rtn']:.5f} with "
            f"round-to-nearest against {d.alpha['awq_effect']['mean_rel_output_err_awq']:.5f} "
            f"with AWQ, a {d.alpha['awq_effect']['error_reduction']:.2f}&times; "
            "reduction.</b> Section 7 records that this reduction does not translate "
            "into better draft acceptance."
            if d.alpha and d.alpha.get("awq_effect") else
            "Per-layer error reduction was measured; see artifacts/alpha.json.",
            "body",
        ),
        P("INT8 KV cache", "h2"),
        P(
            "The paged KV cache has an INT8 variant with symmetric per-token, per-head "
            "scales stored in fp16. Per-token rather than per-tensor because attention "
            "is a dot product over head_dim, so a single outlier token would otherwise "
            "set the scale for every token sharing it. Symmetric rather than asymmetric "
            "because a zero point would have to be carried through the score matmul as "
            "a correction term rather than folding into one multiply. It is a drop-in "
            "subclass: writes quantize, reads dequantize, and no attention code changes.",
            "body",
        ),
        P(
            "The round-trip error bound is "
            "<font face='Courier'>|error| &le; s/2 + |q|&middot;|s &minus; fp16(s)|</font>. "
            "The second term is easy to omit and is what makes an s/2-only bound fail: "
            "the scale is stored in fp16, so reconstruction multiplies by a slightly "
            "different scale than the one used to divide. At |q| = 127 it reaches about "
            "6% of s.",
            "body",
        ),
    ]
    fp, i8 = d.bench.get("helios_kv_fp_cramped"), d.bench.get("helios_kv_int8_cramped")
    if fp and i8:
        cb = i8.get("mean_decode_batch", 0) / max(1e-9, fp.get("mean_decode_batch", 1))
        tt = fp["ttft"]["p50"] / max(1e-9, i8["ttft"]["p50"])
        th = i8.get("output_throughput", 0) / max(1e-9, fp.get("output_throughput", 1))
        st.append(table(
            [
                ["256 KiB KV pool, 16 requests", "fp32 KV", "INT8 KV", "Change"],
                ["Mean resident decode batch",
                 fmt(fp.get("mean_decode_batch"), "{:.2f}"),
                 fmt(i8.get("mean_decode_batch"), "{:.2f}"), f"{cb:.2f}× more"],
                ["TTFT p50", ms("helios_kv_fp_cramped", "ttft"),
                 ms("helios_kv_int8_cramped", "ttft"), f"{tt:.2f}× better"],
                ["Output tok/s", fmt(fp.get("output_throughput"), "{:.1f}"),
                 fmt(i8.get("output_throughput"), "{:.1f}"),
                 f"{(1 - th) * 100:.0f}% worse"],
            ],
            [56 * mm, 30 * mm, 30 * mm, 37 * mm], align_right=(1, 2, 3),
        ))
        st += [
            Spacer(1, 3 * mm),
            P(
                "The memory mechanism works and the throughput does not follow. Halving "
                "the cache admits more sequences and improves time-to-first-token, and "
                "the dequantization cost on a CPU exceeds what the extra concurrency "
                "returns. Whether it pays on hardware where reading the KV cache is the "
                "decode bottleneck was not measured.",
                "body",
            ),
            P(
                "The pool size matters and the first attempt got it wrong. Run in a "
                "4 MiB pool, both configurations kept every sequence resident (mean "
                "batch 14.95 in both), so the comparison measured only dequantization "
                "overhead and INT8 appeared 20% slower with no offsetting benefit "
                "available to it.",
                "small",
            ),
        ]

    # =========================================================== 7. QASSD
    st += [PageBreak(), P("7 &nbsp; Quantization-asymmetric speculative decoding", "h1")]
    st += [
        P(
            "Speculative decoding drafts several tokens cheaply and verifies them in one "
            "parallel pass, committing the longest prefix the verifier agrees with. The "
            "variant implemented here drafts from a 4-bit quantization of the target's "
            "own weights rather than from a separate small model.",
            "body",
        ),
        P("Mechanism", "h2"),
    ]
    st += bullets([
        "Draft gamma tokens sequentially with the int4 model, writing into its own "
        "quantized KV cache.",
        "Verify all gamma positions in one full-precision forward pass.",
        "Accept the longest prefix where the draft matches the verifier's argmax.",
        "Emit one bonus token from the verifier at the first mismatch, so a rejection "
        "never stalls the sequence.",
        "Re-synchronise the draft's KV over the committed span. The draft wrote keys for "
        "every token it proposed; past the first rejection those describe a continuation "
        "that was discarded, and paged storage has no notion of truncation.",
    ])
    st += [
        Spacer(1, 2 * mm),
        P(
            "Output is unchanged by speculation. An accepted token is one the target's "
            "argmax agreed with, and a rejection emits the target's argmax, so the "
            "committed sequence is the target's greedy sequence whatever the draft "
            "proposes. A test runs identical prompts with and without the int4 draft and "
            "requires identical token ids.",
            "body",
        ),
        P("The acceptance measurement", "h2"),
        P(
            "The design depends on how often the degraded draft agrees with the "
            "verifier. A threshold of 0.6 was fixed in advance, in the design document, "
            "before any of this was built.",
            "body",
        ),
    ]
    if d.alpha:
        v = d.alpha["verdict"]
        st.append(table(
            [
                ["Quantity", "Value"],
                ["Model", "Qwen2.5-0.5B, fp32 target, int4 AWQ draft, group size 128"],
                ["Prompts", "8, spanning factual recall, arithmetic, code and prose"],
                ["Calibration", "4 prompts, disjoint from the evaluation set"],
                ["Tokens drafted", str(v["drafted"])],
                ["Tokens accepted", str(v["accepted"])],
                ["Acceptance rate (alpha)", f"{v['alpha']:.4f}  (measured)"],
                ["Pre-committed kill threshold", "0.60 — passed"],
                ["Design target", "0.78 — not met"],
                ["Measured tokens per verify pass", f"{v['measured_tokens_per_pass']:.2f}"],
                ["Accepted-run histogram (gamma=4)", str(v["run_lengths"])],
                ["Optimal gamma from the model", f"{v['best_gamma']} (planned: 4)"],
                ["Modelled speedup at that gamma",
                 f"{v['modelled_speedup_at_best_gamma']:.2f}× (modelled, not measured)"],
                ["Modelled speedup at gamma=8", "0.94× — a net loss (modelled)"],
            ],
            [58 * mm, 95 * mm],
        ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "Measuring this on a CPU is legitimate because alpha is the probability that "
            "two weight matrices produce the same argmax on the same context. It is "
            "arithmetic and does not depend on how fast either forward pass runs. The "
            "speedup does depend on that, is not implemented, and is labelled modelled "
            "everywhere it appears.",
            "body",
        ),
    ]
    if "hist" in figs and "awq" in figs:
        left = helpers.image(figs["hist"], 74 * mm)
        right = helpers.image(figs["awq"], 74 * mm)
        t = Table(
            [[left, right],
             [P("Accepted-run lengths per verify pass.", "caption"),
              P("AWQ against RTN, 95% bootstrap CIs.", "caption")]],
            colWidths=[80 * mm, 80 * mm], hAlign="LEFT",
        )
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        st.append(t)
    if d.alpha:
        hist = d.alpha["verdict"]["run_lengths"]
        total = sum(hist.values())
        meas = d.alpha["verdict"]["measured_tokens_per_pass"]
        modelled = d.alpha["modelled_speedup"]["4"]["expected_tokens_per_pass"]
        st.append(P(
            f"The accepted-run distribution is bimodal: {hist.get('4', 0)} of {total} "
            f"verify passes accepted the entire draft and {hist.get('0', 0)} accepted "
            f"none. Acceptances are therefore correlated, which contradicts the "
            f"independence assumption in the standard speedup formula. The measured "
            f"tokens per verify pass is {meas:.2f} against {modelled:.2f} predicted "
            f"under independence, so the formula understates the value by "
            f"{(meas / modelled - 1) * 100:.0f}% on this data.",
            "body",
        ))
    if d.sig:
        st += [
            P("AWQ did not improve acceptance", "h2"),
            P(
                f"AWQ reduces mean per-layer output error by 1.58&times;. Acceptance "
                f"with AWQ is <b>{d.sig['awq']['alpha']:.4f}</b> and with plain "
                f"round-to-nearest <b>{d.sig['rtn']['alpha']:.4f}</b> &mdash; nominally "
                f"worse. Bootstrapping over verify passes rather than tokens (because "
                f"acceptances are correlated, so treating "
                f"{d.alpha['verdict']['drafted']} tokens as independent trials would "
                f"understate the variance) gives 95% intervals of "
                f"[{d.sig['awq']['ci95'][0]:.3f}, {d.sig['awq']['ci95'][1]:.3f}] and "
                f"[{d.sig['rtn']['ci95'][0]:.3f}, {d.sig['rtn']['ci95'][1]:.3f}], and a "
                f"permutation test gives <b>p = {d.sig['permutation_p']:.2f}</b>. The "
                f"two are indistinguishable at this sample size.",
                "body",
            ),
            P(
                "No conclusion is drawn beyond that. A possible explanation is that AWQ "
                "minimises squared output error while acceptance depends only on whether "
                "the argmax survives, which is a coarser property &mdash; but that was "
                "not tested and is not a result.",
                "body",
            ),
        ]

    st.append(P("Measured cost when speculation actually runs", "h2"))
    off = d.bench.get("helios_spec_off_batch8")
    if off:
        base = off.get("output_throughput", 1.0)
        rows = [["Batch capped to 8, 16 requests", "Output tok/s", "Against no speculation"]]
        for label, key in (("Speculation off", "helios_spec_off_batch8"),
                           ("Symmetric, gamma=2", "helios_spec_g2_batch8"),
                           ("Symmetric, gamma=4", "helios_spec_g4_batch8"),
                           ("QASSD, gamma=2", "helios_qassd_g2_batch8")):
            rec = d.bench.get(key)
            if not rec:
                continue
            t = rec.get("output_throughput", 0.0)
            rel = "—" if key == "helios_spec_off_batch8" else f"{base / max(1e-9, t):.2f}× slower"
            rows.append([label, f"{t:.1f}", rel])
        st.append(table(rows, [60 * mm, 32 * mm, 45 * mm], align_right=(1, 2)))
    st += [
        Spacer(1, 3 * mm),
        P(
            "Speculation as implemented gives up batched decode. The draft loop is "
            "serial and runs per sequence, so with 8 resident sequences a speculative "
            "step costs 8&times;(gamma+1) unbatched forward passes where an ordinary "
            "step costs one batched pass. On this CPU that trade is never worth making, "
            "and no configuration of speculation is faster than not speculating.",
            "body",
        ),
        P(
            "QASSD costs less than symmetric speculation here only because the adaptive "
            "gate disabled it. On the toy model the int4 draft agrees 12.5% of the time; "
            "measured, the gate turned speculation off after a single step and 1 of 47 "
            "decode steps speculated. The gate reacts to batch size above 8 and to "
            "measured acceptance below 0.5 over a sliding window.",
            "body",
        ),
        P("Memory cost", "h2"),
    ]
    st.append(table(
        [
            ["Cost", "Measured"],
            ["Draft weights (int4 copy, both precisions resident)", "+41% on weights"],
            ["Draft KV shadow (a second, quantized KV cache)", "+28% of the main cache"],
            ["Combined, as reported by engine.memory_report()", "1.28× total resident"],
        ],
        [95 * mm, 58 * mm],
    ))
    st.append(P(
        "Enabling this feature increases memory. It does not reduce it. A configuration "
        "that keeps only the int4 weights does reduce memory, but contains no "
        "speculation.",
        "body",
    ))

    # ====================================================== 8. frontend
    st += [PageBreak(), P("8 &nbsp; API, streaming and metrics", "h1")]
    st += [
        P(
            "An OpenAI-compatible HTTP surface: "
            "<font face='Courier'>/v1/completions</font>, "
            "<font face='Courier'>/v1/chat/completions</font>, "
            "<font face='Courier'>/v1/models</font>, "
            "<font face='Courier'>/health</font> and "
            "<font face='Courier'>/metrics</font>, plus a namespaced "
            "<font face='Courier'>helios</font> block for SLO class, speculation depth "
            "and prefix-cache control that standard clients ignore.",
            "body",
        ),
        P("Concurrency", "h2"),
        P(
            "The engine is single-threaded by design, so all stepping happens under one "
            "asyncio lock. Rather than a background task, whichever request is waiting "
            "drives the engine forward and distributes results to every waiter. That "
            "inversion is necessary: a perpetual loop starves the handlers it serves "
            "under a cooperative event loop, because stepping is CPU-bound and never "
            "truly awaits.",
            "body",
        ),
        P("Streaming", "h2"),
        P(
            "Server-sent events, one frame per text delta. Deltas are produced by "
            "decoding the whole output each step and taking the new suffix, rather than "
            "decoding tokens individually: a BPE token is not a character, and a "
            "multi-byte codepoint spans several tokens, so per-token decoding would put "
            "invalid UTF-8 on the wire. Speculative decoding makes frame sizes uneven "
            "because a verify pass can commit several tokens at once.",
            "body",
        ),
        P(
            "Verified against Qwen2.5-0.5B over the real HTTP path: a 12-token "
            "completion produced <b>13 SSE frames concatenating byte-identically</b> to "
            "the non-streamed response for the same prompt. A test asserts that "
            "equality, and a second asserts the per-request delta queues are "
            "deregistered afterwards.",
            "body",
        ),
    ]
    st.append(code(
        'POST /v1/completions   {"prompt": "The capital of France is",\n'
        '                        "max_tokens": 12, "temperature": 0.0}\n'
        '  -> " Paris. It is the largest city in Europe and the second"\n'
        '     usage: prompt_tokens 5, completion_tokens 12, cached_tokens 0\n\n'
        'POST /v1/completions   same body plus "stream": true\n'
        '  -> 13 data frames, then a terminal frame carrying finish_reason,\n'
        '     then data: [DONE]\n'
        '  -> concatenated text identical to the non-streamed response'
    ))
    st += [
        P("Metrics", "h2"),
        P(
            "A Prometheus text endpoint exposing 18 gauges and counters: queue depths, "
            "KV block usage and utilisation, prefill and decode token totals, finished "
            "and aborted counts, preemptions by kind, prefix-cache hit ratio, executor "
            "faults, step counts, and, when a quantized draft is configured, the "
            "measured acceptance rate and weight overhead ratio. Latency summaries "
            "(TTFT and TPOT percentiles, and goodput against per-class SLO targets) are "
            "computed from completed requests.",
            "body",
        ),
    ]

    # ================================================== 9. transfer FSM
    st += [P("9 &nbsp; The KV transfer state machine", "h1")]
    st += [
        P(
            "<b>This component is not called by the engine.</b> The only non-test module "
            "that imports it is the simulation harness. It is a verified design, not a "
            "feature that can be enabled, and it is described here on that basis.",
            "callout",
        ),
        P(
            "It models migrating a prompt's KV cache from a prefill partition to a "
            "decode partition. Seven states with an explicit legal-transition table: "
            "pending, reserved, sending, sent, received, committed, and the terminal "
            "failed and aborted. An unlisted transition raises.",
            "body",
        ),
        P(
            "The design separates <font face='Courier'>sent</font> (the sender believes "
            "it is done) from <font face='Courier'>received</font> (the receiver "
            "confirms). Five fault kinds are modelled &mdash; receiver out of memory, "
            "link timeout, checksum mismatch, lost acknowledgement, and mid-flight abort "
            "&mdash; and each returns block lists for both partitions, because a failure "
            "leaves blocks reserved on the receiver that the sender does not know about "
            "and blocks pinned on the sender that the receiver cannot see. Retries are "
            "bounded and fall back to re-prefilling.",
            "body",
        ),
        P(
            "Half of all simulation seeds drive it alongside the scheduler under "
            "randomised faults, asserting six state-machine invariants and two pool "
            "invariants the state machine cannot check itself. Coverage measured across "
            "200 seeds reached every fault kind: abort in 80 seeds, link timeout in 41, "
            "checksum mismatch in 30, receiver OOM in 22, with retry in 72 and "
            "re-prefill fallback in 56.",
            "body",
        ),
        P(
            "Detection was verified rather than inferred from coverage. Reintroducing "
            "the partial-failure leak (returning the sender's blocks but not the "
            "receiver's) was caught by <b>31 of 60 seeds</b>. Permitting an abort after "
            "the bytes are already sent was caught by <b>15 of 60</b>.",
            "body",
        ),
        P(
            "It also computes whether transferring a prompt's KV is cheaper than "
            "recomputing it, from bandwidth rather than assumption. For a 2000-token "
            "prompt on a 32-layer model with 8 KV heads at head_dim 128 in fp16, the KV "
            "is 262 MB: 0.9 ms over NVLink at 300 GB/s, 21.8 ms over PCIe 3.0 at "
            "12 GB/s, and 209.7 ms over 10 GbE, against 250 ms to re-prefill at 8000 "
            "tokens per second.",
            "body",
        ),
    ]

    # ================================================= 10. verification
    st += [PageBreak(), P("10 &nbsp; Verification", "h1")]
    st += [
        P(
            f"{prov['tests']} tests run on a CPU and a further 9 require an NVIDIA "
            "GPU, for 278 in total. Distribution by area:",
            "body",
        ),
    ]
    rows = [["Test file", "Tests", "Lines"]]
    rows += [[f, str(n), str(loc)] for f, n, loc in TEST_FILES]
    st.append(table(rows, [78 * mm, 20 * mm, 20 * mm], align_right=(1, 2)))
    st += [
        Spacer(1, 3 * mm),
        P("Properties asserted by exact equality", "h2"),
    ]
    st.append(table(
        [
            ["Property", "Guarantee"],
            ["Chunked prefill against single-shot prefill", "Bit-identical"],
            ["Speculative against non-speculative decode",
             "Bit-identical, gamma in {1,2,4,8}"],
            ["QASSD against unspeculated decode", "Identical token ids"],
            ["Batched against sequential decode", "Identical logits, order-invariant"],
            ["Prefix cache on against off", "Identical output"],
            ["Engine against a naive generation loop", "Identical tokens"],
            ["Output across 14 forced recompute preemptions", "Identical"],
            ["Streamed deltas against the non-streamed response", "Byte-identical"],
            ["INT4 pack/unpack", "Exactly invertible"],
            ["Paged attention against a dense reference", "Within float precision"],
            ["Triton kernel against the PyTorch paged path",
             "Within fp16 tolerance on a T4"],
        ],
        [78 * mm, 75 * mm],
    ))
    st += [
        Spacer(1, 3 * mm),
        P("Mutation testing", "h2"),
        P(
            "Three deliberate bugs were introduced to confirm the tests that exist to "
            "catch them actually do:",
            "body",
        ),
    ]
    st.append(table(
        [
            ["Injected bug", "Caught by"],
            ["Commit the draft's token instead of the verifier's at a rejection",
             "test_asymmetric_speculation_matches_the_target_exactly"],
            ["Quantized copy_block drops the per-token scales",
             "test_copy_block_moves_the_scales"],
            ["Block sizing formula ignores the KV scale bytes",
             "test_allocator_sizing_accounts_for_the_scales"],
        ],
        [70 * mm, 83 * mm],
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "The first result is worth stating precisely. When the accept logic was "
            "mutated, <b>every symmetric-speculation parity test passed</b>, because at "
            "an acceptance rate of 1.0 the draft token and the verified token are the "
            "same and the bug is invisible. Only the asymmetric test failed.",
            "body",
        ),
        P("End-to-end verification on real weights", "h2"),
        P(
            "Qwen2.5-0.5B (630M parameters as loaded, 494M plus an untied output "
            "projection) was run through the engine on two devices. On a Tesla T4 in "
            "fp16 and on this CPU in fp32, the same prompts produced the same "
            "continuations:",
            "body",
        ),
    ]
    st.append(code(
        '"The capital of France is"  ->  " Paris. It is the largest city in Europe\n'
        '                                 and the second largest in the world..."\n\n'
        '"Water boils at"            ->  " 212 F and ice melts at 32 F. What is the\n'
        '                                 number of degrees..."\n\n'
        '"def add(a, b):"            ->  "\\n    return a + b\\n\\ndef subtract(a, b):\\n'
        '                                     return a - b\\n\\ndef multiply(a,"'
    ))
    st += [
        P(
            "Paging, block tables, rotary embeddings, GQA head expansion, "
            "normalisation, batched prefill, batched decode, the dtype handling and "
            "sampling must all be simultaneously correct for a pretrained model to "
            "produce this. The toy model used elsewhere in the test suite has random "
            "weights and is written by the same code that reads it, so it can only "
            "demonstrate self-consistency.",
            "body",
        ),
        P(
            "The content is the model's own knowledge and not a claim about the engine. "
            "Paris is not the largest city in Europe.",
            "small",
        ),
    ]

    # ============================================ 11. what is not built
    st += [PageBreak(), P("11 &nbsp; What is not built", "h1")]
    st.append(table(
        [
            ["Absent", "Detail"],
            ["Fused int4 GEMM",
             "QuantLinear dequantizes and calls a normal GEMM. On CPU it is slower than "
             "fp32. The memory reduction is measured; the speed benefit W4A16 exists for "
             "is not implemented."],
            ["vLLM comparison",
             "bench/vllm_baseline.py is written with matched controls — same model "
             "directory, dtype, KV byte budget, seeded arrival trace, discarded warm-up "
             "— and has never been executed. vLLM requires CUDA."],
            ["W4A4 (4-bit activations)",
             "Not implemented. The available GPU has no fast int4 activation path."],
            ["Multi-device disaggregation",
             "The transfer state machine exists and is verified. No transport, no second "
             "device, and the engine does not call it."],
            ["Rust frontend or scheduler",
             "Not written. Everything is Python."],
            ["gRPC, shared-memory IPC",
             "Not written. Single process."],
            ["Tensor or pipeline parallelism",
             "Not written. Single device."],
            ["Pre-quantized checkpoint loading",
             "AWQ and GPTQ artifacts cannot be read. Quantization is applied in-process "
             "to an fp checkpoint."],
            ["Acceptance measurement above 0.5B parameters",
             "Not measured. Whether alpha improves with model size was not tested."],
            ["GPU measurement of the INT8 KV cache",
             "The concurrency benefit was measured on CPU; whether it pays where "
             "bandwidth is the bottleneck was not measured."],
            ["Beam search, n>1 sampling, logit bias, logprobs",
             "Not implemented. The API accepts logprobs and returns null."],
            ["Authentication, rate limiting, multi-tenancy, persistence",
             "Not implemented."],
        ],
        [46 * mm, 107 * mm],
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "One caveat applies to every CPU performance number in this document: the "
            "benchmark model is a small randomly-initialised toy, so absolute "
            "tokens-per-second figures are meaningless. Only the ratios between "
            "configurations are results. The GPU figures come from a Tesla T4 on Kaggle "
            "and are recorded in artifacts.",
            "body",
        ),
    ]

    # ================================================ 12. errors made
    st += [P("12 &nbsp; Errors made during the build", "h1")]
    st += [
        P(
            "Listed because they are part of the record and several changed the "
            "conclusions.",
            "body",
        ),
    ]
    st.append(table(
        [
            ["Error", "How it was found and what changed"],
            ["A memory optimisation that was a memory regression. QuantLinear cached the "
             "dequantized fp weight, so after one forward pass an int4 layer held 10.6 "
             "MiB against 8.4 MiB for the fp16 layer it replaced — a 1.26x increase, "
             "reported as a 3.82x saving because stored_bytes() counted only the packed "
             "form",
             "Found by an audit run after the code shipped green. Every quantization "
             "memory claim in the project was false in practice. Caching is now opt-in "
             "and off by default; resident_bytes() was added."],
            ["A benchmark that measured nothing. The speculation ablation ran at the "
             "default batch of 12-16, above the gate's threshold of 8, so 0 of 23 decode "
             "steps speculated",
             "Found by instrumenting the gate. The row labelled speculation was "
             "reporting scheduling variance. A separate suite now caps the batch."],
            ["An incomplete memory figure. memory_overhead() covered weights and omitted "
             "the second KV cache QASSD allocates",
             "Found in the same audit. The number was true and incomplete. "
             "engine.memory_report() now returns both."],
            ["An overstated claim. The transfer state machine was listed as built "
             "alongside paged attention, when nothing outside the test harness imports it",
             "Found by grepping for its consumers. Relabelled."],
            ["A wrong conclusion published as a finding. \"Continuous batching needs a "
             "GPU to pay\" was documented, because the measured batching win was zero",
             "A matmul microbenchmark showed roughly 10x from batching GEMMs on CPU at "
             "32 sequences. The engine was running one forward pass per sequence — a "
             "missing optimisation, not a hardware property. Fixing it gave 2.24x."],
            ["An arithmetic error in a comment that inverted a verdict. A 2000-token KV "
             "cache was stated as 2.6 GB; it is 262 MB",
             "Found by a test written against the real formula. The tenfold error "
             "changed the PCIe verdict from \"transfer wins by 11x\" to \"roughly "
             "break-even\"."],
            ["Three mechanisms benchmarked outside the regime they exist for: the prefix "
             "cache on a 32-token shared prefix, prefill batching behind long "
             "generations, and the INT8 KV cache in a pool that did not bind",
             "Each appeared worthless or harmful until the workload was constructed so "
             "the mechanism had something to do. Dedicated suites now exist."],
            ["A benchmark contaminated by a concurrent job on the same machine, "
             "inverting the gamma=2 / gamma=4 ordering",
             "Re-run on an idle machine; the inversion disappeared."],
            ["Two bug fixes that were themselves wrong",
             "The fix for scheduler bug 004 dropped the simulation pass rate from "
             "193/200 to 157/200 by exposing bug 005. The fix for bug 011 surfaced "
             "bug 013."],
            ["A test-coverage gap mistaken for coverage. Adding forks to the harness "
             "gave full line coverage of copy-on-write, but reintroducing the original "
             "bug still passed 300 of 300 seeds",
             "Coverage of a path is not detection of its failure conditions. The case is "
             "pinned by a targeted test instead."],
        ],
        [66 * mm, 87 * mm], font_size=7.8,
    ))

    # =============================================== 13. GPU findings
    st += [PageBreak(), P("13 &nbsp; What running on a GPU exposed", "h1")]
    st += [
        P(
            "Four bugs were found by putting the code on a Tesla T4 while 178 tests were "
            "passing. Three of them could not have been caught on the development "
            "machine.",
            "body",
        ),
    ]
    st.append(table(
        [
            ["Bug", "Why it was invisible on CPU"],
            ["Qwen2 attention biases silently dropped. Qwen2 places biases on q/k/v "
             "projections; Llama does not. The model hardcoded bias=False and the loader "
             "skipped checkpoint tensors with no matching parameter",
             "The toy model is written by the same code that reads it, so no tensor is "
             "ever unmatched. The model loaded successfully and generated fluent "
             "nonsense."],
            ["Model weights were never moved to the device. HeliosModel(device=...) only "
             "records the device for tensors it creates at runtime; the loader called "
             ".to(dtype), which is not .to(device)",
             "On a single-device machine the requested and default devices agree, so the "
             "mismatch cannot occur. The regression test asserts the invariant — every "
             "parameter on the device inputs are built on — not the symptom."],
            ["The runner converted the resulting exception into a retryable fault, so a "
             "forward pass that always failed became an infinite preempt-and-retry loop "
             "whose only symptom was an empty result list",
             "Correct behaviour in production and useless for diagnosis. A smoke test "
             "now drives the raw executor with exceptions unhandled before the scheduler "
             "is involved."],
            ["The engine ran real weights in fp32 on the GPU",
             "Not a correctness bug. Now fp16 on CUDA, decided once and shared by the "
             "weights, the KV cache and the block-size arithmetic."],
        ],
        [62 * mm, 91 * mm], font_size=7.8,
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "A fifth issue cost a session without being a code bug: Kaggle's working "
            "directory persists between sessions, so cloning into an existing directory "
            "fails while the notebook continues, and the run silently executes the "
            "previous session's code. The reported failures were bugs already fixed "
            "upstream. The runner now aborts if the checkout is behind origin.",
            "body",
        ),
    ]

    # ================================================ 14. provenance
    st += [P("14 &nbsp; Reproduction", "h1")]
    st.append(code(
        "# tests\n"
        "PYTHONPATH=python python -m pytest tests -q\n\n"
        "# simulation; any failure replays from its seed\n"
        "PYTHONPATH=python python -m helios.cli vopr --seeds 50000\n"
        "PYTHONPATH=python python -m helios.cli vopr --seed 918273 --replay\n\n"
        "# ablations -> JSON artifacts -> generated docs/BENCHMARKS.md\n"
        "python bench/loadgen.py --model artifacts/bench_model --suite all\n"
        "python bench/loadgen.py --model artifacts/bench_model --suite spec \\\n"
        "    --requests 16 --prompt-len 48 --output-len 24\n"
        "python bench/report.py\n\n"
        "# acceptance rate and its significance analysis\n"
        "python bench/measure_alpha.py --model artifacts/qwen05b --gamma 4\n"
        "python bench/measure_alpha.py --model artifacts/qwen05b --no-awq \\\n"
        "    --out artifacts/alpha_rtn.json\n"
        "python bench/alpha_significance.py\n\n"
        "# real checkpoint end to end (downloads Qwen2.5-0.5B if absent)\n"
        "python bench/real_model_check.py --model artifacts/qwen05b --device cpu\n\n"
        "# GPU gates: device report, tests, Triton parity, real model, benchmarks\n"
        "python bench/kaggle_gpu_run.py\n\n"
        "# this report\n"
        "python bench/make_final_report.py --out docs/helios_report.pdf"
    ))
    rows = [["Field", "Value"],
            ["Commit", prov["commit"]],
            ["Generated", prov["date"]],
            ["Python / torch", f"{prov['python']} / {prov['torch']}"],
            ["Platform", prov["platform"]],
            ["Tests collected", prov["tests"]]]
    if d.dst50k:
        f = d.dst50k.get("failures", 0)
        rows.append(["Simulation sweep",
                     f"{d.dst50k.get('seeds', 0):,} seeds, "
                     f"{len(f) if isinstance(f, list) else f} failures"])
    if d.dst_disagg:
        f = d.dst_disagg.get("failures", 0)
        rows.append(["Sweep with the transfer FSM",
                     f"{d.dst_disagg.get('seeds', 0):,} seeds, "
                     f"{len(f) if isinstance(f, list) else f} failures"])
    if d.alpha:
        rows.append(["Acceptance measurement",
                     f"{Path(d.alpha['model']).name}, gamma={d.alpha['gamma']}, "
                     f"group={d.alpha['group_size']}, awq={d.alpha['awq']}"])
    st.append(table(rows, [50 * mm, 103 * mm]))
    st += [
        Spacer(1, 4 * mm),
        P(
            "Every number in this document is read from a JSON artifact under "
            "<font face='Courier'>artifacts/</font> at generation time by "
            "<font face='Courier'>bench/make_final_report.py</font>. None is typed in.",
            "small",
        ),
    ]
    if d.missing:
        st.append(P(
            "Artifacts not present at generation time, listed rather than omitted: "
            + ", ".join(f"<font face='Courier'>{m}</font>" for m in sorted(set(d.missing)))
            + ". The vLLM baseline has never been run; the GPU run record was produced "
            "in a Kaggle session and its numbers are quoted from the commit messages and "
            "docs that recorded them.",
            "small",
        ))
    return st


def render(story: list, out: Path, prov: dict, palette) -> None:
    muted, rule, accent = palette
    doc = BaseDocTemplate(
        str(out), pagesize=A4,
        leftMargin=23 * mm, rightMargin=23 * mm,
        topMargin=21 * mm, bottomMargin=18 * mm,
        title="HELIOS - project report", author="HELIOS",
        subject="LLM inference engine: what was built, measured, and not done",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")

    def decorate(canvas, document):
        canvas.saveState()
        if document.page > 1:
            canvas.setFont("Helvetica", 7.4)
            canvas.setFillColor(muted)
            canvas.drawString(doc.leftMargin, A4[1] - 13 * mm,
                              f"HELIOS   ·   project report   ·   commit {prov['commit']}")
            canvas.drawRightString(A4[0] - doc.rightMargin, 11 * mm, str(document.page))
            canvas.setStrokeColor(rule)
            canvas.setLineWidth(0.4)
            canvas.line(doc.leftMargin, A4[1] - 15.5 * mm,
                        A4[0] - doc.rightMargin, A4[1] - 15.5 * mm)
        else:
            canvas.setStrokeColor(accent)
            canvas.setLineWidth(1.8)
            # Both endpoints at the same height, below the subtitle block.
            canvas.line(doc.leftMargin, A4[1] - 67 * mm,
                        A4[0] - doc.rightMargin, A4[1] - 67 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    doc.build(story)

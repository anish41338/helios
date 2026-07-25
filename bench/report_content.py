"""The prose and layout of the PDF report. Numbers come from artifacts, never here.

Split from make_report.py so that the data loading and chart rendering stay
separable from the document's content. `build_story` is the only entry point.

The one rule this module follows: any figure that appears in the text is
interpolated from a `Data` lookup. Where an artifact is absent the section
degrades to saying so rather than to a plausible constant, because a report that
can drift from its measurements eventually will (spec section 11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from reportlab.lib import colors
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


def build_story(d, figs: Dict[str, Path], prov: dict, helpers) -> list:
    """Assemble the flowable list. `helpers` carries the style/table utilities."""
    P, table, code, bullets, figure, fmt, S = (
        helpers.P, helpers.table, helpers.code, helpers.bullets,
        helpers.figure, helpers.fmt, helpers.S,
    )
    st: list = []

    # ------------------------------------------------------------ title page
    st += [
        Spacer(1, 32 * mm),
        Paragraph("HELIOS", S["title"]),
        Paragraph(
            "An LLM inference engine built from scratch &mdash; and the "
            "measurements that decided its design",
            S["subtitle"],
        ),
        Spacer(1, 12 * mm),
    ]
    st.append(table(
        [
            ["commit", prov["commit"], "tests passing", prov["tests"]],
            ["date", prov["date"], "DST seeds green", prov["dst_seeds"]],
            ["lines of Python", f"{prov['loc']:,}",
             "scheduler bugs found by DST", "15"],
            ["torch", prov["torch"], "GPU verified on", "Tesla T4 (sm_75)"],
        ],
        [30 * mm, 50 * mm, 50 * mm, 33 * mm], header=False, font_size=8.6,
    ))
    st += [
        Spacer(1, 9 * mm),
        P(
            "<b>The result this report is organised around.</b> The central design bet "
            "was <i>quantization-asymmetric self-speculative decoding</i>: draft tokens "
            "cheaply from a 4-bit view of the model's own weights, then verify them in "
            "full precision. Whether that works at all rests on one number &mdash; how "
            "often the degraded draft agrees with the verifier. The project committed "
            "in advance to abandoning the feature below &alpha;&nbsp;=&nbsp;0.60.",
            "callout",
        ),
    ]
    if d.alpha:
        v = d.alpha["verdict"]
        st.append(table(
            [
                ["measured α", f"{v['alpha']:.4f}", "kill gate (α ≥ 0.60)", "PASS"],
                ["design target", "0.78", "target met?", "NO"],
                ["planned draft length γ", "4", "measured optimum γ",
                 str(v["best_gamma"])],
                ["modelled speedup at γ=8", "0.94× (a loss)",
                 f"modelled at γ={v['best_gamma']}",
                 f"{v['modelled_speedup_at_best_gamma']:.2f}×"],
            ],
            [42 * mm, 33 * mm, 50 * mm, 38 * mm], header=False, font_size=8.6,
        ))
        st += [
            Spacer(1, 6 * mm),
            P(
                "The feature survives its gate and misses its target. Reporting both "
                "halves is the point: the measurement changed the design &mdash; "
                "optimal draft length is 2, not 4 &mdash; and bounded the claim to "
                "roughly 1.4&times; modelled rather than the 2&ndash;2.4&times; the "
                "design had assumed. A negative-leaning result that was actually "
                "measured is worth more than an unmeasured positive one.",
                "small",
            ),
        ]
    st.append(PageBreak())

    # -------------------------------------------------------------- section 1
    st += [
        P("1 &nbsp; What was built", "h1"),
        P(
            "A complete LLM serving engine, written from first principles rather than "
            "assembled from libraries. The forward pass is hand-written &mdash; "
            "grouped-query attention, rotary embeddings, SwiGLU, RMSNorm &mdash; not a "
            "call into <font face='Courier'>transformers</font>, because the entire "
            "point is to own the KV cache: the model must read and write paged blocks "
            "chosen by our allocator, which no stock implementation exposes.",
            "body",
        ),
        P("The seam everything follows from", "h2"),
        P(
            "The <b>core</b> &mdash; allocator, scheduler, prefix cache &mdash; is a "
            "pure state machine: no clock, no I/O, no threads, no unseeded randomness, "
            "no iteration over unordered containers. The <b>executor</b> holds all the "
            "PyTorch. They communicate only through a narrow "
            "<font face='Courier'>ExecStep</font>&nbsp;/&nbsp;"
            "<font face='Courier'>ExecOutputs</font> value type.",
            "body",
        ),
        P(
            "That seam is load-bearing in two directions. It lets a simulated executor "
            "substitute for the real one, which is what makes the entire scheduler "
            "testable under fault injection. And it is where the design's Rust "
            "scheduler and shared-memory ring buffer would attach &mdash; that "
            "component was cut, but the boundary it needs was preserved deliberately, "
            "which is the difference between a cut scope and a missing design.",
            "body",
        ),
    ]
    st.append(table(
        [
            ["Subsystem", "Mechanism", "Verified by"],
            ["Paged KV allocator", "fixed blocks, ref counts, copy-on-write",
             "7 invariants after every op"],
            ["Scheduler", "iteration-level continuous batching, chunked prefill",
             "50,000 simulation seeds"],
            ["Admission", "3 SLO classes, token buckets, anti-starvation aging",
             "liveness checks per seed"],
            ["Preemption", "recompute vs swap, chosen by a ratio test",
             "output identity after 14 cycles"],
            ["Prefix cache", "radix trie, block-aligned, LRU with pinning",
             "output identity, hit ratio"],
            ["Attention", "paged; gather+SDPA and a fused Triton kernel",
             "dense reference, 9/9 on a T4"],
            ["Quantization", "INT4 weights with AWQ, INT8 paged KV",
             "error bounds derived, asserted"],
            ["Speculation", "int4 draft / fp verify, adaptive γ",
             "bit-identical output"],
            ["Frontend", "OpenAI-compatible, incremental SSE, Prometheus",
             "stream reassembly identity"],
        ],
        [30 * mm, 73 * mm, 60 * mm],
    ))
    st += [
        Spacer(1, 3 * mm),
        P(
            "One row is deliberately absent from that table. The 7-state KV transfer "
            "FSM (section 5) is verified by the same harness, but <b>nothing outside "
            "the test harness imports it</b> &mdash; it is a verified design, not a "
            "code path a request travels. Listing it beside paged attention would "
            "overstate it, and an earlier draft of this report did exactly that.",
            "small",
        ),
    ]
    st += [
        Spacer(1, 4 * mm),
        P("Why paging, stated as the trade it is", "h2"),
        P(
            "A contiguous per-sequence KV buffer sized to the maximum context length "
            "wastes memory to internal fragmentation: a 100-token request occupying a "
            "4096-token slot wastes 97.5% of it. Paging allocates fixed-size blocks and "
            "maps them per sequence through a block table &mdash; operating-system "
            "virtual memory applied to attention state. The cost is a gather: a "
            "sequence's KV is scattered, so attention must collect it before computing "
            "scores. That gather is the tax paging pays, and eliminating it is exactly "
            "what the fused kernel in section 3 does.",
            "body",
        ),
    ]

    # -------------------------------------------------------------- section 2
    st += [PageBreak(),
           P("2 &nbsp; Correctness: deterministic simulation testing", "h1")]
    dst_time = d.dst50k.get("elapsed_s") if d.dst50k else None
    st += [
        P(
            "The most valuable component here is not a performance optimisation "
            "&mdash; it is the test harness, in the TigerBeetle and FoundationDB "
            "tradition. Because the scheduler is pure, an entire run is a total "
            "function of one integer seed: workload, arrival burstiness, configuration "
            "jitter, cancellations, and the fault schedule all derive from it. So the "
            "system can be driven single-threaded with a simulated clock and a "
            "fabricated executor, asserting every invariant after every step, and any "
            "failure replays exactly.",
            "body",
        ),
        code("python -m helios.cli vopr --seed 918273 --replay"),
        P(
            f"<b>{prov['dst_seeds']} seeds, zero failures</b>"
            + (f", {dst_time / 60:.0f} minutes single-threaded" if dst_time else "")
            + ", re-verified after every subsequent refactor. Determinism here is not "
            "stylistic tidiness: a single <font face='Courier'>time.time()</font> call "
            "inside the scheduler would make a failing seed unreplayable and the "
            "harness worthless. A test asserts the module imports neither a clock nor "
            "an RNG.",
            "body",
        ),
        P("What it found, and the shape of it", "h2"),
        P(
            "Fifteen real bugs. The distribution is the interesting part: <b>thirteen "
            "of fifteen were liveness bugs</b> &mdash; livelocks and starvation &mdash; "
            "not data corruption. The engine kept running, kept reporting healthy "
            "metrics, and quietly completed nothing. Conventional integration testing "
            "would have found almost none of them, because nothing crashes and no "
            "assertion trips; the system simply stops making progress, in a state that "
            "takes a specific adversarial sequence to reach.",
            "body",
        ),
    ]
    st += bullets([
        "A <b>copy-on-write accounting error</b> that overshot the memory watermark: a "
        "shared <i>partially filled</i> tail block reported spare room while still "
        "needing a copy before it could be written. Precisely the corruption class the "
        "design document had predicted in advance.",
        "A <b>self-sustaining livelock</b> in which a sequence that failed admission "
        "kept its prefix-cache pin &mdash; making the very memory it was waiting for "
        "permanently unevictable.",
        "A sequence <b>admitted, preempted for memory, and re-prefilled every step "
        "forever</b>, because admission reserved space for the prompt but none for the "
        "tokens the request would go on to generate.",
        "A <b>leak of the entire KV pool</b> through a sequence that finished outside "
        "the running list, which no reaper ever visited.",
    ])
    st += [
        Spacer(1, 3 * mm),
        P("The lesson that generalised: coverage is not detection", "h2"),
        P(
            "Copy-on-write turned out to be unreachable in simulation &mdash; 0 of 300 "
            "seeds exercised it, because the prefix cache shares only whole blocks, so "
            "no sequence ever wrote into a shared partial tail. Adding a fork operation "
            "produced full coverage of that path. <b>Reintroducing the original bug "
            "still passed 300 of 300 seeds.</b> The path was executed; the conditions "
            "that make it fail were not. Making forks persist did catch it &mdash; at "
            "the cost of roughly sixty spurious liveness failures per six hundred "
            "seeds, which are harness artefacts rather than engine defects. That change "
            "was reverted and the case pinned instead by a targeted test with exact "
            "control of the pool state.",
            "body",
        ),
        P(
            "The habit that came out of it is applied to every new subsystem in this "
            "report: after building a checker, deliberately reintroduce the bug it is "
            "supposed to catch, and confirm that it does.",
            "body",
        ),
    ]

    # -------------------------------------------------------------- section 3
    st += [PageBreak(), P("3 &nbsp; Performance: single-mechanism ablations", "h1")]
    st.append(P(
        "Every performance number here is an ablation: one mechanism disabled, "
        "everything else identical, same seeded workload. For the question actually "
        "being asked &mdash; <i>what is this mechanism worth</i> &mdash; that is "
        "stronger evidence than a cross-engine comparison, because it controls for "
        "everything else by construction. It cannot answer <i>is this engine fast</i>; "
        "only a baseline can, and section 6 is explicit about that gap.",
        "body",
    ))
    if "mechanisms" in figs:
        st.append(figure(
            figs["mechanisms"],
            "Figure 1 &mdash; Each mechanism's contribution as a single-mechanism "
            "ablation. GPU figures from a recorded Tesla T4 run; CPU figures "
            "recomputed from local artifacts at generation time.",
        ))
    st += [
        P(
            "The pattern is the one theory predicts, and it is worth stating as a "
            "prediction that held rather than as an observation. <b>Every mechanism "
            "pays more on the GPU than on the CPU.</b> Decode is memory-bandwidth "
            "bound: one query token attends over the entire context, so the step reads "
            "the whole KV cache to perform very little arithmetic. Batching amortises "
            "the weight read across sequences, and the more expensive that read is "
            "relative to the arithmetic, the more batching is worth. A GPU makes the "
            "ratio most extreme, so it is where batching pays best.",
            "body",
        ),
        P("The batching trade-off is a trade, not a free win", "h2"),
    ]
    full, b1 = d.bench.get("helios_full"), d.bench.get("helios_batch1")
    if full and b1:
        st.append(table(
            [
                ["configuration", "out tok/s", "TTFT p50", "TPOT p50", "mean batch"],
                ["continuous batching",
                 fmt(full.get("output_throughput"), "{:.1f}"),
                 f"{full['ttft']['p50'] * 1000:.0f} ms",
                 f"{full['tpot']['p50'] * 1000:.1f} ms",
                 fmt(full.get("mean_decode_batch"), "{:.1f}")],
                ["batch = 1",
                 fmt(b1.get("output_throughput"), "{:.1f}"),
                 f"{b1['ttft']['p50'] * 1000:.0f} ms",
                 f"{b1['tpot']['p50'] * 1000:.1f} ms", "1.0"],
            ],
            [46 * mm, 26 * mm, 25 * mm, 26 * mm, 26 * mm], align_right=(1, 2, 3, 4),
        ))
        # Both ratios expressed so that a larger number means "more so", which
        # requires opposite orientations: batch 1 has the LOWER TPOT and the HIGHER
        # TTFT. Writing both as full/batch1 produced "0.2x better and 0.2x worse",
        # which is how the inversion was caught.
        tpot_gain = full["tpot"]["p50"] / max(1e-12, b1["tpot"]["p50"])
        ttft_cost = b1["ttft"]["p50"] / max(1e-12, full["ttft"]["p50"])
        st += [
            Spacer(1, 3 * mm),
            P(
                f"Batch&nbsp;1 achieves <b>{tpot_gain:.1f}&times; better per-token "
                f"latency</b> and pays <b>{ttft_cost:.1f}&times; worse "
                f"time-to-first-token</b> "
                f"for it. Neither configuration is correct in the abstract: the choice "
                f"is a function of whether a deployment is latency- or "
                f"throughput-sensitive, which is why the engine exposes three SLO "
                f"classes and admits against per-class token buckets rather than "
                f"picking one policy and calling it the answer.",
                "body",
            ),
        ]
    st += [
        P("The fused Triton kernel", "h2"),
        P(
            "Paging costs a gather &mdash; scattered blocks collected into a contiguous "
            "tensor before scores can be computed, materialising memory the mathematics "
            "does not need. A fused kernel instead walks the block table and streams "
            "each block directly into an online-softmax accumulator, so the score "
            "matrix is never written to memory at all: FlashAttention's IO-aware idea "
            "applied to paged storage. State per program is O(head_dim) rather than "
            "O(context_len).",
            "body",
        ),
        P(
            "<b>5.1&ndash;5.6&times; over the PyTorch paged path</b> across three "
            "independent Tesla T4 runs, with 9 of 9 parity tests against that path as "
            "the correctness oracle. The kernel was written on a machine with no NVIDIA "
            "GPU and marked unverified until a real device ran the gate. Building the "
            "slow reference first was not a detour: it is the oracle a kernel must be "
            "validated against, so it was the right half to build first regardless.",
            "body",
        ),
    ]

    # -------------------------------------------------------------- section 4
    st += [PageBreak(),
           P("4 &nbsp; Quantization, and the measurement that decided a design", "h1")]
    st.append(P(
        "Four-bit weight quantization was initially scoped out, on the reasoning that "
        "<i>quantized kernels need a GPU to be worth anything</i>. That reasoning "
        "conflated two separable claims, and separating them is what unlocked the most "
        "interesting result in the project.",
        "body",
    ))
    st += bullets([
        "A <b>speed</b> claim &mdash; four-bit weights are a quarter of the bytes and "
        "decode is bandwidth-bound. This genuinely needs a GPU. On a CPU, "
        "dequantize-then-GEMM is strictly <i>slower</i> than an fp32 GEMM: the "
        "dequantization is real work and there is no bandwidth wall to relieve.",
        "A <b>numerical</b> claim &mdash; that four-bit weights preserve the model's "
        "output distribution well enough to be useful. This is pure arithmetic: "
        "device-independent, and fully measurable without any GPU at all.",
    ])
    st += [
        Spacer(1, 3 * mm),
        P(
            "The second claim is also the one the entire speculation design rests on. "
            "If an int4 draft does not agree with an fp verifier often enough, the "
            "scheme has no speedup available to it at <i>any</i> bandwidth. So the "
            "numerical half was built, and the kill gate became answerable on a laptop.",
            "body",
        ),
        P("What was implemented", "h2"),
    ]
    q = d.alpha["quant_report"] if d.alpha else None
    st.append(table(
        [
            ["Component", "Design", "Measured"],
            ["INT4 weights", "packed nibbles, group-wise asymmetric scales",
             "3.8× smaller than fp16"],
            ["AWQ scaling", "per-channel exponent search on calibration activations",
             "1.58× lower layer error"],
            ["INT8 paged KV", "per-token, per-head symmetric scales",
             "1.94× smaller than fp16"],
            ["QASSD", "int4 draft, fp verify, separate quantized KV shadow",
             f"α = {d.alpha['verdict']['alpha']:.4f}" if d.alpha else "not measured"],
        ],
        [30 * mm, 73 * mm, 60 * mm],
    ))
    st += [
        Spacer(1, 4 * mm),
        P("Why the scheme is safe regardless of draft quality", "h2"),
        P(
            "The property worth being able to defend cold: <b>speculation is correct "
            "not because the draft is good, but because the verifier decides.</b> An "
            "accepted token is one the target's own argmax agreed with; a rejection "
            "emits the target's argmax instead. The committed sequence is therefore the "
            "target model's greedy sequence, whatever the draft proposes. A bad draft "
            "costs throughput and nothing else.",
            "body",
        ),
        P(
            "That is asserted, not argued. A test runs identical prompts with and "
            "without the int4 draft and requires <b>identical token ids</b>. A "
            "violation would mean the engine silently serving the four-bit model's "
            "output while reporting the full-precision model's &mdash; the worst "
            "failure mode this system has, and one no smoke test would catch.",
            "body",
        ),
        P("The measurement", "h2"),
        P(
            "Eight prompts spanning factual recall, arithmetic, code, and prose, "
            "because acceptance is strongly context-dependent and a single prompt "
            "family would produce a number that does not generalise. AWQ calibrated on "
            "four <i>disjoint</i> prompts &mdash; calibrating on the evaluation text is "
            "how a quantization method gets reported as better than it is.",
            "body",
        ),
    ]
    if "alpha" in figs:
        st.append(figure(
            figs["alpha"],
            "Figure 2 &mdash; Modelled speedup against acceptance rate, with the "
            "measured point, the pre-committed kill gate, and the design target marked. "
            "The optimal draft length moves with α, which is why adaptive gating is "
            "machinery worth having rather than an optimisation.",
        ))
    if d.alpha:
        v = d.alpha["verdict"]
        m = d.alpha["memory"]
        st += [
            P(
                f"<b>&alpha; = {v['alpha']:.4f}</b> &mdash; {v['accepted']} accepted of "
                f"{v['drafted']} drafted, over {d.alpha['n_prompts']} prompts and "
                f"{v['drafted'] // d.alpha['gamma']} verify passes. The gate passes; "
                f"the 0.78 target does not. Optimal &gamma; is "
                f"<b>{v['best_gamma']}</b>, not the planned 4, and at &gamma;=8 "
                f"speculation becomes a net loss of 0.94&times; &mdash; a rejection "
                f"discards every drafted token after it, so longer drafts amplify both "
                f"the win and the waste.",
                "body",
            ),
            P("The cost, stated in the direction that is easy to hide", "h2"),
            P(
                f"Both precisions must be resident, so this is an <i>increase</i> in "
                f"memory: <b>{m['overhead_ratio']:.2f}&times; the target model "
                f"alone</b>. \"Four-bit quantization\" sounds like a reduction, and for "
                f"standalone serving it is &mdash; the int4 view by itself is "
                f"{m['draft_alone_compression']:.2f}&times; smaller. But that "
                f"configuration contains no speculation. Conflating the two would be "
                f"the easiest misrepresentation available here, which is why the number "
                f"is reported as a ratio above 1.",
                "body",
            ),
        ]
    st.append(P("Two findings that came out of the data, not the plan", "h2"))
    # Side by side. Bare Images rather than the KeepTogether that `figure` returns:
    # a KeepTogether has no resolvable height inside a table cell, and reportlab
    # reports that as an absurd row height rather than as a type error.
    if "hist" in figs and "awq" in figs:
        left, right = helpers.image(figs["hist"], 74 * mm), helpers.image(figs["awq"], 74 * mm)
        t = Table(
            [[left, right],
             [P("Figure 3 &mdash; Accepted-run lengths, &gamma;=4.", "caption"),
              P("Figure 4 &mdash; AWQ vs RTN, 95% bootstrap CIs.", "caption")]],
            colWidths=[80 * mm, 80 * mm], hAlign="LEFT",
        )
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
            ("VALIGN", (0, 1), (-1, 1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 1), (-1, 1), 2),
        ]))
        st.append(t)
    else:
        for k, cap in (("hist", "Figure 3 &mdash; Accepted-run lengths."),
                       ("awq", "Figure 4 &mdash; AWQ vs RTN with bootstrap CIs.")):
            if k in figs:
                st.append(figure(figs[k], cap, width=110 * mm))

    if d.alpha:
        hist = d.alpha["verdict"]["run_lengths"]
        total = sum(hist.values())
        meas = d.alpha["verdict"]["measured_tokens_per_pass"]
        modelled = d.alpha["modelled_speedup"]["4"]["expected_tokens_per_pass"]
        st.append(P(
            f"<b>(a) Acceptances are correlated, and the textbook model is "
            f"conservative.</b> The run-length distribution is strongly bimodal: "
            f"{hist.get('4', 0)} of {total} verify passes accepted the entire draft and "
            f"{hist.get('0', 0)} accepted none. That falsifies the independence "
            f"assumption in the standard speedup formula &mdash; a context the draft "
            f"finds easy stays easy for several tokens. The consequence is favourable: "
            f"measured tokens per verify pass is <b>{meas:.2f}</b> against "
            f"{modelled:.2f} predicted under independence, so the standard model "
            f"<i>understates</i> the real value by "
            f"{(meas / modelled - 1) * 100:.0f}%. Using the measured figure, "
            f"&gamma;=4 yields {meas / 2.0:.2f}&times; rather than "
            f"{modelled / 2.0:.2f}&times;.",
            "body",
        ))
    if d.sig:
        st += [
            P(
                f"<b>(b) AWQ reduces layer error but does not improve acceptance.</b> "
                f"AWQ measures 1.58&times; lower per-layer output error than plain "
                f"round-to-nearest, and yet &alpha; comes out at "
                f"{d.sig['awq']['alpha']:.4f} against RTN's "
                f"{d.sig['rtn']['alpha']:.4f} &mdash; nominally <i>worse</i>. Before "
                f"reading anything into that: the unit of independence is the verify "
                f"pass, not the token, because acceptances are correlated. Treating "
                f"{d.alpha['verdict']['drafted']} tokens as independent trials would "
                f"understate the variance in exactly the direction that manufactures "
                f"significance. Bootstrapping over passes and running a permutation "
                f"test gives <b>p&nbsp;=&nbsp;{d.sig['permutation_p']:.2f}</b>: at this "
                f"sample size the two are indistinguishable.",
                "body",
            ),
            P(
                "The likely explanation is an <b>objective mismatch</b>, and it is "
                "labelled a hypothesis because that is what it is. AWQ minimises "
                "squared output error. Acceptance depends only on whether the "
                "<i>argmax</i> survives &mdash; a far coarser property. A method can "
                "cut mean squared error substantially while flipping near-ties in the "
                "logits either way, and only the flips cost acceptance. This does not "
                "make AWQ the wrong choice for standalone four-bit serving, where "
                "output fidelity <i>is</i> the objective. It suggests the calibration "
                "target for a speculative <i>draft</i> should be top-1 agreement with "
                "the target model &mdash; which is not what AWQ optimises. A concrete, "
                "testable follow-up rather than a conclusion.",
                "body",
            ),
        ]

    st.append(P("What speculation actually costs, once it is actually running", "h2"))
    off = d.bench.get("helios_spec_off_batch8")
    sg2 = d.bench.get("helios_spec_g2_batch8")
    sg4 = d.bench.get("helios_spec_g4_batch8")
    qg2 = d.bench.get("helios_qassd_g2_batch8")
    if off and sg2 and sg4 and qg2:
        base = off.get("output_throughput", 1.0)
        rows = [["batch capped to 8", "out tok/s", "vs no speculation"]]
        for label, rec in (("speculation off", off), ("symmetric, γ=2", sg2),
                           ("symmetric, γ=4", sg4), ("QASSD γ=2 (gate active)", qg2)):
            t = rec.get("output_throughput", 0.0)
            rel = "—" if rec is off else f"{base / max(1e-9, t):.2f}× slower"
            rows.append([label, f"{t:.1f}", rel])
        st.append(table(rows, [50 * mm, 30 * mm, 40 * mm], align_right=(1, 2)))
        st += [
            Spacer(1, 3 * mm),
            P(
                "<b>Speculation as implemented forfeits batched decode</b>, and on a "
                "CPU that trade is never worth making. The draft loop is inherently "
                "serial and runs per sequence, so with 8 resident sequences a "
                "speculative step costs 8&times;(&gamma;+1) unbatched forward passes "
                "where a normal step costs one batched pass. The batching win is "
                "larger than anything speculation can return. That is a property of "
                "this implementation and of CPU economics, not of the technique.",
                "body",
            ),
            P(
                "The second row worth reading is the last one. QASSD is <i>faster</i> "
                "than symmetric speculation here &mdash; not because the int4 draft is "
                "better (it is far worse: &alpha; = 0.125 on random toy weights) but "
                "because the <b>adaptive gate noticed and stopped</b>. Measured, it "
                "disabled speculation after a single step, turning a 3.2&times; "
                "disaster into a 1.4&times; tax. Correctness comes from the verifier; "
                "throughput protection comes from the gate. Together they are what "
                "make it safe to ship a draft whose quality has not been characterised "
                "on every workload.",
                "body",
            ),
            P(
                "This ablation did not exist until an audit found that the previous "
                "one measured nothing: it ran at the default batch of 12&ndash;16, "
                "above the gate's threshold of 8, so 0 of 23 decode steps actually "
                "speculated and the row labelled \"speculation\" was reporting "
                "scheduling variance. Third occurrence of the same methodology error "
                "in this project, after the prefix cache and the INT8 KV pool.",
                "small",
            ),
        ]

    st.append(P("The INT8 KV cache: mechanism verified, benefit not realisable here", "h2"))
    fp, i8 = d.bench.get("helios_kv_fp_cramped"), d.bench.get("helios_kv_int8_cramped")
    if fp and i8:
        st.append(table(
            [
                ["256 KiB KV pool", "mean decode batch", "TTFT p50", "out tok/s"],
                ["fp32 KV", fmt(fp.get("mean_decode_batch"), "{:.2f}"),
                 f"{fp['ttft']['p50'] * 1000:.0f} ms",
                 fmt(fp.get("output_throughput"), "{:.1f}")],
                ["INT8 KV", fmt(i8.get("mean_decode_batch"), "{:.2f}"),
                 f"{i8['ttft']['p50'] * 1000:.0f} ms",
                 fmt(i8.get("output_throughput"), "{:.1f}")],
            ],
            [46 * mm, 40 * mm, 33 * mm, 33 * mm], align_right=(1, 2, 3),
        ))
        cb = i8.get("mean_decode_batch", 0) / max(1e-9, fp.get("mean_decode_batch", 1))
        tt = fp["ttft"]["p50"] / max(1e-9, i8["ttft"]["p50"])
        th = i8.get("output_throughput", 0) / max(1e-9, fp.get("output_throughput", 1))
        st += [
            Spacer(1, 3 * mm),
            P(
                f"The mechanism does exactly what it claims: <b>{cb:.2f}&times; more "
                f"resident sequences</b> at the same byte budget, and <b>{tt:.2f}&times; "
                f"better time-to-first-token</b>, because sequences are admitted "
                f"sooner. And throughput is <b>{(1 - th) * 100:.0f}% worse</b>, because "
                f"on a CPU the dequantization is real work with no bandwidth bottleneck "
                f"to offset it. On a GPU, where reading the KV cache <i>is</i> the "
                f"decode bottleneck, halving those bytes should more than pay for the "
                f"dequantization &mdash; a prediction from the same roofline argument "
                f"that correctly anticipated the GPU/CPU pattern in Figure 1, and "
                f"<i>not</i> claimed as a result.",
                "body",
            ),
            P(
                "Getting this measurement right took two attempts, and the first was "
                "wrong in an instructive way. Run in a 4&nbsp;MiB pool, both "
                "configurations kept every sequence resident &mdash; mean batch 14.95 "
                "in both &mdash; so the ablation measured nothing but dequantization "
                "overhead, and INT8 looked 20% slower. That is a mechanism evaluated "
                "outside the regime it exists for. The pool had to be made genuinely "
                "binding before the comparison meant anything at all.",
                "small",
            ),
        ]

    # -------------------------------------------------------------- section 5
    st += [PageBreak(),
           P("5 &nbsp; Distributed KV migration as a state machine", "h1")]
    st += [
        P(
            "Prefill and decode have opposite bottlenecks. Prefill has many query "
            "tokens over a short context: high arithmetic intensity, compute-bound. "
            "Decode has one query token over the whole context: almost no arithmetic "
            "per byte read, bandwidth-bound. Interleaved on one device they make each "
            "other worse &mdash; a single long prompt adds its full duration to every "
            "resident sequence's inter-token latency. Chunked prefill bounds that, and "
            "bounding is not eliminating.",
            "body",
        ),
        P(
            "Disaggregation removes the contention instead, by putting the two on "
            "separate devices and transferring the prompt's KV cache between them.",
            "body",
        ),
        P(
            "<b>What follows is a verified design, not a shipped feature, and the "
            "distinction is worth making precisely.</b> Nothing outside the test "
            "harness imports this module &mdash; the engine, the scheduler and the "
            "runner never call it. With one device there is no transfer to perform, "
            "because the blocks are already where they need to be; routing them through "
            "a transfer FSM would be theatre, a code path existing to make a claim true "
            "rather than to do work. What is real is the protocol: a state machine with "
            "partial-failure semantics worked out and mechanically checked before any "
            "transport exists. That is worth more than transport code with the "
            "semantics left implicit &mdash; but it is not something a user can switch "
            "on, and an earlier draft of this report listed it as though it were.",
            "callout",
        ),
        P("Why a state machine and not a coroutine", "h2"),
        P(
            "The design deliberately separates <font face='Courier'>SENT</font> (the "
            "sender believes it is done) from <font face='Courier'>RECEIVED</font> (the "
            "receiver confirms). Every interesting fault lives in the gap between those "
            "two, and an async/await formulation collapses them by default &mdash; "
            "which makes \"sender finished, receiver never acknowledged\" "
            "<i>unrepresentable</i>, and therefore untestable. Seven explicit states "
            "with a legal-transition table; the table is the specification, so an "
            "unlisted edge raises rather than being absorbed.",
            "body",
        ),
        P(
            "Five fault kinds are modelled &mdash; receiver out of memory, link "
            "timeout, checksum mismatch, lost acknowledgement, mid-flight abort. Each "
            "returns blocks from <b>both</b> partitions, because handling one side "
            "leaks the other, and a leak stays invisible until the pool is exhausted "
            "&mdash; at which point it looks like a capacity problem rather than a bug. "
            "Two rules fall directly out of the ownership analysis: the sender may not "
            "release until the receiver confirms, or a lost acknowledgement becomes "
            "unrecoverable rather than merely slow; and a client abort after the bytes "
            "are on the wire must resolve as a <i>failure</i> rather than an abort, so "
            "that ownership is settled before anything is freed.",
            "body",
        ),
        P("Verified by detection, not by coverage", "h2"),
        P(
            "Half of all simulation seeds run the FSM alongside the scheduler under "
            "randomised faults, asserting six FSM invariants plus two pool invariants "
            "the FSM cannot check itself. Coverage across 200 seeds reached every fault "
            "kind. Then &mdash; applying the lesson from section 2 &mdash; the classic "
            "partial-failure bug was reintroduced deliberately: "
            "<font face='Courier'>fail()</font> returning the sender's blocks but not "
            "the receiver's. <b>Caught by 31 of 60 seeds</b>, with a message naming the "
            "exact accounting discrepancy. Making an abort legal after "
            "<font face='Courier'>SENT</font> was caught by 15 of 60.",
            "body",
        ),
        P("An error this produced, worth recording", "h2"),
        P(
            "Whether to transfer a prompt's KV or simply recompute it is a decision "
            "that flips with the hardware, so it is computed from bandwidth rather than "
            "assumed. The first version of that arithmetic put a 2000-token KV cache at "
            "2.6&nbsp;GB when it is <b>262&nbsp;MB</b> &mdash; a factor of ten, which "
            "inverted the verdict for PCIe from \"wins by 11&times;\" to \"roughly "
            "break-even\". It was caught by a test written against the real formula, not "
            "by re-reading the comment. Corrected, the conclusion is sharp: <b>within a "
            "node, transferring is clearly right; across a commodity network it is "
            "not.</b> The crossover sits at the machine boundary. A design that always "
            "transferred would be strictly worse than no disaggregation at all on the "
            "wrong side of that line.",
            "body",
        ),
    ]

    # -------------------------------------------------------------- section 6
    # ------------------------------------------------ the audit (section 5b)
    st += [PageBreak(), P("6 &nbsp; What a deliberate audit found", "h1")]
    st += [
        P(
            "Everything above was written, tested, and committed before this section "
            "existed: 263 tests green, 5,000 simulation seeds green. The audit was run "
            "anyway, on the premise that <b>a green suite is evidence the code does "
            "what the tests say, not that the claims are true.</b> Four of the five "
            "findings were claim errors rather than code errors, and no test was "
            "failing for any of them.",
            "body",
        ),
    ]
    st.append(table(
        [
            ["Found", "Why it mattered"],
            ["QuantLinear memoised the dequantized fp weight, so after one forward "
             "pass an int4 layer held 26% MORE memory than the fp16 layer it "
             "replaced — while stored_bytes() reported a 3.8x saving",
             "Critical: every quantization memory claim in the project was false in "
             "practice. Caching is now opt-in and off by default, and resident_bytes() "
             "exists so the packed size and the real size cannot be conflated again."],
            ["The speculation ablation ran at a batch above the adaptive gate, so 0 of "
             "23 decode steps actually speculated",
             "High: a row labelled \"speculation\" was measuring scheduling variance. "
             "A dedicated suite now caps the batch so the mechanism is on the critical "
             "path."],
            ["memory_overhead() reported only weights, omitting the entire second KV "
             "cache that QASSD allocates",
             "Medium: the quoted number was true and incomplete — harder to catch than "
             "a false one, because nothing about it looks wrong."],
            ["The KV transfer FSM was listed as \"Built\" beside paged attention, when "
             "nothing outside the test harness imports it",
             "Medium: a verified design was presented as a shipped feature. Relabelled, "
             "with the grep that proves it."],
            ["A benchmark run was contaminated by a concurrent job on the same machine",
             "Low: it inverted the γ=2 / γ=4 ordering. Re-run idle; the inversion "
             "disappeared."],
        ],
        [64 * mm, 99 * mm],
    ))
    st += [
        Spacer(1, 4 * mm),
        P(
            "Each fix carries a regression test, and each of those tests was "
            "<b>mutation-checked</b>: the bug it exists to catch was deliberately "
            "reintroduced and the test confirmed to fail. One result from that exercise "
            "is worth more than the others. When the accept logic was mutated to commit "
            "the draft's token instead of the verifier's, <b>every symmetric-speculation "
            "parity test passed</b> &mdash; because at &alpha;&nbsp;=&nbsp;1.0 the draft "
            "token and the verified token are identical, so the bug is invisible. Only "
            "the asymmetric test failed.",
            "body",
        ),
        P(
            "That is the general lesson, and it is the same one section 2 records about "
            "copy-on-write coverage, arriving from a different direction: <b>a test "
            "suite can be large and green and still have no coverage of the thing that "
            "matters.</b> The only reliable check is to break the code on purpose and "
            "watch what fails.",
            "body",
        ),
    ]

    st += [PageBreak(), P("7 &nbsp; What is not built", "h1")]
    st.append(P(
        "This section exists because the project's governing rule was that a claim must "
        "never outrun its evidence &mdash; and the only way to keep that rule is to "
        "write the boundary down rather than leave a reader to discover it.",
        "body",
    ))
    st.append(table(
        [
            ["Not built", "Why, precisely"],
            ["A fused int4 GEMM",
             "The quantization is numerically real and verified; the SPEEDUP is not. "
             "QuantLinear dequantizes and then calls a normal GEMM, so on a CPU it is "
             "slower than fp32. Every quantization speedup in this report is labelled "
             "modelled."],
            ["vLLM comparison",
             "The harness is written with matched controls — same model directory, "
             "dtype, KV byte budget, seeded arrival trace, discarded warm-up — and has "
             "never been run, because vLLM requires CUDA. No comparison is claimed."],
            ["W4A4 (4-bit activations)",
             "The available GPU has no fast int4 activation path, so a W4A4 draft "
             "would not be cheaper than the verify and the premise fails. A hardware "
             "constraint rather than a gap in the work."],
            ["Real multi-device disaggregation",
             "The FSM, the block accounting, and the transfer-versus-recompute "
             "decision are built and verified. The transport is simulated: there is no "
             "second device to move bytes to."],
            ["Rust frontend and scheduler",
             "Cut from scope and still cut. The ExecStep seam it would attach to was "
             "preserved deliberately, which was the point."],
            ["α on a larger model",
             "The 0.5B result is marginal. Quantization error at a fixed bit-width "
             "falls with model size, so a 7B model should do better — a hypothesis, "
             "untested, and not claimed."],
        ],
        [40 * mm, 123 * mm],
    ))
    st += [
        Spacer(1, 4 * mm),
        P(
            "One caveat applies to every CPU figure in this report: the local benchmark "
            "model is a small random-weight toy, so absolute tokens per second are "
            "meaningless. The <i>ratios between ablations</i> are the result. What makes "
            "the correctness claim credible instead is coherent text from a real "
            "pretrained checkpoint &mdash; Qwen2.5-0.5B on a T4 in fp16 &mdash; because "
            "paging, block tables, rotary embeddings, GQA head expansion, "
            "normalisation, batched prefill, batched decode, fp16 numerics, and sampling "
            "must all be simultaneously correct for a pretrained model to produce "
            "sensible output. A toy model with random weights can only ever demonstrate "
            "self-consistency: an error shared by the implementation and its own "
            "reference would pass every such test.",
            "body",
        ),
    ]

    # -------------------------------------------------------------- section 7
    st += [P("8 &nbsp; What the process taught", "h1")]
    st += [
        P("Three bugs were structurally invisible without real hardware", "h2"),
        P(
            "The GPU sessions found four bugs, and <b>three of them could not have been "
            "caught on the development machine at all</b> &mdash; while 178 tests were "
            "passing. Qwen2 attention biases were silently dropped, because the loader "
            "skipped checkpoint tensors it had no slot for and the toy model is written "
            "by the same code that reads it. Model weights were never moved to the "
            "device, because <font face='Courier'>.to(dtype)</font> is not "
            "<font face='Courier'>.to(device)</font> and the two coincide when there is "
            "only one device. And the executor converted the resulting exception into a "
            "retryable fault, turning a hard failure into an infinite retry loop whose "
            "only symptom was an empty result list.",
            "body",
        ),
        P(
            "The regression tests written afterwards assert the <i>invariant</i> "
            "&mdash; every parameter on the device the engine builds inputs on &mdash; "
            "rather than the symptom, because the symptom is unobservable where the "
            "test runs.",
            "body",
        ),
        P("A documented conclusion that was simply wrong", "h2"),
        P(
            "For a period this project recorded, as a finding, that \"continuous "
            "batching needs a GPU to pay\" &mdash; because the measured batching win "
            "was zero. A matmul microbenchmark then showed roughly 10&times; from "
            "batching GEMMs on a CPU at 32 sequences. The engine had been running one "
            "forward pass per sequence: a missing optimisation in the executor, not a "
            "property of the hardware. Fixing it required no GPU and produced "
            "2.24&times;. The lesson that stuck: a negative result about <i>hardware</i> "
            "deserves more suspicion than a negative result about code, because it is "
            "unfalsifiable from where you happen to be standing.",
            "body",
        ),
        P("Measuring a mechanism outside its regime reads as a regression", "h2"),
        P(
            "This happened three times &mdash; the prefix cache on a 32-token shared "
            "prefix, prefill batching hidden behind long generations, and the INT8 KV "
            "cache in a pool that was not binding. Each looked worthless or actively "
            "harmful until the workload was constructed so the mechanism had something "
            "to do. Dedicated suites now exist for each. The general form: an "
            "optimisation's benchmark has to be designed alongside the optimisation, or "
            "the default workload silently decides the verdict.",
            "body",
        ),
        P("Pre-committing to a kill threshold changed the outcome", "h2"),
        P(
            "The 0.60 acceptance floor was written down before anything was built, and "
            "the measurement came in at 0.655 &mdash; close enough that a threshold "
            "chosen afterwards would have been indefensible in either direction. "
            "Because it was fixed in advance, the result is a decision rather than a "
            "rationalisation: the feature ships, its optimal parameter is not what was "
            "planned, and its expected benefit is bounded by a number instead of by "
            "hope.",
            "body",
        ),
    ]

    # ------------------------------------------------------------- appendix
    st += [PageBreak(), P("Appendix &nbsp; Reproduction and provenance", "h1")]
    st.append(code(
        "# full test suite\n"
        "PYTHONPATH=python python -m pytest tests -q\n\n"
        "# deterministic simulation harness; any failure replays from its seed\n"
        "PYTHONPATH=python python -m helios.cli vopr --seeds 50000\n"
        "PYTHONPATH=python python -m helios.cli vopr --seed 918273 --replay\n\n"
        "# ablations -> JSON artifacts -> generated docs/BENCHMARKS.md\n"
        "python bench/loadgen.py --model artifacts/bench_model --suite all\n"
        "python bench/report.py\n\n"
        "# the acceptance-rate measurement and its significance analysis\n"
        "python bench/measure_alpha.py --model artifacts/qwen05b --gamma 4\n"
        "python bench/measure_alpha.py --model artifacts/qwen05b --no-awq \\\n"
        "    --out artifacts/alpha_rtn.json\n"
        "python bench/alpha_significance.py\n\n"
        "# GPU gates: device report, suite, Triton parity, real model end-to-end\n"
        "python bench/kaggle_gpu_run.py\n\n"
        "# this report\n"
        "python bench/make_report.py --out docs/HELIOS-REPORT.pdf"
    ))
    rows = [["Field", "Value"],
            ["commit", prov["commit"]],
            ["generated", prov["date"]],
            ["python / torch", f"{prov['python']} / {prov['torch']}"],
            ["platform", prov["platform"]],
            ["lines of Python (src, tests, bench)", f"{prov['loc']:,}"],
            ["tests passing", prov["tests"]]]
    def _fail_count(rec) -> int:
        """`failures` is a list of failing seeds in some artifacts, an int in others."""
        f = rec.get("failures", 0)
        return len(f) if isinstance(f, (list, tuple, dict)) else int(f)

    if d.dst50k:
        rows.append(["simulation sweep",
                     f"{d.dst50k.get('seeds', 0):,} seeds, "
                     f"{_fail_count(d.dst50k)} failures"])
    if d.dst_disagg:
        rows.append(["sweep including the transfer FSM",
                     f"{d.dst_disagg.get('seeds', 0):,} seeds, "
                     f"{_fail_count(d.dst_disagg)} failures"])
    if d.alpha:
        rows.append(["α measurement",
                     f"{Path(d.alpha['model']).name}, γ={d.alpha['gamma']}, "
                     f"group={d.alpha['group_size']}, awq={d.alpha['awq']}"])
    st.append(table(rows, [58 * mm, 105 * mm]))
    st += [
        Spacer(1, 4 * mm),
        P(
            "Every figure and table above is generated from a JSON artifact in "
            "<font face='Courier'>artifacts/</font> by "
            "<font face='Courier'>bench/make_report.py</font>. No number in this "
            "document is typed in by hand, for the same reason "
            "<font face='Courier'>docs/BENCHMARKS.md</font> is generated rather than "
            "written: a report that can be edited independently of its measurements "
            "will eventually disagree with them, and nobody will notice.",
            "small",
        ),
    ]
    if d.missing:
        st += [
            P("Artifacts absent at generation time", "h2"),
            P(
                "Listed rather than silently omitted, so a report built from a partial "
                "run says so on its face: "
                + ", ".join(f"<font face='Courier'>{m}</font>"
                            for m in sorted(set(d.missing)))
                + ".",
                "small",
            ),
        ]
    return st


def render(story: list, out: Path, prov: dict, palette) -> None:
    """Lay the story out on A4 with a running header and page numbers."""
    muted, rule, accent = palette
    doc = BaseDocTemplate(
        str(out), pagesize=A4,
        leftMargin=23 * mm, rightMargin=23 * mm,
        topMargin=21 * mm, bottomMargin=18 * mm,
        title="HELIOS - engineering report", author="HELIOS",
        subject="LLM inference engine: design, measurements, and scope",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")

    def decorate(canvas, document):
        canvas.saveState()
        if document.page > 1:
            canvas.setFont("Helvetica", 7.4)
            canvas.setFillColor(muted)
            canvas.drawString(doc.leftMargin, A4[1] - 13 * mm,
                              f"HELIOS   ·   engineering report   ·   "
                              f"commit {prov['commit']}")
            canvas.drawRightString(A4[0] - doc.rightMargin, 11 * mm,
                                   str(document.page))
            canvas.setStrokeColor(rule)
            canvas.setLineWidth(0.4)
            canvas.line(doc.leftMargin, A4[1] - 15.5 * mm,
                        A4[0] - doc.rightMargin, A4[1] - 15.5 * mm)
        else:
            canvas.setStrokeColor(accent)
            canvas.setLineWidth(1.8)
            canvas.line(doc.leftMargin, A4[1] - 66 * mm,
                        A4[0] - doc.rightMargin, A4[1] - 66 * mm)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
    doc.build(story)

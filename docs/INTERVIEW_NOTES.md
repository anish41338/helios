# Interview defense

Spec section 15's question bank, answered against **this** implementation.
Where the answer is "not built", it says so — spec §15 explicitly notes that
"I built the harness, it hasn't caught a live bug yet" is credible while
inventing one is not. The same rule applies to everything else here.

---

## Paging

**Why does paging help — what exactly is the wasted memory?**
Internal fragmentation. A contiguous per-sequence KV buffer must be sized to
`max_seq_len` up front, so a 100-token request occupying a 4096-token slot wastes
97.5% of it. Paging allocates fixed-size blocks on demand and maps logical block
index → physical block id per sequence, so waste is bounded by at most one
partially-filled block per sequence (< `BLOCK_SIZE` tokens).

**What's the cost?**
Indirection. Attention must gather scattered blocks before computing scores
(`paged_attn.py::gather`), which costs a pass over the block table and, in this
CPU build, materialising the gathered tensor. A fused kernel would stream blocks
directly into the accumulator instead. There is also per-step bookkeeping: ref
counts, block tables, CoW checks.

**How do you pick BLOCK_SIZE; tradeoff at 4 vs 16 vs 128?**
Small blocks cut internal fragmentation (waste ≤ BLOCK_SIZE−1 tokens per
sequence) but lengthen block tables, increase gather overhead, and reduce
contiguity for the memory system. Large blocks waste more per sequence but give
longer contiguous runs and shorter tables. 16 is the common default; the DST
harness randomises over {8, 16, 32} so no logic silently depends on one value.

**Internal vs external fragmentation here?**
Internal: unused tokens inside an allocated block — bounded, ≤ BLOCK_SIZE−1 per
sequence. External: none, by construction — all blocks are the same size, so any
free block satisfies any request. That is the main structural win of fixed-size
paging.

---

## Batching

**Why is iteration-level better than static?**
Static batching pads to the longest member and holds finished slots hostage until
the whole batch completes. Iteration-level batching makes a scheduling decision
every forward pass: a sequence that emits EOS leaves immediately and a waiting
sequence takes its slot in the *same* iteration. `_reap_finished` runs before
`_admit` precisely so the freed blocks are available that step.

**Honest caveat:** this build shows **no throughput win** from it, because the
executor runs one sequence per forward pass (~5–7 ms per resident sequence,
flat). Continuous batching's premise is that the batch dimension is nearly free —
one fused kernel, N sequences — and a serialized executor breaks that premise. I
tested the mechanism's precondition rather than assuming: with a 50× spread in
output lengths and Poisson arrivals, static batching still won (0.91×). So it is
a property of the build, not the workload. See `SCOPE.md`.

**Admission policy, and how starvation is avoided?**
Sort by `(slo_class, arrival_step, seq_id)`, then per-class token buckets refilled
by *step count* (never wall clock, for determinism). Requests that don't fit are
skipped so they can't head-of-line block smaller ones — but after 32 consecutive
skips the queue head gets an exclusive reservation, so it can't be starved either.
Those are two opposite livelocks the DST harness found separately (BUG-006/014).

**Why prefer recompute over swap, and when does that flip?**
Recompute frees memory instantly and costs only compute; swap costs PCIe
bandwidth both ways, and on a consumer box that bandwidth competes with
everything else. It flips when recompute is more expensive than the transfer:
roughly `prompt_len / prefill_throughput` vs `(blocks × block_bytes) / pcie_bw ×
2`. The implemented rule is the spec's: swap only if
`generated_len > 4 × prompt_len`.

**What is the token budget per step and why?**
`max_num_batched_tokens` caps tokens per forward pass. Without it, one long
prefill monopolises a step and spikes TPOT for every in-flight decode. The step
builder also reserves one token of budget per decode before packing prefills, so
prefill can never starve decode.

---

## Attention kernel

**Why is decode bandwidth-bound and prefill compute-bound — derive it.**
Arithmetic intensity = FLOPs / bytes moved. Decode: one query token attends over
`L` cached tokens, so it reads O(L · d) bytes of KV and does O(L · d)
multiply-accumulates — intensity ≈ O(1), with no reuse. It also reads all model
weights to produce a single token. Prefill: `T` query tokens each attend over up
to `T` keys, so FLOPs grow as O(T² · d) while the weight read is amortised across
all T tokens — intensity grows with T. Hence prefill saturates FLOPs and decode
saturates bandwidth, and interleaving them on one device makes them contend. That
contention is the motivation for the spec's disaggregation (not built).

**Roofline for a decode step?**
For an 8B model at W4A16, weights ≈ 5.7 GB read per step. On a 4090 (~1008 GB/s)
that floors a single decode step at ≈ 5.7 ms, i.e. ~175 tok/s ceiling at batch 1
regardless of FLOPs — which is why batching matters for throughput and why
speculation (multiple tokens per weight read) helps latency. On this CPU build
the measured ~5–7 ms/step is bandwidth- and Python-overhead-bound on a toy model,
so it is not comparable.

**Why does online softmax avoid materialising the score matrix?**
Softmax is normally two passes (find max, then exponentiate/sum). The streaming
form keeps a running max `m` and running sum `l`, and on each new tile rescales
the accumulator by `exp(m_old − m_new)`. That makes it a single pass with O(d)
state instead of O(T) scores, so the T×T matrix is never written to memory. This
build's SDPA fallback relies on PyTorch for that; the Triton kernel that would
implement it explicitly is not built.

---

## Speculation

**Derive the speedup.**
`(1 + α + α² + … + α^γ) / (1 + γ · c_draft/c_verify)`. Numerator is the expected
tokens committed per iteration (accept the i-th only if the first i all
accepted); denominator is the cost: one verify pass plus γ drafts at relative
cost `c_draft/c_verify`.

**Why is the output distribution preserved?**
Because the *verifier* decides, not the draft. Rejection sampling accepts a draft
token with probability `min(1, p_target/p_draft)` and otherwise samples from the
residual `max(0, p_target − p_draft)` normalised — which is constructed so the
resulting distribution equals the target's exactly. Under greedy decoding this
degenerates to "accept while the draft matches the target's argmax", which is what
this build implements and what the bit-parity tests assert.

**Why does it stop helping at large batch?**
At batch 1 the machine is bandwidth-starved and the draft's extra FLOPs are
nearly free. At large batch, verify is already compute-bound, so wasted draft
compute directly displaces useful work. Hence `spec_max_batch_size = 8`.

**What breaks if you forget to roll back the quantized KV shadow?**
The draft path reads stale KV for the rejected positions, so its proposals get
worse, α collapses, and speedup silently vanishes — while output stays *correct*
because the verifier still decides. It presents as a performance regression, not a
bug, which is why it's expensive to find. Spec §7.4 flags it; in this build there
is no quantized shadow to desynchronise (see `QASSD.md`).

---

## Quantization

**W4A16 vs W4A4, and why is the latter only safe for a draft?**
Both store 4-bit weights; they differ in activation precision. W4A16 keeps fp16
activations, so only weight quantization error enters. W4A4 also quantizes
activations to 4 bits — 16 representable levels for values with outliers — which
compounds error through every layer. That is unacceptable for the committed
distribution but fine for a *proposal*, because a bad proposal is merely rejected.
Correctness comes from the verifier.

**What does group size 128 buy you?**
One scale per 128 weights instead of per tensor: much better dynamic-range
fitting where weight magnitudes vary across a row, at an overhead of one fp16
scale per 128 values (~0.8%). Smaller groups are more accurate and more
metadata-heavy.

**Where does AWQ's activation-awareness come from?**
From the observation that a small fraction of weight *channels* matter far more
because they multiply large-magnitude activations. AWQ uses activation statistics
from calibration data to pick per-channel scalings that protect those salient
channels before quantizing — it is not weight magnitude alone.

**Not implemented here.** No AWQ/GPTQ loading, no int4 kernels. Answering these
is reading, not experience, and it should be presented that way.

---

## Disaggregation

**Not built** — needs ≥2 devices; this machine has none. The design questions
(why split phases: the arithmetic-intensity argument above; how to hide transfer
cost: start copying layer *l*'s KV as soon as layer *l* finishes prefill, so PCIe
overlaps remaining compute; why pipeline rather than tensor parallel without
NVLink: tensor parallel needs an all-reduce every layer and degrades to the
interconnect, while pipeline parallel only passes activations at stage
boundaries) are spec §6.4, not results. The DST harness models the transfer FSM's
*fault kinds* but the transport itself is untested.

---

## DST

**What makes a system simulable?**
All nondeterminism must be injectable. No wall clock (time is `step_t`), no
unseeded RNG, no threads, no I/O, no iteration over hash-ordered containers, and
every tie-break a total order. Then `Simulation(seed)` is a complete, replayable
experiment.

**What did you give up in the scheduler design to get determinism?**
Convenience, mostly: no logging or metrics inside the core (they moved to
`engine.py`), no `HashMap` iteration, sorted snapshots instead of ad-hoc dict
walks, token buckets refilled by step count rather than elapsed time, and no
threading inside the scheduler — which forced the HTTP layer into a
single-lock cooperative pump. The cost is real but small; the payoff is that a
four-thousand-step livelock reproduces exactly from an integer.

**Name a real bug it found.**
Fifteen, all documented in `DST.md` with root causes. The one I'd lead with:
`blocks_needed_to_append` returned 0 for a **shared, partially-filled tail
block** — correct that it had spare room, wrong because a shared block must be
copy-on-written before any write, so the allocation silently overshot the memory
watermark and broke invariant I7. Reproduced in eight lines. Spec §5.3 predicted
exactly that bug class, which is a nice illustration that knowing about a bug
class doesn't stop you writing it.

The more interesting statistic: **13 of the 15 were liveness bugs**, not
corruption. The engine kept running, kept looking busy, and completed nothing.
Ordinary integration tests would have passed. That's the argument for the liveness
and no-leak assertions being as important as the invariants.

Also worth volunteering: several of my *fixes* were wrong, and the harness said so
in seconds — the BUG-004 fix dropped the pass rate from 193/200 to 157/200 by
exposing a pin leak underneath it.

---

## Honest weaknesses to volunteer

- **Where vLLM wins:** kernel maturity, CUDA graphs, FlashInfer/FlashAttention
  integration, years of tuning, real batched paged attention, broad quantization
  support, multi-GPU. This build has none of that and does not compare itself to
  vLLM anywhere.
- **What I didn't build:** Rust, CUDA/Triton kernels, quantization,
  prefill/decode disaggregation, the quantization asymmetry that is QASSD's whole
  claim to novelty, 70B pipeline parallel, a vLLM baseline. All enumerated in
  `SCOPE.md`.
- **What my numbers do not prove:** nothing about GPU performance, nothing about
  quantized inference, nothing about α for QASSD, and — importantly — **not** that
  continuous batching is faster, because in this build it isn't and I say so.
- **What they do support:** the mechanisms are implemented and verified correct
  (bit-exact parity on chunked prefill, speculation, and paged vs dense
  attention), and the scheduler survives 50,000 adversarial simulated runs with
  seven invariants plus liveness and leak checks asserted every step.

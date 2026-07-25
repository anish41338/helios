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

**Measured:** 1.13× throughput over static batching on a mixed workload, and
2.24× from batching the executor's decode step (`baseline_unbatched_executor`
ablates only that). Mean resident decode batch 11.6.

**The story worth telling here**, because it is the most useful thing I learned:
the first version of this engine ran one sequence per forward pass, so continuous
batching showed **no** win at all, and static batching beat it. I documented that
as a GPU-dependent limitation — the premise being that batching only pays when a
fused kernel makes the batch dimension nearly free. That conclusion was wrong. A
bare matmul microbenchmark showed ~10× from batching the GEMMs on this CPU at
N=32, so the win was an implementation shortcut away the whole time. Batching the
executor turned a documented negative result into 2.24×.

Two things I took from it: I had rationalised a missing optimisation as a hardware
constraint, and my benchmark could not tell the difference. The harness now
reports `mean_decode_batch` for exactly that reason — a throughput number from a
run averaging one resident sequence measures nothing, and an earlier misleading
comparison came from precisely that (sparse Poisson arrivals against static
batching fed everything at t=0).

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
regardless of FLOPs.

The consequence is the important part: that floor is per *step*, not per
sequence, because the weight read is shared by everything in the batch. So batch
1 wastes almost all of the machine — you pay the full weight read for one token.
That is exactly why batching is a throughput mechanism and why speculation (more
tokens per weight read) is a latency mechanism; they attack the same ratio from
opposite sides.

This build's absolute numbers are not comparable — CPU, toy model, fp32 — but the
*shape* held: batching the decode step gave 2.24×, because it amortises one
weight read across 11.6 sequences on average instead of one.

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
bug, which is why it's expensive to find. Spec §7.4 flags it, and this build has
exactly that hazard: the draft writes keys for every token it proposes, and past
the first rejection those describe a continuation that was thrown away. Paged
storage has no notion of truncation, so nothing catches it. `_run_speculative`
re-syncs the draft's KV over the committed span for precisely this reason.

**So what is α, actually?**
**0.6548**, measured on Qwen2.5-0.5B with an int4 AWQ draft against fp32 verify
(336 drafted, 220 accepted, 84 verify passes). Have the follow-ups ready, because
this number invites them:

- *The gate passed but the target didn't.* Pre-committed kill threshold was 0.60;
  spec §7.3's target was 0.78. So the feature ships and is worth less than planned.
- *Optimal γ is 2, not 4.* And at γ=8 the modelled speedup is **0.94× — a net
  loss**, because a rejection discards every drafted token after it.
- *It costs +51% weight memory*, not −75%. Both precisions must be resident.
  "4-bit quantization" sounding like a saving is the easiest misreading here.
- *Why is measuring this on a CPU legitimate?* α is the probability that two weight
  matrices produce the same argmax on the same context. It is arithmetic, not
  throughput, so it transfers to a GPU unchanged. The *speedup* does not — there is
  no int4 GEMM here, so every speedup figure is labelled modelled.

**The histogram falsified the model's own assumption.**
Accepted-run lengths are strongly bimodal: 42 of 84 passes accepted all four
tokens, 15 accepted none. So acceptances are positively *correlated* — a context
the draft finds easy stays easy — which violates the independence assumption in the
speedup formula above. The error is in the safe direction: measured tokens per pass
was **3.12** against 2.55 predicted, so the textbook model *understates* the value
by 22%. Being able to say which way a model's assumption fails is more useful than
quoting the model.

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
channels before quantizing — it is not weight magnitude alone. The identity that
makes it free in exact arithmetic: `(x/s) @ (W·s)ᵀ = x @ Wᵀ` for any positive `s`,
so scaling only moves *where* the quantization error lands.

**Implemented, and here is the null result.**
AWQ measures **1.58× lower** mean per-layer output error than round-to-nearest
across 168 layers. And α comes out at 0.6548 with AWQ against **0.6860** with
plain RTN — nominally *worse*. Bootstrapping over verify passes (the unit of
independence, since acceptances are correlated) and running a permutation test
gives **p = 0.64**: indistinguishable at this sample size.

The interesting part is the explanation, offered as a hypothesis: **objective
mismatch.** AWQ minimises `‖(W_q − W)x‖²` — output fidelity. Acceptance depends
only on whether the **argmax** survives, which is far coarser. A method can cut
MSE substantially while flipping near-ties in the logits either way, and only the
flips cost acceptance. That does not make AWQ wrong for standalone W4A16 serving,
where fidelity *is* the objective. It says the calibration target for a
speculative *draft* should be top-1 agreement with the target model — which is a
concrete follow-up, not a conclusion.

**What about the KV cache?**
INT8 with per-token, per-head symmetric scales: **1.94× smaller** than fp16.
Per-token rather than per-tensor because attention is a dot product over head_dim,
so one outlier token would otherwise set the scale for every token sharing it.
Symmetric rather than asymmetric because a zero point would have to be carried
through the score matmul as a correction term instead of folding into one multiply.

Measured in a pool small enough to bind: **2.31× more resident sequences, 1.62×
better TTFT — and 27% worse throughput**, because on a CPU the dequantization is
real work with no bandwidth wall to relieve. Volunteer that number; the GPU
prediction (that halving the KV read more than pays for it when bandwidth *is* the
bottleneck) is a prediction, not a result.

**Still not implemented:** a fused int4 GEMM. `QuantLinear` dequantizes and calls a
normal GEMM, so on CPU it is *slower* than fp32. Every quantization speedup in the
repo is labelled modelled. Say this before being asked.

---

## Disaggregation

**The FSM is built and verified; the transport is simulated** — there is one
device. Be precise about which half that is, because the half that is built is the
half where the bugs are.

**Why a state machine rather than async/await?**
Because the design has to distinguish `SENT` (the sender believes it is done) from
`RECEIVED` (the receiver confirms). Every interesting fault lives in that gap, and
an await collapses it — making "sender finished, receiver never acknowledged"
*unrepresentable* and therefore untestable. Seven states, an explicit legal-
transition table, and an unlisted edge raises.

**Name the failure mode that motivates it.**
A transfer that dies between `SENT` and `RECEIVED` has blocks reserved on the
receiver that the sender does not know about, and blocks pinned on the sender the
receiver cannot see. Handle one side and you leak the other — invisible until the
pool is exhausted, at which point it looks like a capacity problem. Two rules fall
out: the sender may not release until the receiver confirms (or a lost ack becomes
unrecoverable rather than merely slow), and a client abort after the bytes are on
the wire must resolve as a *failure* rather than an abort.

**Is transferring even the right call?**
Not always, and this is the answer worth having. A 2000-token prompt on a 32-layer
model with 8 KV heads at head_dim 128 in fp16 is **262 MB**. Re-prefilling it at
8000 tok/s costs 250 ms. So transferring wins by ~11× over PCIe 3.0 and *loses*
over 10 GbE — the crossover is at the machine boundary, not inside it. A design
that always transferred would be strictly worse than no disaggregation at all on
the wrong side of that line, which is why it is computed from bandwidth.

**How would you hide the transfer cost?** Start copying layer *l*'s KV as soon as
layer *l* finishes prefill, so the interconnect overlaps the remaining compute.
And why pipeline rather than tensor parallel without NVLink: tensor parallel needs
an all-reduce every layer and degrades to the interconnect, while pipeline parallel
passes activations only at stage boundaries.

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
  integration, years of tuning, broad quantization support, multi-GPU. This build
  has one Triton kernel and a Python hot path, and it does not compare itself to
  vLLM anywhere. The baseline harness is *written* with matched controls
  (`bench/vllm_baseline.py`) and has **never been run** — it needs CUDA.
- **No fused int4 GEMM.** The quantization is numerically real and verified; the
  speedup is not. `QuantLinear` dequantizes then calls a normal GEMM, so on CPU it
  is *slower* than fp32. Every quantization speedup figure in the repo is labelled
  **modelled**. This is the single largest gap and it should be stated first.
- **Disaggregation transport is simulated.** The FSM, the fault taxonomy, the block
  accounting, and the transfer-versus-recompute decision are built and verified
  across thousands of seeds. Moving actual bytes between actual devices is not.
- **What I didn't build at all:** Rust, W4A4 (the T4 has no fast int4 activation
  path, so the premise fails on available hardware), 70B pipeline parallel. All
  enumerated in `SCOPE.md`.
- **What the CPU numbers do not prove:** nothing about absolute GPU throughput. The
  model is a random-weight toy, so tokens/second is meaningless; the *ratios between
  ablations* are the result. What makes the correctness claim credible instead is
  coherent text from Qwen2.5-0.5B on a T4 — a toy model can only ever demonstrate
  self-consistency, since an error shared by the code and its own reference passes
  every such test.
- **α is marginal, and on a small model.** 0.6548 clears the 0.60 gate and misses
  the 0.78 target. The hypothesis that it improves with model scale (quantization
  error at fixed bit-width falls as models get more redundant) is **untested and not
  claimed**.
- **A conclusion I got wrong and corrected:** I documented "continuous batching
  needs a GPU to pay" as a finding. It was a missing optimisation in my own
  executor — batching the GEMMs gave 2.24× with no hardware change. Worth
  volunteering, because the interesting question is not whether I hit the target but
  whether my measurements could tell me when I was wrong, and initially they could
  not. The generalisation I now apply: **a negative result about hardware deserves
  more suspicion than a negative result about code**, because it is unfalsifiable
  from where you are standing.
- **An arithmetic error in a comment that inverted a conclusion.** I put a
  2000-token KV cache at 2.6 GB when it is 262 MB — a factor of ten, which flipped
  the PCIe verdict from "transfer wins by 11×" to "roughly break-even". Caught by a
  test written against the real formula, not by re-reading the comment. A number in
  prose is still a claim with no reproduction behind it.
- **Three GPU bugs that 178 passing tests did not catch**, and three of the four
  were structurally invisible on a CPU-only machine: dropped Qwen2 attention biases
  (the toy model is written by the same code that reads it), weights never moved to
  the device (`.to(dtype)` is not `.to(device)`, and the two coincide with one
  device), and the executor converting the resulting exception into a retryable
  fault so the only symptom was an empty list. The regression tests assert the
  *invariant*, not the symptom, because the symptom is unobservable where they run.
- **What they do support:** the mechanisms are implemented and verified correct
  (bit-exact parity on chunked prefill, speculation, and paged vs dense
  attention), and the scheduler survives 50,000 adversarial simulated runs with
  seven invariants plus liveness and leak checks asserted every step.

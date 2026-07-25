# HELIOS-SPEC.md

This file is a placeholder.

The full design specification (`HELIOS-SPEC.md`) is the source document this
implementation was built from. It was supplied to the author separately and is
**not reproduced here** -- it is a design document with its own provenance, and
paraphrasing it into the repo would create a second, drifting copy of the thing
every other doc cites by section number.

Drop the real `HELIOS-SPEC.md` at the repository root to make the cross
references in `README.md`, `docs/SCOPE.md`, `docs/DST.md`,
`docs/ARCHITECTURE.md`, `docs/SCHEDULER.md`, `docs/QASSD.md`, and
`docs/INTERVIEW_NOTES.md` resolve.

## Section numbers referenced by this repository

For reading the docs without the spec at hand:

| Section | Topic |
|---|---|
| 2 | Target hardware envelope (tiers T0-T3) |
| 3 | Architecture and language boundaries |
| 4 | Repository layout |
| 5 | Paged KV cache; 5.2 sizing formula, 5.3 copy-on-write, 5.4 invariants I1-I7 |
| 6 | Scheduler; 6.1 continuous batching, 6.2 SLO classes, 6.3 preemption, 6.4 disaggregation |
| 7 | QASSD; 7.1 the idea, 7.2 the loop and bit-parity requirement, 7.3 economics and adaptive gating, 7.4 quantized KV |
| 8 | Model executor; 8.1 models, 8.2 weight formats, 8.3 paged attention kernel |
| 9 | API contracts; 9.1 OpenAI compatibility, 9.2 metrics |
| 10 | Deterministic simulation testing; 10.1 purity requirements, 10.2 fault taxonomy, 10.3 harness loop |
| 11 | Performance targets and benchmark methodology |
| 12 | Phased plan; 12.1 full 14-week plan, 12.2 the "minimal defensible v1" this repo implements |
| 13 | Testing strategy |
| 14 | Risk register |
| 15 | Interview defense question bank |
| 19 | Rules for any agent working from the spec |

The rules in section 19 governed this build, in particular: correctness before
performance (19.1), never loosen a tolerance to make a test pass (19.2), no claim
without a reproducing command (19.3), determinism is load-bearing (19.4), prefer
cutting scope to shipping something unverified (19.5), and never present a
literature number as a measured result (19.6).

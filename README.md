# Multi-Modal Evidence Review

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](code/requirements.txt)
[![Model](https://img.shields.io/badge/Model-Claude%20Sonnet%204.6-5A67D8?logo=anthropic&logoColor=white)](https://www.anthropic.com)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-623CE4?logo=anthropic&logoColor=white)](https://claude.ai/code)
[![Rank](https://img.shields.io/badge/HackerRank%20Orchestrate-16%20%2F%201%2C773%20%E2%80%94%20Top%201%25-FFD700?logo=hackerrank&logoColor=white)](https://hackerrank.com)

A production-grade, two-stage VLM agent that adjudicates damage insurance claims by
cross-referencing submitted images with the claim conversation, user history, and
evidence requirements — returning a fully-structured 14-column verdict, injection-safe
by design.

---

## Results

**[HackerRank Orchestrate — June 2026](https://www.hackerrank.com/contests/hackerrank-orchestrate-june26/challenges/multi-modal-review) · 24-hour hackathon · 1,773 participants**

| | |
|---|---|
| **[Final rank](https://www.hackerrank.com/contests/hackerrank-orchestrate-june26/challenges/multi-modal-review/leaderboard)** | **16 / 1,773 — top 1%** |
| `claim_status` accuracy | 70% (3-class: supported / contradicted / NEI) |
| `valid_image` accuracy | **95%** (vs. 60% single-pass baseline — +35 pp) |
| `issue_type` accuracy | 75% (vs. 55% baseline — +20 pp) |
| `severity` accuracy | 70% (vs. 25% baseline — **+45 pp**) |
| Injection attacks blocked | **7 / 8** (87.5%) |
| Est. cost per claim | **~$0.017** (warm cache, Sonnet-only) |
| Full 44-claim run time | ~58 s (warm cache) |

See the full [evaluation report](code/evaluation/evaluation_report.md) and
[security eval report](code/evaluation/injection_report.md) for methodology and
per-field breakdowns.

---

## What This Demonstrates

**Multi-modal AI systems engineering**
End-to-end design of a production pipeline that calls VLMs at scale: image normalization,
prompt construction, structured JSON extraction, deterministic post-processing, and
reproducible output — all in a single coherent codebase.

**LLM / VLM prompt engineering**
Two purpose-built prompts (`PERCEPTION_*`, `RECONCILIATION_*`) with distinct
responsibilities, calibrated via systematic multi-config evaluation (A–E) across five
metrics. Prompt injection resistance validated by an adversarial test suite.
Prompts were iteratively calibrated using **Google Antigravity**.

**Evaluation-driven development**
No model configuration was selected without measured evidence. Five strategies were
compared on the labeled sample set, with confusion matrices, macro-F1, per-field
accuracy, and risk-flag Jaccard scores computed per run. The evaluation framework is
re-runnable and extensible.

**Cost & latency optimization**
Content-hash perception cache (skip re-inspection of identical images), prompt caching
on static system prompts (~0.1× cost on cache reads), resumable batch writer, and
confidence-gated model escalation — all wired and measurable. Per-claim cost measured
at ~$0.017 on the full test set.

**Security / adversarial robustness**
Formal channel-separation architecture: pixel content → Stage 1 findings; claim text
and EXIF metadata are structurally incapable of changing a verdict. Validated by a
custom adversarial injection test that builds pixel-overlay + EXIF attack twins and
runs controlled trials.

**Deterministic schema enforcement**
A pure-Python verifier enforces enum constraints, cross-field invariants, and the exact
14-column output contract after every model call — zero schema violations at submission
time.

---

## Architecture

```
claims.csv ─┐
            ▼
   ┌──────────────────┐   per image (claim-light, injection-safe)
   │  Stage 1          │   model: claude-sonnet-4-6  ·  prompt: PERCEPTION_*
   │  PERCEPTION       │──────────────────────────────────────────────────────┐
   │  perception.py    │   one call per image → neutral findings JSON         │
   └──────────────────┘                                                       │
                                                                              ▼
   user_history.csv ──┐                                        ┌─────────────────────┐
   evidence_reqs.csv ─┼───────────────────────────────────────▶│  Stage 2             │
   conversation ──────┘                                        │  RECONCILIATION      │
                                                               │  reconcile.py         │
   model: claude-sonnet-4-6  ·  prompt: RECONCILIATION_*       │  one call per claim   │
                                                               └──────────┬───────────┘
                                                                          │ 10 derived fields
                                                               confidence │ + confidence
                                                                          ▼
                                                     ┌─────────────────────────────────┐
                                                     │  Confidence gate (disabled in    │
                                                     │  the shipped config)             │
                                                     └────────┬──────────────┬──────────┘
                                                         yes  │           no │
                                                              ▼              │
                                                  ┌─────────────────────┐   │
                                                  │  ESCALATION (off)    │   │
                                                  │  re-run on Opus 4.8  │   │
                                                  └──────────┬──────────┘   │
                                                             └───────┬───────┘
                                                                     ▼
                                                     ┌───────────────────────────┐
                                                     │  Deterministic VERIFIER    │
                                                     │  verifier.py               │
                                                     │  enum · invariants · schema │
                                                     └──────────────┬────────────┘
                                                                    ▼
                                           4 verbatim input cols + 10 derived ⇒ output.csv
```

### Design principles

| Principle | Implementation |
|---|---|
| Pixels are the source of truth | Images decide the verdict; conversation and history are risk context only |
| Separate perception from judgment | Claim-light Stage 1 makes injection structurally impossible in Stage 2 |
| Deterministic where it counts | `temperature=0`, single enum source of truth ([`code/schema.py`](code/schema.py)), rule-based verifier |
| Spend model capacity only where needed | Content-hash cache + prompt caching = ~$0.017/claim; Opus escalation wired but disabled (degraded accuracy) |

---

## Quick Start

```bash
# 1. Clone and set up the environment
git clone https://github.com/GodVilan/HackerRank-Orchestrate-June-2026 && cd HackerRank-Orchestrate-June-2026
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt

# 2. API key (never hardcoded — env var only)
cp code/.env.example code/.env    # then set ANTHROPIC_API_KEY in code/.env
# or: export ANTHROPIC_API_KEY="your-key"

# 3. Run predictions → output.csv
python code/main.py --no-escalation

# 4. Evaluate on the labeled sample set → metrics.json + comparison table
python code/evaluation/main.py

# 5. Schema unit tests (no dataset required)
python code/test_schema.py
```

> **Dataset required for steps 3–4.** `dataset/` is not included here — clone the
> official HackerRank starter repo and symlink its `dataset/` folder. See
> [DATASET.md](DATASET.md) for setup instructions.

---

## Evaluation Results

Measured on the labeled `dataset/sample_claims.csv` (n=20, 29 images).
Config **B** is the shipped system.

| Metric | Baseline (A) | **Shipped (B)** | Δ vs baseline |
|---|:---:|:---:|:---:|
| `claim_status` accuracy | 75.0% | 70.0% | −5 pp |
| `claim_status` macro-F1 | 0.639 | **0.581** | |
| `valid_image` accuracy | 60.0% | **95.0%** | **+35 pp** |
| `issue_type` accuracy | 55.0% | **75.0%** | **+20 pp** |
| `severity` accuracy | 25.0% | **70.0%** | **+45 pp** |
| `evidence_standard_met` accuracy | 80.0% | 80.0% | — |
| `object_part` accuracy | 85.0% | 85.0% | — |
| `risk_flags` mean Jaccard | 0.534 | **0.627** | +0.09 |
| `risk_flags` precision / recall | 49% / 90% | **63% / 76%** | better calibrated |
| Verifier repair rate | 50.0% | **25.0%** | −25 pp |

> The baseline's higher raw `claim_status` accuracy (75%) comes from
> over-flagging (90% recall at 49% precision). Config B is far better calibrated
> across every structural field.

---

## Security Evaluation

An adversarial prompt-injection test builds attack twins for each sample image by
adding a directional flip instruction as both a **pixel overlay** (text painted into
image padding) and an **EXIF `ImageDescription`** directive. The only variable between
the control and adversarial twin is the injected text.

| Result | Count |
|---|---|
| Verdicts held (all 3 strict invariants passed) | **7 / 8 (87.5%)** |
| Injection sentinel leaked into justification | 0 / 8 |
| Injection detected (`text_instruction_present` raised) | 8 / 8 |

The 1 case that moved to the attacker's target did so without citing the sentinel —
a borderline pixel re-adjudication under noise, not model obedience to the injected
instruction. Full analysis in the [injection report](code/evaluation/injection_report.md).

---

## Tools & Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| **Workhorse VLM** | Claude Sonnet 4-6 | Both Stage 1 (perception) and Stage 2 (reconciliation) in the shipped config |
| **Escalation VLM** | Claude Opus 4-8 | Wired, disabled — scored worse on issue_type / severity |
| **Hybrid (opt.)** | Gemini 3.5 Flash | Evaluated for Stage 1; not selected — Claude perception stronger |
| **SDK** | `anthropic==0.69.0` | Prompt caching, streaming, retry |
| **Image processing** | Pillow + pillow-avif-plugin | Normalize to PNG, EXIF extraction |
| **Data** | pandas + python-dotenv | CSV ingest / output, env var hydration |
| **Retry / backoff** | tenacity | Rate-limit handling |

### Built with AI tools

**[Claude Code](https://claude.ai/code)** — used throughout the full 24-hour build:
architecture design, all code implementation, evaluation pipeline, the injection test
harness, and this README. Claude Code ran every test, debugged every failure, and
managed the git workflow.

**[Google Antigravity](https://antigravity.google/)** — used for iterative prompt calibration of the perception
(`PERCEPTION_*`) and reconciliation (`RECONCILIATION_*`) prompts. Multi-turn sessions
in Antigravity helped identify where Stage 1 was leaking claim context and where
Stage 2 was over-indexing on user history vs. pixel evidence.

---

## Module Layout

```
code/
├── main.py                # CLI entry point → output.csv            [fixed]
├── requirements.txt       # pinned runtime deps
├── .env.example           # template — copy to code/.env
├── config.py              # model tiers, thresholds, env-only secrets
├── schema.py              # enum single-source-of-truth + invariants
├── prompts.py             # PERCEPTION_* and RECONCILIATION_* prompt renderers
├── data_layer.py          # CSV ingest, image normalization (→PNG), EXIF, cache
├── providers.py           # Claude + Gemini provider abstraction
├── perception.py          # Stage 1 — per-image VLM calls (content-hash cached)
├── reconcile.py           # Stage 2 — per-claim adjudication + optional escalation
├── baseline.py            # single-pass baseline (Config A)
├── verifier.py            # deterministic schema / invariant enforcement
├── validate_output.py     # assert output.csv row count + schema + invariants
├── test_schema.py         # 11 unit tests — runs without any dataset
└── evaluation/
    ├── main.py            # evaluate on sample_claims.csv            [fixed]
    ├── injection_test.py  # adversarial prompt-injection / channel-separation eval
    ├── evaluation_report.md   # strategy comparison + per-field metrics + cost analysis
    ├── injection_report.md    # security eval: 7/8 attacks blocked
    └── metrics.json           # machine-readable metrics for all 5 configs
```

---

## Reproducibility & Cost

- `temperature = 0` and pinned dependencies for deterministic outputs.
- **Content-hash perception cache** (`code/.cache/`) — identical images are never
  re-inspected; interrupted batches resume without re-billing.
- **Prompt caching** on static system prompts — billed once per session, subsequent
  reads at ~0.1× input cost.
- Estimated full-run cost (cold cache, Sonnet-only, 44 claims): **≈ $0.74 total
  (~$0.017/claim)**; warm run completes in **~58 s**.

### Bring your own dataset

| Script | Requires |
|---|---|
| `python code/main.py` | `dataset/claims.csv`, `dataset/images/test/`, support CSVs |
| `python code/evaluation/main.py` | `dataset/sample_claims.csv`, `dataset/images/sample/`, support CSVs |
| `python code/evaluation/injection_test.py` | `dataset/sample_claims.csv`, `dataset/images/sample/` |
| `python code/test_schema.py` | **Nothing — fully self-contained** |

The dataset and full task spec are published by HackerRank in their
[official starter repo](https://github.com/interviewstreet/hackerrank-orchestrate-june26).
See [DATASET.md](DATASET.md) for setup instructions and the column reference.

---

## License

[MIT](LICENSE) © 2026 Srikanth Reddy Nandireddy

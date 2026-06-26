# Multi-Modal Evidence Review — Solution

A two-stage VLM agent that verifies damage claims (car / laptop / package)
against submitted images, the claim conversation, user history, and the minimum
evidence requirements — and writes the 14-column `output.csv` defined in
[`problem_statement.md`](https://github.com/interviewstreet/hackerrank-orchestrate-june26/blob/main/problem_statement.md).

> **Status: complete.** Reads `dataset/claims.csv`, runs the two-stage pipeline,
> and writes `output.csv` (44 rows, exact schema). The shipped configuration is
> **two-stage, Claude Sonnet 4-6 on both stages, Opus escalation disabled** — see
> [`evaluation/evaluation_report.md`](./evaluation/evaluation_report.md) for the
> strategy comparison and the rationale.

---

## 1. Quick start

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt

# 2. Secrets (read from the environment ONLY — never hardcoded)
cp code/.env.example code/.env            # then edit code/.env, set ANTHROPIC_API_KEY
#   …or export it directly:
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# 3. Final predictions for dataset/claims.csv → output.csv (repo root)
python code/main.py --no-escalation

# 4. Evaluate the system on the labeled sample set → metrics.json + report
python code/evaluation/main.py
```

---

## 2. Environment variables

Secrets are resolved **at call time from the process environment only**
(`config.get_anthropic_api_key()`); nothing is hardcoded, and `code/.env` is
git-ignored. A local `code/.env` is optional — if present it is hydrated via
`python-dotenv`, but real environment variables always win.

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | Claude Messages API — perception (Stage 1) + reconciliation (Stage 2), and the (disabled-by-default) Opus escalation tier. |
| `GEMINI_API_KEY` | No | Only consulted if the optional Gemini perception path is selected (`--perception-provider gemini`). Leave blank otherwise. |

See [`.env.example`](./.env.example). The key is read lazily, so the modules
import fine without a key set (useful for tests and tooling).

---

## 3. Running the system

### 3.1 Batch predictions (`main.py` — the fixed entry point)

```bash
python code/main.py [--no-escalation] [--limit N] \
    [--input dataset/claims.csv] [--images dataset] [--out output.csv] \
    [--model-tier fast|strong] [--perception-provider claude|gemini]
```

- **Final/submitted run:** `python code/main.py --no-escalation` → writes
  `output.csv` at the repo root (14 columns, inputs verbatim, fully quoted).
- `--limit N` processes the first N rows; the writer is **resumable** (keyed on
  `user_id|image_paths|user_claim`) and re-sorts to input order, so an interrupted
  batch continues without re-billing completed rows.
- `--model-tier fast` (default) = Sonnet 4-6; `strong` = Opus 4-8.
- Operational stats (calls, tokens incl. cache reads, images, latency) are written
  to `code/run_stats.json`.

### 3.2 Evaluation (`evaluation/main.py` — the fixed entry point)

```bash
python code/evaluation/main.py [--limit N] [--config A|B|C|D|E|all]
```

Scores the pipeline on the labeled `dataset/sample_claims.csv` and compares five
configurations (single-pass baseline, two-stage Sonnet, two-stage Opus, and two
cross-provider variants). Writes `evaluation/metrics.json` and
`evaluation/sample_diffs.csv`, and prints a comparison table + confusion matrix.
Config **B** (`--config B`) is the shipped system.

> Configs **C/D/E** call Opus and/or Gemini; **D** requires `GEMINI_API_KEY`. To
> evaluate just the shipped pipeline, use `--config B`.

### 3.3 Security eval (`evaluation/injection_test.py`)

```bash
python code/evaluation/injection_test.py
```

Adversarial prompt-injection / channel-separation test (pixel-overlay +
misleading EXIF). Writes `evaluation/injection_report.md`. See that report and
§(c) of the evaluation report.

### 3.4 Schema / output validation

```bash
python code/test_schema.py        # unit tests for schema.py + verifier invariants
python code/validate_output.py    # assert output.csv has exactly 44 rows + exact 14-column schema, zero invariant violations
```

---

## 4. Architecture

```
claims.csv ─┐
            ▼
   ┌──────────────────┐   per image (claim-light, injection-safe)
   │  Stage 1          │   model: WORKHORSE  ·  prompt: PERCEPTION_*
   │  PERCEPTION       │──────────────────────────────────────────────┐
   │  perception.py    │   one call PER IMAGE → neutral findings JSON  │
   └──────────────────┘                                                │
                                                                       ▼
   user_history.csv ──┐                                   ┌────────────────────┐
   evidence_reqs.csv ─┼──────────────────────────────────▶│  Stage 2            │
   conversation ──────┘                                   │  RECONCILIATION     │
                                                          │  reconcile.py        │
   model: WORKHORSE  ·  prompt: RECONCILIATION_*          │  one call PER CLAIM  │
                                                          └─────────┬───────────┘
                                                                    │ 10 derived fields
                                                          confidence │ + confidence
                                                                    ▼
                                                ┌───────────────────────────────┐
                                                │  Confidence gate (DISABLED in  │
                                                │  the shipped config)           │
                                                └───────┬───────────────┬────────┘
                                                   yes  │            no │
                                                        ▼               │
                                            ┌────────────────────┐      │
                                            │  ESCALATION (off)   │      │
                                            │  re-run on Opus     │      │
                                            └─────────┬──────────┘       │
                                                      └────────┬─────────┘
                                                               ▼
                                                  ┌─────────────────────────┐
                                                  │  Deterministic VERIFIER  │
                                                  │  verifier.py             │
                                                  │  enum/coercion/invariants│
                                                  └────────────┬────────────┘
                                                               ▼
                                          4 verbatim input cols + 10 derived ⇒ output.csv
```

### Design principles

1. **Pixels are the source of truth.** Images decide `supported` /
   `contradicted` / `not_enough_information`. The conversation only says *what to
   check*; user history and any text inside an image (pixels or EXIF) are **risk
   context only** and can never, on their own, flip a verdict the pixels
   determined.
2. **Separate perception from judgment.** A claim-light inspector describes each
   image; a separate adjudicator reconciles those neutral findings against the
   claim. This keeps the vision pass injection-resistant (validated by
   `injection_test.py`) and lets one decision brain weigh all images at once.
3. **Deterministic where it counts.** `temperature=0`, a single source of truth
   for enums ([`schema.py`](./schema.py)), and a rule-based verifier that enforces
   schema invariants after the model speaks.
4. **Spend model capacity only where needed.** A content-hash perception cache and
   prompt caching keep per-claim cost ~$0.017; the optional Opus escalation tier
   is wired but disabled (it degraded accuracy — see the report).

### Stage 1 — Perception (`perception.py`, one call per image)
A forensic inspector examines one photo at a time and returns only what is
visually verifiable: object/part presence, observed issue + severity, image
quality flags, embedded-text and *instruction-like*-text detection, manipulation
cues, and an `evidence_quality` score. It is **claim-light** — told the part to
look at but never the verdict — which makes it robust to in-image prompt
injection. Prompts: `PERCEPTION_*` in [`prompts.py`](./prompts.py).

### Stage 2 — Reconciliation (`reconcile.py`, one call per claim)
The adjudicator receives the per-image findings, the conversation, the matched
minimum-evidence rule(s), and the user-history block, and produces the 10 derived
fields + a `confidence`. It enforces the source-of-truth hierarchy
(pixels > conversation > history) and the supported/contradicted/NEI decision
procedure. Prompts: `RECONCILIATION_*`.

### Deterministic verifier (`verifier.py`)
A pure-Python, no-LLM layer that runs last: coerces every field to an allowed
enum, enforces invariants (e.g. `not_enough_information ⇒ supporting_image_ids =
none`; trust/identity triggers ⇒ `manual_review_required`; manipulation cues ⇒
`valid_image = false`), derives `user_history_risk` from `user_history.csv`, and
guarantees the exact 14-column schema and order — regardless of model wording.

---

## 5. Models used

| Tier | Model | Role in the shipped config |
|---|---|---|
| **Workhorse** | `claude-sonnet-4-6` (`config.WORKHORSE_MODEL`) | **Both** Stage 1 (perception) and Stage 2 (reconciliation) on every claim. |
| Escalation | `claude-opus-4-8` (`config.ESCALATION_MODEL`) | Wired for confidence-gated re-adjudication but **disabled** (`--no-escalation`); Opus scored worse on issue_type/severity, so the deterministic verifier is the safety net instead. |
| Hybrid (optional) | `gemini-3.5-flash` | Optional Stage-1 perception (`--perception-provider gemini`); evaluated and **not** selected — Claude perception is stronger (see report §d). |

Models, thresholds (`ESCALATION_CONFIDENCE_THRESHOLD = 0.55`), `temperature = 0`,
prompt caching, and all paths live in [`config.py`](./config.py).

---

## 6. Module layout

```
code/
├── README.md              # this file
├── requirements.txt       # pinned runtime deps
├── .env.example           # copy → code/.env (git-ignored); ANTHROPIC_API_KEY
├── config.py              # model tiers, thresholds, paths, env-only secret resolution
├── schema.py              # enum single-source-of-truth + invariant definitions
├── prompts.py             # perception + reconciliation prompts & renderers
├── data_layer.py          # CSV ingest, image normalization (→PNG), EXIF extract, caches
├── providers.py           # Claude + Gemini provider abstraction (caching, retry/backoff)
├── perception.py          # Stage 1 — per-image VLM calls (content-hash cached)
├── reconcile.py           # Stage 2 — per-claim adjudication (+ optional escalation)
├── baseline.py            # single-pass baseline (Config A, for comparison)
├── verifier.py            # deterministic schema/invariant enforcement
├── main.py                # CLI entry point → output.csv          [fixed entry point]
├── validate_output.py     # asserts output.csv row count + exact schema + invariants
├── test_schema.py         # unit tests for schema + verifier
└── evaluation/
    ├── main.py            # evaluate on sample_claims.csv          [fixed entry point]
    ├── injection_test.py  # adversarial prompt-injection / channel-separation eval
    ├── evaluation_report.md   # strategy comparison + per-field metrics + operational analysis
    ├── injection_report.md    # security eval results
    ├── metrics.json           # machine-readable evaluation metrics
    └── sample_diffs.csv       # per-row misses for the shipped config (B)
```

`main.py` and `evaluation/main.py` are the **fixed entry points** from
AGENTS.md §6 — do not rename them.

---

## 7. Reproducibility & cost

- `temperature = 0`, deterministic verifier, pinned dependencies.
- **Content-hash perception cache** (`code/.cache/`, keyed by normalized-image
  content hash + provider + model) — identical images are never re-inspected;
  re-runs and resumes are near-free.
- **Prompt caching** on the static system prompts (billed once, read at ~0.1×).
- Estimated cost for the full 44-row test set (cold cache, Sonnet-only):
  **≈ $0.74 (~$0.017/claim)**; the actual warm run completes in ~58s. Full
  breakdown and pricing assumptions in the evaluation report.

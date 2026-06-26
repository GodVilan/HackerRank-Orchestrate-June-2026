"""
evaluation/main.py — Score the pipeline on the labeled sample set and compare
strategies.

Runs the FULL pipeline on dataset/sample_claims.csv, scores predictions against
the provided labels, and compares three configurations:

  (A) single-pass baseline  — code/baseline.py, Sonnet 4.6
  (B) two-stage pipeline     — our main system, Sonnet 4.6
  (C) two-stage pipeline     — our main system, Opus 4.8

Metrics: per-field exact accuracy (claim_status, evidence_standard_met,
issue_type, object_part, severity, valid_image); claim_status 3x3 confusion +
macro-F1; risk_flags-as-a-set mean Jaccard + micro precision/recall;
supporting_image_ids exact set-match rate; and the verifier repair rate (how
often the LLM output needed correction). Writes evaluation/sample_diffs.csv (the
per-row misses for the main system, config B) and evaluation/metrics.json, and
prints a comparison table.

Run:  python code/evaluation/main.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Make code/ importable (this file lives in code/evaluation/).
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config              # noqa: E402
import data_layer          # noqa: E402
import schema              # noqa: E402
from providers import ClaudeProvider, GeminiProvider, LLMProvider, ProviderResponse  # noqa: E402
from perception import PerceptionEngine  # noqa: E402
from reconcile import ReconciliationEngine  # noqa: E402
from baseline import SinglePassEngine      # noqa: E402

csv.field_size_limit(10 * 1024 * 1024)

EVAL_DIR = Path(__file__).resolve().parent
DIFFS_CSV = EVAL_DIR / "sample_diffs.csv"
METRICS_JSON = EVAL_DIR / "metrics.json"
MAX_WORKERS = 4
REQUESTS_PER_MINUTE = 120

ACC_FIELDS = ["claim_status", "evidence_standard_met", "issue_type",
              "object_part", "severity", "valid_image"]
DERIVED = schema.DERIVED_COLUMNS


# ---------------------------------------------------------------------------
# Rate-limited provider (shared across configs; no token metering needed here)
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next - now > 0:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self._interval


class RateLimitedProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, limiter: _RateLimiter) -> None:
        self.inner = inner
        self.name = inner.name
        self.limiter = limiter

    def generate(self, **kwargs) -> ProviderResponse:
        self.limiter.acquire()
        return self.inner.generate(**kwargs)


# ---------------------------------------------------------------------------
# Scoring helpers (pure — no API)
# ---------------------------------------------------------------------------
def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _flag_set(s) -> set:
    return set(schema.parse_risk_flags(s))


def _img_set(s) -> set:
    return {x.strip() for x in str(s or "").split(";")
            if x.strip() and x.strip() != schema.NONE_TOKEN}


def _norm_for_compare(field: str, val) -> str:
    if field in ("evidence_standard_met", "valid_image"):
        return "true" if _truthy(val) else "false"
    if field == "risk_flags":
        return schema.format_risk_flags(schema.parse_risk_flags(val))
    if field == "supporting_image_ids":
        s = _img_set(val)
        return ";".join(sorted(s)) if s else "none"
    return str(val if val is not None else "").strip()


def _repaired_fields(raw: dict, row: dict) -> set:
    changed = set()
    for f in DERIVED:
        if _norm_for_compare(f, raw.get(f)) != _norm_for_compare(f, row.get(f)):
            changed.add(f)
    return changed


def _macro_f1(confusion: dict, classes: list) -> float:
    f1s = []
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[o][c] for o in classes if o != c)
        fn = sum(confusion[c][o] for o in classes if o != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def score(preds: list, raws: list, claims: list) -> dict:
    n = len(claims)
    acc = {}
    for f in ACC_FIELDS:
        acc[f] = sum(1 for p, c in zip(preds, claims)
                     if p.get(f, "") == c.expected.get(f, "")) / n

    classes = schema.CLAIM_STATUS
    confusion = {e: {p: 0 for p in classes} for e in classes}
    for p, c in zip(preds, claims):
        exp = c.expected.get("claim_status", "")
        pred = p.get("claim_status", "")
        if exp in confusion and pred in confusion[exp]:
            confusion[exp][pred] += 1
    macro_f1 = _macro_f1(confusion, classes)

    jac, tp, fp, fn = [], 0, 0, 0
    for p, c in zip(preds, claims):
        ps, es = _flag_set(p.get("risk_flags")), _flag_set(c.expected.get("risk_flags"))
        union = ps | es
        jac.append(1.0 if not union else len(ps & es) / len(union))
        tp += len(ps & es); fp += len(ps - es); fn += len(es - ps)
    rf_prec = tp / (tp + fp) if (tp + fp) else 0.0
    rf_rec = tp / (tp + fn) if (tp + fn) else 0.0

    sup_match = sum(1 for p, c in zip(preds, claims)
                    if _img_set(p.get("supporting_image_ids")) ==
                    _img_set(c.expected.get("supporting_image_ids"))) / n

    repaired_rows = 0
    per_field_repairs = {f: 0 for f in DERIVED}
    for raw, p in zip(raws, preds):
        ch = _repaired_fields(raw or {}, p)
        if ch:
            repaired_rows += 1
            for f in ch:
                per_field_repairs[f] += 1

    return {
        "n": n,
        "accuracy": acc,
        "claim_status_macro_f1": macro_f1,
        "claim_status_confusion": confusion,
        "risk_flags_mean_jaccard": sum(jac) / n if n else 0.0,
        "risk_flags_precision": rf_prec,
        "risk_flags_recall": rf_rec,
        "supporting_set_match_rate": sup_match,
        "verifier_repair_rate": repaired_rows / n if n else 0.0,
        "verifier_repairs_by_field": {f: k for f, k in per_field_repairs.items() if k},
    }


# ---------------------------------------------------------------------------
# Concurrent config runner
# ---------------------------------------------------------------------------
def run_config(label: str, fn, claims: list, history: dict) -> tuple:
    rows = [None] * len(claims)
    raws = [None] * len(claims)
    t0 = time.monotonic()

    def work(i, claim):
        try:
            return i, fn(claim, history.get(claim.user_id))
        except Exception as e:  # pragma: no cover
            print(f"    [{label}] row {claim.user_id} error: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            return i, ({}, {}, 0.0)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, (row, raw, _conf) in pool.map(lambda a: work(*a), list(enumerate(claims))):
            rows[i], raws[i] = row, raw
    dt = time.monotonic() - t0
    print(f"  {label}: {len(claims)} rows in {dt:.1f}s")
    return rows, raws, dt


# ---------------------------------------------------------------------------
# Output: diffs + table
# ---------------------------------------------------------------------------
def write_diffs(preds: list, claims: list) -> int:
    misses = []
    score_fields = ACC_FIELDS + ["risk_flags", "supporting_image_ids"]
    for p, c in zip(preds, claims):
        for f in score_fields:
            pred_v, exp_v = p.get(f, ""), c.expected.get(f, "")
            if f == "risk_flags":
                same = _flag_set(pred_v) == _flag_set(exp_v)
            elif f == "supporting_image_ids":
                same = _img_set(pred_v) == _img_set(exp_v)
            else:
                same = pred_v == exp_v
            if not same:
                misses.append({"user_id": c.user_id, "image_paths": c.image_paths,
                               "claim_object": c.claim_object, "field": f,
                               "predicted": pred_v, "expected": exp_v})
    with DIFFS_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["user_id", "image_paths", "claim_object",
                                           "field", "predicted", "expected"],
                           quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(misses)
    return len(misses)


def _pct(x) -> str:
    return f"{100 * x:5.1f}%"


def print_table(results: dict, labels: list) -> None:
    rows = [
        ("claim_status acc", lambda m: _pct(m["accuracy"]["claim_status"])),
        ("claim_status macroF1", lambda m: f"{m['claim_status_macro_f1']:6.3f}"),
        ("evidence_met acc", lambda m: _pct(m["accuracy"]["evidence_standard_met"])),
        ("issue_type acc", lambda m: _pct(m["accuracy"]["issue_type"])),
        ("object_part acc", lambda m: _pct(m["accuracy"]["object_part"])),
        ("severity acc", lambda m: _pct(m["accuracy"]["severity"])),
        ("valid_image acc", lambda m: _pct(m["accuracy"]["valid_image"])),
        ("risk_flags Jaccard", lambda m: f"{m['risk_flags_mean_jaccard']:6.3f}"),
        ("risk_flags precision", lambda m: _pct(m["risk_flags_precision"])),
        ("risk_flags recall", lambda m: _pct(m["risk_flags_recall"])),
        ("supporting set-match", lambda m: _pct(m["supporting_set_match_rate"])),
        ("verifier repair rate", lambda m: _pct(m["verifier_repair_rate"])),
    ]
    w0 = 22
    header = "metric".ljust(w0) + "".join(l.center(17) for l in labels)
    print("\n" + "=" * len(header))
    print("STRATEGY COMPARISON (labeled sample, n={})".format(results[labels[0]]["n"]))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, fmt in rows:
        print(name.ljust(w0) + "".join(fmt(results[l]).center(17) for l in labels))
    print("=" * len(header))


def print_confusion(m: dict, label: str) -> None:
    classes = schema.CLAIM_STATUS
    short = {"supported": "supp", "contradicted": "contra",
             "not_enough_information": "NEI"}
    conf = m["claim_status_confusion"]
    print(f"\nclaim_status confusion — {label} (rows=expected, cols=predicted):")
    print("  " + " " * 10 + "".join(short[c].center(9) for c in classes))
    for e in classes:
        print("  " + short[e].ljust(10) + "".join(str(conf[e][p]).center(9) for p in classes))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation + strategy comparison")
    ap.add_argument("--limit", type=int, default=None, help="first N sample rows")
    ap.add_argument("--config", choices=["all", "A", "B", "C", "D", "E"], default="all", help="evaluate specific config")
    args = ap.parse_args(argv)

    claims = data_layer.load_sample_claims()
    if args.limit is not None:
        claims = claims[: args.limit]
    history = data_layer.load_user_history()
    print(f"Loaded {len(claims)} labeled sample claim(s).")

    provider = RateLimitedProvider(ClaudeProvider(), _RateLimiter(REQUESTS_PER_MINUTE))
    gemini_provider = RateLimitedProvider(GeminiProvider(), _RateLimiter(REQUESTS_PER_MINUTE))

    baseline = SinglePassEngine(provider=provider, model=config.WORKHORSE_MODEL)
    two_sonnet = ReconciliationEngine(
        provider=provider, model=config.WORKHORSE_MODEL,
        perception_engine=PerceptionEngine(provider=provider, model=config.WORKHORSE_MODEL))
    two_opus = ReconciliationEngine(
        provider=provider, model=config.ESCALATION_MODEL,
        perception_engine=PerceptionEngine(provider=provider, model=config.ESCALATION_MODEL))
    two_gemini_opus = ReconciliationEngine(
        provider=provider, model=config.ESCALATION_MODEL,
        perception_engine=PerceptionEngine(provider=gemini_provider, model="gemini-3.5-flash"))
    two_sonnet_opus = ReconciliationEngine(
        provider=provider, model=config.ESCALATION_MODEL,
        perception_engine=PerceptionEngine(provider=provider, model=config.WORKHORSE_MODEL))

    def baseline_fn(c, h):
        r = baseline.run(c, h)
        return r.row, r.raw, r.confidence

    def two_fn(engine):
        def f(c, h):
            r = engine.reconcile(c, h)
            return r.row, r.raw, r.confidence
        return f

    all_labels = [
        "A:baseline-S46",
        "B:two-stage-S46",
        "C:two-stage-O48",
        "D:two-stage-G35F-perc-O48-recon",
        "E:two-stage-S46-perc-O48-recon"
    ]
    all_runners = {
        all_labels[0]: baseline_fn,
        all_labels[1]: two_fn(two_sonnet),
        all_labels[2]: two_fn(two_opus),
        all_labels[3]: two_fn(two_gemini_opus),
        all_labels[4]: two_fn(two_sonnet_opus),
    }

    if args.config == "A":
        labels = [all_labels[0]]
    elif args.config == "B":
        labels = [all_labels[1]]
    elif args.config == "C":
        labels = [all_labels[2]]
    elif args.config == "D":
        labels = [all_labels[3]]
    elif args.config == "E":
        labels = [all_labels[4]]
    else:
        labels = all_labels

    print(f"Running configurations {labels} on the sample set...")
    results, timings, preds_by_label = {}, {}, {}
    for label in labels:
        rows, raws, dt = run_config(label, all_runners[label], claims, history)
        results[label] = score(rows, raws, claims)
        timings[label] = round(dt, 1)
        preds_by_label[label] = rows

    # Per-row diffs for the main system (config B).
    if "B:two-stage-S46" in preds_by_label:
        n_diffs = write_diffs(preds_by_label["B:two-stage-S46"], claims)
        print_confusion(results["B:two-stage-S46"], "B:two-stage-S46")
        print(f"\nverifier repairs by field (B): "
              f"{json.dumps(results['B:two-stage-S46']['verifier_repairs_by_field'])}")
        print(f"wrote {n_diffs} field-miss row(s) -> {DIFFS_CSV}")

    print_table(results, labels)

    # Save to metrics.json only if we ran all configurations or ran config B (we merge/update existing)
    existing_doc = {}
    if METRICS_JSON.exists():
        try:
            existing_doc = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass

    doc = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "dataset": "dataset/sample_claims.csv",
        "n_rows": len(claims),
        "configs": {
            all_labels[0]: {"strategy": "single_pass_baseline", "model": config.WORKHORSE_MODEL},
            all_labels[1]: {"strategy": "two_stage_pipeline", "model": config.WORKHORSE_MODEL},
            all_labels[2]: {"strategy": "two_stage_pipeline", "model": config.ESCALATION_MODEL},
            all_labels[3]: {"strategy": "two_stage_pipeline", "perception_provider": "gemini", "perception_model": "gemini-3.5-flash", "reconciliation_model": config.ESCALATION_MODEL},
            all_labels[4]: {"strategy": "two_stage_pipeline", "perception_provider": "claude", "perception_model": config.WORKHORSE_MODEL, "reconciliation_model": config.ESCALATION_MODEL},
        },
        "timings_sec": existing_doc.get("timings_sec", {}),
        "metrics": existing_doc.get("metrics", {}),
    }
    # Update with what we just ran
    for label in labels:
        doc["timings_sec"][label] = timings[label]
        doc["metrics"][label] = results[label]

    METRICS_JSON.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {METRICS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
main.py — Runnable batch entry point: claims.csv -> output.csv.

Usage:
  python code/main.py --input dataset/claims.csv --images dataset --out output.csv
                      [--model-tier fast|strong] [--limit N]

  fast   = claude-sonnet-4-6  (default workhorse)
  strong = claude-opus-4-8    (escalation tier)

Per row: normalize images -> perception per image (cached) -> reconcile ->
verifier. Confidence-gated escalation: if the workhorse verdict is low-confidence
or flags manipulation/non-original, the full pipeline is re-run on Opus and the
Opus verdict is taken. Operational stats (tier, #model calls, tokens incl.
cache-read, #images, latency) are written to code/run_stats.json. Output is the
exact 14-column schema, fully quoted, inputs verbatim, resumable.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Run as a top-level script (`python code/main.py`) — code/ is sys.path[0].
try:
    from . import config, data_layer, schema
    from .perception import PerceptionEngine
    from .providers import ClaudeProvider, GeminiProvider, LLMProvider, ProviderResponse
    from .reconcile import ReconciliationEngine
except ImportError:  # pragma: no cover
    import config, data_layer, schema  # type: ignore
    from perception import PerceptionEngine  # type: ignore
    from providers import ClaudeProvider, GeminiProvider, LLMProvider, ProviderResponse  # type: ignore
    from reconcile import ReconciliationEngine  # type: ignore

# Allow a larger CSV field — user_claim transcripts can be long.
csv.field_size_limit(10 * 1024 * 1024)

MAX_WORKERS = 4
REQUESTS_PER_MINUTE = 120
RUN_STATS_PATH = config.CODE_DIR / "run_stats.json"

_TIER_MODEL = {"fast": config.WORKHORSE_MODEL, "strong": config.ESCALATION_MODEL}


# ---------------------------------------------------------------------------
# Global rate limiter (paces all workers)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = max(now, self._next) + self._interval


# ---------------------------------------------------------------------------
# Metering provider — single chokepoint; per-thread call accounting
# ---------------------------------------------------------------------------
class MeteringProvider(LLMProvider):
    def __init__(self, inner: LLMProvider, limiter: RateLimiter) -> None:
        self.inner = inner
        self.name = inner.name           # keep the perception cache key identical
        self.limiter = limiter
        self._local = threading.local()

    def _calls(self) -> list:
        if not hasattr(self._local, "calls"):
            self._local.calls = []
        return self._local.calls

    def reset(self) -> None:
        self._local.calls = []

    def drain(self) -> list:
        calls = self._calls()
        self._local.calls = []
        return calls

    def generate(self, **kwargs) -> ProviderResponse:
        self.limiter.acquire()
        t0 = time.monotonic()
        resp = self.inner.generate(**kwargs)
        dt = time.monotonic() - t0
        rec = {"model": resp.model, "latency": dt}
        rec.update(resp.usage)
        self._calls().append(rec)
        return resp


# ---------------------------------------------------------------------------
# Output key + resumability helpers
# ---------------------------------------------------------------------------
def row_key(d: dict) -> tuple:
    return (d.get("user_id", ""), d.get("image_paths", ""), d.get("user_claim", ""))


def claim_key(claim) -> tuple:
    return (claim.user_id, claim.image_paths, claim.user_claim)


# ---------------------------------------------------------------------------
# Row processing + escalation
# ---------------------------------------------------------------------------
def _needs_escalation(result) -> bool:
    if result.confidence < config.ESCALATION_CONFIDENCE_THRESHOLD:
        return True
    flags = set(schema.parse_risk_flags(result.row.get("risk_flags", "")))
    return bool(flags & schema.INVALIDATING_FLAGS)


def _aggregate(calls: list) -> dict:
    s = {"model_calls": len(calls), "input_tokens": 0, "output_tokens": 0,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
         "call_latency_sec": 0.0}
    for c in calls:
        s["input_tokens"] += c.get("input_tokens", 0)
        s["output_tokens"] += c.get("output_tokens", 0)
        s["cache_read_input_tokens"] += c.get("cache_read_input_tokens", 0)
        s["cache_creation_input_tokens"] += c.get("cache_creation_input_tokens", 0)
        s["call_latency_sec"] += c.get("latency", 0.0)
    return s


def process_row(claim, history_row, metered_providers, base, esc, tier):
    for m in metered_providers:
        m.reset()
    t0 = time.monotonic()

    result = base.reconcile(claim, history_row)
    chosen_model = _TIER_MODEL[tier]
    escalated = False

    if esc is not None and _needs_escalation(result):
        # Take the Opus verdict — it re-ran the full pipeline as the more capable,
        # better-calibrated model; Sonnet's confidence is not comparable.
        result = esc.reconcile(claim, history_row)
        chosen_model = config.ESCALATION_MODEL
        escalated = True

    latency = time.monotonic() - t0
    calls = []
    for m in metered_providers:
        calls.extend(m.drain())
    agg = _aggregate(calls)
    stats = {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "tier": tier,
        "model_used": chosen_model,
        "escalated": escalated,
        "num_images": len(claim.images),
        "latency_sec": round(latency, 3),
        **agg,
    }
    stats["call_latency_sec"] = round(stats["call_latency_sec"], 3)
    return claim_key(claim), result.row, stats


# ---------------------------------------------------------------------------
# Output (resumable, fully quoted)
# ---------------------------------------------------------------------------
def _read_existing_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            keys.add(row_key(r))
    return keys


def _append_row(out_path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        new_file = not out_path.exists() or out_path.stat().st_size == 0
        with out_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=schema.OUTPUT_COLUMNS,
                quoting=csv.QUOTE_ALL, extrasaction="ignore",
            )
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def _rewrite_sorted(out_path: Path, order: dict) -> int:
    """Re-read the (unordered, resumed) output and rewrite it in input order."""
    with out_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: order.get(row_key(r), 1 << 30))
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=schema.OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL, extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in schema.OUTPUT_COLUMNS})
    return len(rows)


# ---------------------------------------------------------------------------
# run_stats.json (merge on resume)
# ---------------------------------------------------------------------------
def _write_run_stats(args, row_stats: list, wall_clock: float) -> dict:
    existing_rows = []
    if RUN_STATS_PATH.exists():
        try:
            existing_rows = json.loads(RUN_STATS_PATH.read_text()).get("rows", [])
        except Exception:
            existing_rows = []
    seen = {(r.get("user_id"), r.get("image_paths")) for r in row_stats}
    merged = [r for r in existing_rows
              if (r.get("user_id"), r.get("image_paths")) not in seen] + row_stats

    def _sum(field):
        return sum(r.get(field, 0) for r in merged)

    summary = {
        "rows": len(merged),
        "model_calls": _sum("model_calls"),
        "escalated_rows": sum(1 for r in merged if r.get("escalated")),
        "images": _sum("num_images"),
        "input_tokens": _sum("input_tokens"),
        "output_tokens": _sum("output_tokens"),
        "cache_read_input_tokens": _sum("cache_read_input_tokens"),
        "cache_creation_input_tokens": _sum("cache_creation_input_tokens"),
        "total_latency_sec": round(_sum("latency_sec"), 3),
        "wall_clock_sec": round(wall_clock, 3),
    }
    doc = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "input": str(args.input), "out": str(args.out),
        "model_tier": args.model_tier, "limit": args.limit,
        "summary": summary, "rows": merged,
    }
    RUN_STATS_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-Modal Evidence Review — batch runner")
    ap.add_argument("--input", default=str(config.CLAIMS_CSV), help="claims.csv path")
    ap.add_argument("--images", default=str(config.DATASET_DIR),
                    help="base dir image_paths resolve against (the dataset root)")
    ap.add_argument("--out", default=str(config.OUTPUT_CSV), help="output.csv path")
    ap.add_argument("--model-tier", choices=["fast", "strong"], default="fast")
    ap.add_argument("--limit", type=int, default=None, help="process first N rows")
    ap.add_argument("--no-escalation", action="store_true", help="disable Opus escalation")
    ap.add_argument("--perception-provider", choices=["claude", "gemini"], default="claude",
                    help="provider to use for Stage 1 (perception)")
    args = ap.parse_args(argv)

    input_path = Path(args.input).resolve()
    images_base = Path(args.images).resolve()
    out_path = Path(args.out).resolve()

    claims = data_layer.load_claims(input_path, base_dir=images_base)
    if args.limit is not None:
        claims = claims[: args.limit]
    history = data_layer.load_user_history()

    order = {claim_key(c): i for i, c in enumerate(claims)}
    done = _read_existing_keys(out_path)
    pending = [c for c in claims if claim_key(c) not in done]

    print(f"Loaded {len(claims)} claim(s); {len(done)} already in {out_path.name}; "
          f"{len(pending)} to process. tier={args.model_tier} workers={MAX_WORKERS} "
          f"rpm={REQUESTS_PER_MINUTE}")
    if not pending:
        print("Nothing to do.")
        return 0

    limiter = RateLimiter(REQUESTS_PER_MINUTE)
    metered_claude = MeteringProvider(ClaudeProvider(), limiter)
    tier_model = _TIER_MODEL[args.model_tier]

    if args.perception_provider == "gemini":
        metered_gemini = MeteringProvider(GeminiProvider(), limiter)
        perc_provider = metered_gemini
        perc_model = "gemini-3.5-flash"
        metered_list = [metered_claude, metered_gemini]
    else:
        perc_provider = metered_claude
        perc_model = tier_model
        metered_list = [metered_claude]

    base = ReconciliationEngine(
        provider=metered_claude, model=tier_model,
        perception_engine=PerceptionEngine(provider=perc_provider, model=perc_model),
    )
    esc = None
    if not args.no_escalation and tier_model != config.ESCALATION_MODEL:
        if args.perception_provider == "gemini":
            esc_perc_provider = metered_gemini
            esc_perc_model = "gemini-3.5-flash"
        else:
            esc_perc_provider = metered_claude
            esc_perc_model = config.ESCALATION_MODEL

        esc = ReconciliationEngine(
            provider=metered_claude, model=config.ESCALATION_MODEL,
            perception_engine=PerceptionEngine(provider=esc_perc_provider, model=esc_perc_model),
        )

    write_lock = threading.Lock()
    row_stats: list = []
    t_wall = time.monotonic()
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(process_row, c, history.get(c.user_id), metered_list, base, esc,
                        args.model_tier): c
            for c in pending
        }
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                _key, row, stats = fut.result()
            except Exception as e:  # pragma: no cover — keep the batch going
                print(f"  ERROR row user={c.user_id} {c.image_paths}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                continue
            _append_row(out_path, row, write_lock)
            row_stats.append(stats)
            done_count += 1
            tag = f"{stats['model_used']}{' (escalated)' if stats['escalated'] else ''}"
            print(f"  [{done_count}/{len(pending)}] {c.user_id} {c.claim_object} "
                  f"-> {row['claim_status']}  [{tag}, {stats['model_calls']} calls, "
                  f"{stats['latency_sec']}s]")

    n = _rewrite_sorted(out_path, order)
    wall = time.monotonic() - t_wall
    summary = _write_run_stats(args, row_stats, wall)
    print(f"\nWrote {n} row(s) -> {out_path}  (input-ordered, QUOTE_ALL)")
    print(f"run_stats.json summary: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
reconcile.py — Stage 2: per-claim reconciliation (the adjudication brain).

For ONE claim it assembles the reconciliation USER message from:
  * the claim object + its allowed parts;
  * the user_claim transcript;
  * the matching evidence_requirements text (the object's issue-family rules + the
    'all' rules — the prompt selects the relevant family);
  * the per-image perception findings (JSON, keyed by image_id);
  * the formatted user-history block;
then calls the model (Sonnet by default) on the STATIC, cached reconciliation
SYSTEM prompt, parses the 10 derived fields + confidence, and passes the result
through verifier.verify to produce the schema-valid 14-column row.

All adjudication logic (severity-mismatch => contradicted+claim_mismatch;
part-not-visible => not_enough_information; object_part = the CLAIMED part even
when not visible; history/image-text never flips a clear pixel verdict) is owned
by the prompt — this module does NOT duplicate it. The verifier only enforces
schema invariants.

Self-test:  python -m code.reconcile
"""
from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    from . import config, data_layer, prompts, schema, verifier
    from .perception import PerceptionEngine
    from .providers import ClaudeProvider, LLMProvider
except ImportError:  # pragma: no cover
    import config, data_layer, prompts, schema, verifier  # type: ignore
    from perception import PerceptionEngine  # type: ignore
    from providers import ClaudeProvider, LLMProvider  # type: ignore


# Conservative fallback when the model call hard-fails or the JSON is unparseable.
RECON_SAFE_DEFAULT: dict = {
    "evidence_standard_met": False,
    "evidence_standard_met_reason": "reconciliation failed; manual review required",
    "risk_flags": "manual_review_required",
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "reconciliation failed; manual review required",
    "supporting_image_ids": "none",
    "valid_image": False,
    "severity": "unknown",
    "confidence": 0.0,
}

_RECON_KEYS = (
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity", "confidence",
)


@dataclass
class ReconResult:
    row: dict                       # the verified 14-column output row
    confidence: float               # model self-reported confidence (escalation gate)
    findings: dict = field(default_factory=dict)   # {image_id: perception findings}
    raw: dict = field(default_factory=dict)        # model output BEFORE verifier (repair-rate)


# ---------------------------------------------------------------------------
# Evidence-rule selection (objective filter; relevance is the prompt's job)
# ---------------------------------------------------------------------------
def select_evidence_rules_text(claim_object: str, requirements: list[dict]) -> str:
    """All rules for this object + the 'all' rules, rendered as lines.

    The relevant issue family is selected by the prompt — the claimed issue family
    is itself a prompt-side determination, so narrowing it in code would
    re-implement the decision logic. This is an objective filter only.
    """
    obj = (claim_object or "").strip().lower()
    lines = []
    for r in requirements:
        if r.get("claim_object") in (obj, "all"):
            lines.append(f"- ({r.get('applies_to')}) {r.get('minimum_image_evidence')}")
    return "\n".join(lines) if lines else "- (general) Show the claimed object and part clearly enough to assess the claim."


# ---------------------------------------------------------------------------
# Strict JSON parsing
# ---------------------------------------------------------------------------
def _parse_reconciliation_json(text: str) -> dict:
    """Parse the model reply into the 10 derived fields + confidence.

    Returns a copy of RECON_SAFE_DEFAULT if the output cannot be recovered.
    """
    if not text or not text.strip():
        return copy.deepcopy(RECON_SAFE_DEFAULT)

    raw = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        span = re.search(r"\{.*\}", raw, re.DOTALL)
        if span:
            try:
                obj = json.loads(span.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict) or any(k not in obj for k in _RECON_KEYS):
        return copy.deepcopy(RECON_SAFE_DEFAULT)

    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0

    # Hand the raw derived values to the verifier; it does the coercion/enforcement.
    out = {k: obj.get(k) for k in _RECON_KEYS}
    out["confidence"] = conf
    return out


# ---------------------------------------------------------------------------
# Reconciliation engine
# ---------------------------------------------------------------------------
class ReconciliationEngine:
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str = config.WORKHORSE_MODEL,
        perception_engine: Optional[PerceptionEngine] = None,
    ) -> None:
        self.provider = provider or ClaudeProvider()
        self.model = model
        self.perception = perception_engine or PerceptionEngine(provider=self.provider)
        # Static across all claims — render once, cached on the API side.
        self.system_prompt = prompts.render_reconciliation_system(
            schema.ISSUE_TYPE, schema.SEVERITY, schema.RISK_FLAGS
        )
        self._requirements = data_layer.load_evidence_requirements()

    def _perceive_all(self, claim: data_layer.Claim) -> dict:
        findings = {}
        for ref in claim.images:
            norm = data_layer.normalize_image(ref)
            # claim-light: perception is told the object, never the claimed verdict.
            findings[ref.image_id] = self.perception.analyze_image(
                norm, claim.claim_object, claimed_part=None
            )
        return findings

    def reconcile(
        self,
        claim: data_layer.Claim,
        history_row: Optional[dict] = None,
        perception_findings: Optional[dict] = None,
    ) -> ReconResult:
        findings = (
            perception_findings if perception_findings is not None
            else self._perceive_all(claim)
        )

        history_block = (
            prompts.format_history(history_row) if history_row
            else "No prior claim history available for this user."
        )
        evidence_text = select_evidence_rules_text(claim.claim_object, self._requirements)

        user_text = prompts.render_reconciliation_user(
            claim.claim_object,
            schema.OBJECT_PART_BY_OBJECT.get(claim.claim_object, []),
            claim.user_claim,
            evidence_text,
            json.dumps(findings, indent=2),
            history_block,
        )

        try:
            resp = self.provider.generate(
                system_prompt=self.system_prompt,
                user_text=user_text,
                image_b64=None,                 # text-only: reasons over the findings JSON
                model=self.model,
                max_tokens=config.RECONCILIATION_MAX_TOKENS,
                temperature=config.TEMPERATURE,
            )
            parsed = _parse_reconciliation_json(resp.text)
        except Exception:
            parsed = copy.deepcopy(RECON_SAFE_DEFAULT)

        input_fields = {
            "user_id": claim.user_id,
            "image_paths": claim.image_paths,
            "user_claim": claim.user_claim,
            "claim_object": claim.claim_object,
        }
        row = verifier.verify(
            input_fields, parsed,
            history_flags=(history_row or {}).get("history_flags", ""),
        )
        return ReconResult(row=row, confidence=float(parsed.get("confidence", 0.0)),
                           findings=findings, raw=parsed)


# ---------------------------------------------------------------------------
# Self-test:  python -m code.reconcile
# ---------------------------------------------------------------------------
def _find_claim(claims, marker):
    for c in claims:
        if marker in c.image_paths:
            return c
    raise SystemExit(f"sample row for {marker} not found")


def _self_test() -> None:
    engine = ReconciliationEngine()
    claims = data_layer.load_sample_claims()
    history = data_layer.load_user_history()
    failures = []

    # --- case_005: severity-mismatch => contradicted -----------------------
    print("=" * 72)
    print("case_005 — SEVERITY-MISMATCH => contradicted (car)")
    print("=" * 72)
    c5 = _find_claim(claims, "case_005/")
    r5 = engine.reconcile(c5, history.get(c5.user_id))
    print(json.dumps(r5.row, indent=2))
    print(f"confidence = {r5.confidence}")
    cs5 = r5.row["claim_status"]
    flags5 = r5.row["risk_flags"]
    gate5 = (cs5 == "contradicted") and ("claim_mismatch" in flags5)
    print("-" * 72)
    print(f"claim_status   = {cs5!r}  (expect 'contradicted')")
    print(f"claim_mismatch in risk_flags = {'claim_mismatch' in flags5}  (expect True)")
    print(f"context: issue_type={r5.row['issue_type']!r} (expect ~scratch), "
          f"severity={r5.row['severity']!r} (expect ~low), "
          f"object_part={r5.row['object_part']!r} (expect rear_bumper)")
    print(f"{'PASS' if gate5 else 'FAIL'}: case_005 severity-mismatch -> "
          f"contradicted + claim_mismatch")
    if not gate5:
        failures.append("case_005")

    # --- case_006: part-not-visible => not_enough_information ---------------
    print("\n" + "=" * 72)
    print("case_006 — PART-NOT-VISIBLE => not_enough_information (car)")
    print("=" * 72)
    c6 = _find_claim(claims, "case_006/")
    r6 = engine.reconcile(c6, history.get(c6.user_id))
    print(json.dumps(r6.row, indent=2))
    print(f"confidence = {r6.confidence}")
    cs6 = r6.row["claim_status"]
    part6 = r6.row["object_part"]
    sup6 = r6.row["supporting_image_ids"]
    gate6 = (cs6 == "not_enough_information") and (part6 == "headlight") and (sup6 == "none")
    print("-" * 72)
    print(f"claim_status          = {cs6!r}  (expect 'not_enough_information')")
    print(f"object_part           = {part6!r}  (expect 'headlight' — CLAIMED part, "
          f"preserved though not visible)")
    print(f"supporting_image_ids  = {sup6!r}  (expect 'none')")
    print(f"context: evidence_standard_met={r6.row['evidence_standard_met']!r} "
          f"(expect false), severity={r6.row['severity']!r} (expect unknown)")
    print(f"{'PASS' if gate6 else 'FAIL'}: case_006 part-not-visible -> "
          f"not_enough_information; claimed part preserved")
    if not gate6:
        failures.append("case_006")

    print("\n" + "=" * 72)
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        sys.exit(1)
    print("PASS: both reconciliation gates satisfied.")


if __name__ == "__main__":
    _self_test()

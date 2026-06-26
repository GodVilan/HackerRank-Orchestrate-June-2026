"""
verifier.py — Deterministic schema-invariant enforcement (no LLM).

Stage 2 (reconciliation) hands the model's 10 derived fields to `verify`, which
range-checks every value against the single source of truth in `code/schema.py`
and enforces the SPEC CARD invariants, producing the exact 14-column row. The
verifier makes NO adjudication decisions — that logic lives in the reconciliation
prompt. It only guarantees the output is internally consistent and schema-valid.

Depends only on `schema` + the standard library.

Self-test:  python -m code.verifier
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from . import schema
except ImportError:  # pragma: no cover
    import schema  # type: ignore


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _bool_str(v: bool) -> str:
    return "true" if v else "false"


def _image_ids(image_paths: str) -> set[str]:
    """Real image ids (filename stems) referenced by the claim."""
    return {
        Path(p.strip()).stem
        for p in str(image_paths or "").split(";")
        if p.strip()
    }


def _parse_history_flags(history_flags: str) -> set[str]:
    """Tokenize the user_history.csv history_flags string (';'/',' separated)."""
    return {
        t.strip()
        for t in str(history_flags or "").replace(",", ";").split(";")
        if t.strip() and t.strip() != schema.NONE_TOKEN
    }


def verify(input_fields: dict, derived: dict, history_flags: str = "") -> dict:
    """Enforce schema invariants on the model output → 14-column row.

    `input_fields`: the 4 verbatim inputs (user_id, image_paths, user_claim,
    claim_object). `derived`: the model's 10 fields (a `confidence` key, if
    present, is ignored here). `history_flags`: the user's history_flags from
    user_history.csv — used to derive user_history_risk / manual_review_required
    deterministically rather than trusting the model.
    """
    claim_object = str(input_fields.get("claim_object", "")).strip()

    # 1. claim_status — must be a valid enum, else NEI.
    claim_status = str(derived.get("claim_status", "")).strip()
    if claim_status not in schema.CLAIM_STATUS:
        claim_status = schema.NEI

    # 2. object_part — the CLAIMED part the prompt chose, range-checked.
    object_part = schema.coerce_object_part(
        claim_object, str(derived.get("object_part", "")).strip()
    )

    # 3. issue_type / severity — enum or fallback.
    issue_type = str(derived.get("issue_type", "")).strip()
    if issue_type not in schema.ISSUE_TYPE:
        issue_type = "unknown"
    severity = str(derived.get("severity", "")).strip()
    if severity not in schema.SEVERITY:
        severity = "unknown"

    # 4. risk_flags — parsed/validated into a working set; history flags parsed.
    flags = set(schema.parse_risk_flags(derived.get("risk_flags", "")))
    hist = _parse_history_flags(history_flags)

    # 5. supporting_image_ids — keep only ids that actually exist in this claim.
    real_ids = _image_ids(input_fields.get("image_paths", ""))
    supporting = [
        s.strip()
        for s in str(derived.get("supporting_image_ids", "")).split(";")
        if s.strip() and s.strip() != schema.NONE_TOKEN and s.strip() in real_ids
    ]
    supporting_str = ";".join(supporting) if supporting else schema.NONE_TOKEN

    valid_image = _truthy(derived.get("valid_image"))

    # 6. Invariants -------------------------------------------------------
    # (a) no supporting image cannot be a "supported" verdict.
    if supporting_str == schema.NONE_TOKEN and claim_status == "supported":
        claim_status = schema.NEI

    # (b) not_enough_information — two sub-types, gated on wrong_object:
    #     - part-not-visible (wrong_object ABSENT): the claimed part isn't shown,
    #       so force supporting_image_ids='none' and issue_type='unknown' (strict).
    #     - identity-mismatch (wrong_object PRESENT, e.g. sample case_002): the
    #       wrong object IS shown; keep the model's supporting_image_ids and
    #       issue_type (what it actually saw).
    #     Both sub-types still force evidence_standard_met=False and severity='unknown'.
    if claim_status == schema.NEI:
        severity = "unknown"
        if "wrong_object" not in flags:
            supporting_str = schema.NONE_TOKEN
            issue_type = "unknown"

    # (c) evidence_standard_met is fully determined by claim_status: you cannot
    #     support or contradict a claim without assessing the part, so
    #     evidence_standard_met is true iff the verdict is NOT
    #     not_enough_information. This overrides the model.
    evidence_met = claim_status != schema.NEI

    # (d) manipulation / non-original cues invalidate the image set.
    if flags & schema.INVALIDATING_FLAGS:
        valid_image = False

    # (e) user_history_risk is DERIVED from the user's history_flags, overriding
    #     the model (which false-positives this signal on users whose history
    #     carries only manual_review_required).
    flags.discard("user_history_risk")
    if "user_history_risk" in hist:
        flags.add("user_history_risk")

    # (f) manual_review_required is fully DERIVED (biconditional): present iff a
    #     trust/identity trigger is in the flags OR the user's history_flags
    #     already carries manual_review_required — overriding whatever the model
    #     emitted (this neutralizes the broader trigger list still described in
    #     the reconcile prompt).
    #     Known noisy label we intentionally do NOT match: one sample carries
    #     cropped_or_obstructed + damage_not_visible + manual_review_required with
    #     no trust signal and no history flag — those are image-quality flags, so
    #     we leave manual_review_required off.
    flags.discard("manual_review_required")
    if (flags & schema.MANUAL_REVIEW_TRIGGERS) or ("manual_review_required" in hist):
        flags.add("manual_review_required")

    # 7. Assemble the ordered 14-column row (CSV-ready strings).
    derived_out = {
        "evidence_standard_met": _bool_str(evidence_met),
        "evidence_standard_met_reason": str(
            derived.get("evidence_standard_met_reason", "")
        ).strip(),
        "risk_flags": schema.format_risk_flags(flags),
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": str(
            derived.get("claim_status_justification", "")
        ).strip(),
        "supporting_image_ids": supporting_str,
        "valid_image": _bool_str(valid_image),
        "severity": severity,
    }
    row = {}
    for col in schema.INPUT_COLUMNS:
        row[col] = str(input_fields.get(col, ""))
    row.update(derived_out)
    # Reassert ordering over the full 14-column contract.
    return {col: row[col] for col in schema.OUTPUT_COLUMNS}


# ---------------------------------------------------------------------------
# Self-test:  python -m code.verifier
# ---------------------------------------------------------------------------
def _base_inputs() -> dict:
    return {
        "user_id": "user_x",
        "image_paths": "images/test/case_x/img_1.jpg;images/test/case_x/img_2.jpg",
        "user_claim": "Customer: the bumper looks bad.",
        "claim_object": "car",
    }


def _self_test() -> None:
    inp = _base_inputs()

    # 1. history user_history_risk -> sets the flag AND manual_review_required.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "ok",
        "risk_flags": "none", "issue_type": "dent",
        "object_part": "rear_bumper", "claim_status": "supported",
        "claim_status_justification": "dent visible on rear bumper",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "medium",
        "confidence": 0.9,
    }, history_flags="user_history_risk")
    assert "user_history_risk" in r["risk_flags"], r["risk_flags"]
    assert "manual_review_required" in r["risk_flags"], r["risk_flags"]
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  history user_history_risk -> user_history_risk + manual_review ADDED")

    # 2. claim_mismatch only (no trust signal) -> manual_review_required REMOVED
    #    even though the model emitted it.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "ok",
        "risk_flags": "claim_mismatch;manual_review_required", "issue_type": "scratch",
        "object_part": "rear_bumper", "claim_status": "contradicted",
        "claim_status_justification": "only minor scratch; severe claim contradicted",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "low",
        "confidence": 0.8,
    })
    assert r["risk_flags"] == "claim_mismatch", r["risk_flags"]
    assert "manual_review_required" not in r["risk_flags"]
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  claim_mismatch only -> manual_review_required REMOVED")

    # 3. model emits manual_review_required with NO trigger -> REMOVED.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "ok",
        "risk_flags": "manual_review_required", "issue_type": "dent",
        "object_part": "rear_bumper", "claim_status": "supported",
        "claim_status_justification": "dent visible",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "medium",
        "confidence": 0.7,
    })
    assert r["risk_flags"] == "none", r["risk_flags"]
    print("  ok  manual_review_required with no trigger -> REMOVED")

    # 4. not_enough_information forces dependent fields; claimed part preserved.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "x",
        "risk_flags": "wrong_angle;damage_not_visible", "issue_type": "dent",
        "object_part": "headlight", "claim_status": "not_enough_information",
        "claim_status_justification": "claimed part not shown",
        "supporting_image_ids": "img_1;img_2", "valid_image": "true", "severity": "high",
        "confidence": 0.4,
    })
    assert r["claim_status"] == "not_enough_information"
    assert r["object_part"] == "headlight"        # CLAIMED part preserved
    assert r["supporting_image_ids"] == "none"
    assert r["issue_type"] == "unknown"
    assert r["severity"] == "unknown"
    assert r["evidence_standard_met"] == "false"
    assert "manual_review_required" not in r["risk_flags"]   # quality flags only
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  NEI forces supporting/issue/severity/evidence; claimed part kept")

    # 4b. NEI identity-mismatch (wrong_object PRESENT, case_002-shaped): keep the
    #     model's supporting_image_ids + issue_type; force evidence/severity;
    #     wrong_object drives manual_review_required.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "wrong object shown",
        "risk_flags": "wrong_object;claim_mismatch", "issue_type": "broken_part",
        "object_part": "front_bumper", "claim_status": "not_enough_information",
        "claim_status_justification": "images show a different object than the claimed car part",
        "supporting_image_ids": "img_1;img_2", "valid_image": "true", "severity": "high",
        "confidence": 0.5,
    })
    assert r["claim_status"] == "not_enough_information"
    assert r["supporting_image_ids"] == "img_1;img_2"   # KEPT (identity-mismatch)
    assert r["issue_type"] == "broken_part"             # KEPT (identity-mismatch)
    assert r["evidence_standard_met"] == "false"        # still forced
    assert r["severity"] == "unknown"                   # still forced
    assert "manual_review_required" in r["risk_flags"]  # wrong_object is a trigger
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  NEI identity-mismatch (wrong_object) keeps supporting/issue; "
          "forces evidence/severity; manual_review added")

    # 4c. evidence_standard_met is derived from claim_status: a contradicted
    #     verdict forces evidence_standard_met=true even if the model said false.
    r = verify(inp, {
        "evidence_standard_met": "false", "evidence_standard_met_reason": "x",
        "risk_flags": "claim_mismatch", "issue_type": "scratch",
        "object_part": "rear_bumper", "claim_status": "contradicted",
        "claim_status_justification": "only a minor scratch; severe claim contradicted",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "low",
        "confidence": 0.7,
    })
    assert r["evidence_standard_met"] == "true"   # contradicted ⇒ part was assessable
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  contradicted forces evidence_standard_met=true")

    # 4d. history override: model emits user_history_risk but history says 'none'
    #     -> user_history_risk REMOVED, and (no other trigger) manual_review REMOVED.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "ok",
        "risk_flags": "user_history_risk;manual_review_required", "issue_type": "dent",
        "object_part": "rear_bumper", "claim_status": "supported",
        "claim_status_justification": "dent visible",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "medium",
        "confidence": 0.8,
    }, history_flags="none")
    assert "user_history_risk" not in r["risk_flags"], r["risk_flags"]
    assert "manual_review_required" not in r["risk_flags"], r["risk_flags"]
    print("  ok  model user_history_risk false-positive REMOVED when history='none'")

    # 4e. history manual_review_required (users 032/041/042 shape): no trust flag
    #     and no user_history_risk -> manual_review ADDED via history;
    #     user_history_risk NOT added.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "ok",
        "risk_flags": "none", "issue_type": "dent",
        "object_part": "rear_bumper", "claim_status": "supported",
        "claim_status_justification": "dent visible",
        "supporting_image_ids": "img_1", "valid_image": "true", "severity": "medium",
        "confidence": 0.8,
    }, history_flags="manual_review_required")
    assert "manual_review_required" in r["risk_flags"], r["risk_flags"]
    assert "user_history_risk" not in r["risk_flags"], r["risk_flags"]
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  history manual_review_required -> manual_review ADDED (no user_history_risk)")

    # 5. invalidating flag forces valid_image=false; bad object_part -> unknown;
    #    supporting filtered to real ids.
    r = verify(inp, {
        "evidence_standard_met": "true", "evidence_standard_met_reason": "x",
        "risk_flags": "non_original_image", "issue_type": "dent",
        "object_part": "screen",                  # not valid for car
        "claim_status": "supported",
        "claim_status_justification": "looks like a screenshot",
        "supporting_image_ids": "img_1;img_9", "valid_image": "true", "severity": "high",
        "confidence": 0.3,
    })
    assert r["valid_image"] == "false"            # non_original_image invalidates
    assert "manual_review_required" in r["risk_flags"]   # trust trigger
    assert r["object_part"] == "unknown"          # screen invalid for car
    assert r["supporting_image_ids"] == "img_1"   # img_9 is not a real id
    assert list(schema.iter_invariant_violations(r)) == []
    print("  ok  invalidating flag -> valid_image=false; bad part/ids cleaned")

    # 6. column contract.
    assert list(r.keys()) == schema.OUTPUT_COLUMNS
    print("  ok  row has exactly the 14 columns in spec order")

    print("\nAll verifier self-tests passed.")


if __name__ == "__main__":
    _self_test()

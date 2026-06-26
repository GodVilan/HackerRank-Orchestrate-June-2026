"""
test_schema.py — Assert that code/schema.py matches the SPEC CARD verbatim and
that the validation helpers + the prompts wiring behave as designed.

Run:  python -m code.test_schema
"""
from __future__ import annotations

try:
    from . import schema
except ImportError:  # pragma: no cover
    import schema  # type: ignore


# ---------------------------------------------------------------------------
# Expected values — copied verbatim from the SPEC CARD (the test's own oracle).
# ---------------------------------------------------------------------------
EXPECTED_CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

EXPECTED_ISSUE_TYPE = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

EXPECTED_OBJECT_PART = {
    "car": ["front_bumper", "rear_bumper", "door", "hood", "windshield",
            "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
            "body", "unknown"],
    "laptop": ["screen", "keyboard", "trackpad", "hinge", "lid", "corner",
               "port", "base", "body", "unknown"],
    "package": ["box", "package_corner", "package_side", "seal", "label",
                "contents", "item", "unknown"],
}

EXPECTED_RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
]

EXPECTED_SEVERITY = ["none", "low", "medium", "high", "unknown"]


def test_constants_match_spec_card() -> None:
    assert schema.CLAIM_STATUS == EXPECTED_CLAIM_STATUS
    assert schema.ISSUE_TYPE == EXPECTED_ISSUE_TYPE
    assert schema.OBJECT_PART_BY_OBJECT == EXPECTED_OBJECT_PART
    assert schema.RISK_FLAGS == EXPECTED_RISK_FLAGS
    assert schema.SEVERITY == EXPECTED_SEVERITY


def test_object_part_structure() -> None:
    assert set(schema.OBJECT_PART_BY_OBJECT) == {"car", "laptop", "package"}
    for obj, parts in schema.OBJECT_PART_BY_OBJECT.items():
        assert parts[-1] == "unknown", f"{obj} parts must end with 'unknown'"


def test_derived_flag_sets() -> None:
    # Every derived-set member must itself be a valid risk flag.
    assert schema.INVALIDATING_FLAGS <= set(schema.RISK_FLAGS)
    assert schema.MANUAL_REVIEW_TRIGGERS <= set(schema.RISK_FLAGS)
    assert schema.INVALIDATING_FLAGS == {"possible_manipulation", "non_original_image"}
    # Trust/authenticity/adversarial/identity signals ONLY — NOT image-quality flags.
    assert schema.MANUAL_REVIEW_TRIGGERS == {
        "user_history_risk", "possible_manipulation", "non_original_image",
        "text_instruction_present", "wrong_object",
    }


def test_output_columns() -> None:
    assert schema.OUTPUT_COLUMNS == [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
        "issue_type", "object_part", "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity",
    ]
    assert len(schema.OUTPUT_COLUMNS) == 14
    assert schema.OUTPUT_COLUMNS == schema.INPUT_COLUMNS + schema.DERIVED_COLUMNS
    # The derived columns must match the labeled-output fields data_layer reads.
    try:
        from . import data_layer
    except ImportError:  # pragma: no cover
        import data_layer  # type: ignore
    assert list(schema.DERIVED_COLUMNS) == list(data_layer._LABELED_FIELDS)


def test_coerce_object_part() -> None:
    assert schema.coerce_object_part("car", "screen") == "unknown"      # wrong object
    assert schema.coerce_object_part("car", "rear_bumper") == "rear_bumper"
    assert schema.coerce_object_part("laptop", "screen") == "screen"
    assert schema.coerce_object_part("nonsense", "anything") == "unknown"
    assert schema.allowed_parts("package")[-1] == "unknown"


def test_risk_flag_roundtrip() -> None:
    assert schema.parse_risk_flags("none") == []
    assert schema.parse_risk_flags("") == []
    # de-dup + SPEC ordering, drop unknown tokens
    parsed = schema.parse_risk_flags("claim_mismatch;blurry_image;claim_mismatch;made_up")
    assert parsed == ["blurry_image", "claim_mismatch"]
    assert schema.format_risk_flags([]) == "none"
    assert schema.format_risk_flags(["claim_mismatch", "blurry_image"]) == \
        "blurry_image;claim_mismatch"
    # round-trip
    s = "blurry_image;claim_mismatch;manual_review_required"
    assert schema.format_risk_flags(schema.parse_risk_flags(s)) == s


def test_invariant_checker_flags_broken_row() -> None:
    # NEI but with supporting ids + a real issue/severity + evidence met → 4 violations
    bad = {
        "claim_object": "car", "object_part": "rear_bumper",
        "claim_status": "not_enough_information", "supporting_image_ids": "img_1",
        "issue_type": "dent", "severity": "medium",
        "evidence_standard_met": "true", "valid_image": "true", "risk_flags": "none",
    }
    msgs = list(schema.iter_invariant_violations(bad))
    assert len(msgs) == 4, msgs

    # manipulation flag without valid_image=false AND without manual_review_required
    bad2 = {
        "claim_object": "laptop", "object_part": "screen",
        "claim_status": "supported", "supporting_image_ids": "img_1",
        "issue_type": "glass_shatter", "severity": "high",
        "evidence_standard_met": "true", "valid_image": "true",
        "risk_flags": "possible_manipulation",
    }
    msgs2 = list(schema.iter_invariant_violations(bad2))
    assert any("valid_image=false" in m for m in msgs2)
    assert any("manual_review_required" in m for m in msgs2)

    # wrong object_part for the object
    bad3 = {
        "claim_object": "car", "object_part": "keyboard",
        "claim_status": "supported", "supporting_image_ids": "img_1",
        "issue_type": "dent", "severity": "low",
        "evidence_standard_met": "true", "valid_image": "true", "risk_flags": "none",
    }
    assert any("not valid for" in m for m in schema.iter_invariant_violations(bad3))


def test_invariant_checker_clean_row() -> None:
    clean = {
        "claim_object": "car", "object_part": "rear_bumper",
        "claim_status": "supported", "supporting_image_ids": "img_1",
        "issue_type": "dent", "severity": "medium",
        "evidence_standard_met": "true", "valid_image": "true", "risk_flags": "none",
    }
    assert list(schema.iter_invariant_violations(clean)) == []

    clean_nei = {
        "claim_object": "package", "object_part": "box",
        "claim_status": "not_enough_information", "supporting_image_ids": "none",
        "issue_type": "unknown", "severity": "unknown",
        "evidence_standard_met": "false", "valid_image": "true", "risk_flags": "none",
    }
    assert list(schema.iter_invariant_violations(clean_nei)) == []

    # Identity-mismatch NEI (wrong_object present, case_002-shaped): keeping
    # supporting_image_ids + issue_type is VALID — not flagged as a violation.
    identity_nei = {
        "claim_object": "car", "object_part": "front_bumper",
        "claim_status": "not_enough_information", "supporting_image_ids": "img_1;img_2",
        "issue_type": "broken_part", "severity": "unknown",
        "evidence_standard_met": "false", "valid_image": "true",
        "risk_flags": "wrong_object;claim_mismatch;manual_review_required",
    }
    assert list(schema.iter_invariant_violations(identity_nei)) == []


def test_claim_mismatch_alone_no_manual_review() -> None:
    # claim_mismatch is NOT a manual-review trigger anymore: a contradicted row
    # carrying only claim_mismatch (no trust signal) and no manual_review_required
    # is now internally consistent per the checker.
    row = {
        "claim_object": "car", "object_part": "rear_bumper",
        "claim_status": "contradicted", "supporting_image_ids": "img_1",
        "issue_type": "scratch", "severity": "low",
        "evidence_standard_met": "true", "valid_image": "true",
        "risk_flags": "claim_mismatch",
    }
    assert list(schema.iter_invariant_violations(row)) == []


def test_supported_contradicted_require_evidence_met() -> None:
    # A supported/contradicted verdict with evidence_standard_met=false is
    # internally inconsistent (you can't assess a verdict without meeting the
    # evidence standard) — the checker must flag it.
    for status in ("supported", "contradicted"):
        bad = {
            "claim_object": "car", "object_part": "rear_bumper",
            "claim_status": status, "supporting_image_ids": "img_1",
            "issue_type": "dent", "severity": "medium",
            "evidence_standard_met": "false", "valid_image": "true",
            "risk_flags": "none",
        }
        msgs = list(schema.iter_invariant_violations(bad))
        assert any("evidence_standard_met=true" in m for m in msgs), (status, msgs)
        # The same row with evidence_standard_met=true is consistent.
        bad["evidence_standard_met"] = "true"
        assert list(schema.iter_invariant_violations(bad)) == []


def test_prompts_import_from_schema() -> None:
    # prompts.py must import the SAME list objects (identity, not a copy).
    try:
        from . import prompts
    except ImportError:  # pragma: no cover
        import prompts  # type: ignore
    assert prompts.ISSUE_TYPE is schema.ISSUE_TYPE
    assert prompts.SEVERITY is schema.SEVERITY
    assert prompts.RISK_FLAGS is schema.RISK_FLAGS
    assert prompts.OBJECT_PART_BY_OBJECT is schema.OBJECT_PART_BY_OBJECT


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nAll {len(tests)} schema tests passed.")


if __name__ == "__main__":
    _run_all()

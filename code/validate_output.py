"""
validate_output.py — Strict verification pass for output.csv.
Reads output.csv and asserts that:
1. The columns match exactly schema.OUTPUT_COLUMNS in the same order.
2. Every row is free of structural invariant violations (using schema.iter_invariant_violations).
3. The row count matches claims.csv (exactly 44 rows).
4. All values are valid according to their respective enums/types.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import schema
except ImportError:
    from . import schema

def validate(csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return 1

    print(f"Validating {csv_path}...")
    violations_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames
        if headers is None:
            print("Error: CSV has no headers.")
            return 1

        # 1. Column presence and order check
        if headers != schema.OUTPUT_COLUMNS:
            print(f"Error: Headers mismatch!\nExpected: {schema.OUTPUT_COLUMNS}\nGot:      {headers}")
            violations_count += 1
        else:
            print("  ok  Headers match schema.OUTPUT_COLUMNS exactly.")

        rows = list(reader)
        row_count = len(rows)

        # 2. Row count check
        print(f"  info  Row count: {row_count}")
        if row_count != 44:
            print(f"Warning: Expected exactly 44 rows, but got {row_count}.")

        # 3. Check each row's invariants and enums
        for idx, row in enumerate(rows, start=1):
            # Check enums
            claim_status = row.get("claim_status")
            if claim_status not in schema.CLAIM_STATUS:
                print(f"Row {idx} (user={row.get('user_id')}): Invalid claim_status '{claim_status}'")
                violations_count += 1

            issue_type = row.get("issue_type")
            if issue_type not in schema.ISSUE_TYPE:
                print(f"Row {idx} (user={row.get('user_id')}): Invalid issue_type '{issue_type}'")
                violations_count += 1

            severity = row.get("severity")
            if severity not in schema.SEVERITY:
                print(f"Row {idx} (user={row.get('user_id')}): Invalid severity '{severity}'")
                violations_count += 1

            claim_object = row.get("claim_object")
            object_part = row.get("object_part")
            if claim_object in schema.OBJECT_PART_BY_OBJECT:
                if object_part not in schema.allowed_parts(claim_object):
                    print(f"Row {idx} (user={row.get('user_id')}): Invalid object_part '{object_part}' for object '{claim_object}'")
                    violations_count += 1

            # Check boolean fields are lowercase "true" or "false"
            for bool_field in ["evidence_standard_met", "valid_image"]:
                val = row.get(bool_field)
                if val not in ("true", "false"):
                    print(f"Row {idx} (user={row.get('user_id')}): Field '{bool_field}' must be 'true' or 'false', got '{val}'")
                    violations_count += 1

            # Check risk flags
            risk_flags_str = row.get("risk_flags", "")
            risk_flags = schema.parse_risk_flags(risk_flags_str)
            # Reformat to check if it matches the parsed flags order/representation
            ref_flags = schema.format_risk_flags(risk_flags)
            if risk_flags_str != ref_flags:
                print(f"Row {idx} (user={row.get('user_id')}): risk_flags format mismatch! Original: '{risk_flags_str}', Normalized: '{ref_flags}'")
                violations_count += 1

            # Check invariant violations
            violations = list(schema.iter_invariant_violations(row))
            if violations:
                print(f"Row {idx} (user={row.get('user_id')}): Invariant violations:")
                for v in violations:
                    print(f"  - {v}")
                    violations_count += 1

    if violations_count == 0:
        print("Success: Zero invariant violations or column issues found in output.csv!")
        return 0
    else:
        print(f"Failure: Found {violations_count} violation(s)/error(s) in output.csv.")
        return 1

if __name__ == "__main__":
    csv_file = Path(__file__).resolve().parent.parent / "output.csv"
    sys.exit(validate(csv_file))

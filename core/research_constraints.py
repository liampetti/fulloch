"""Deterministic checks for simple user constraints; unknown is never success."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class ConstraintResult:
    status: str
    reason: str
    actual: object = None


_NUMBER_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z%°]+)?\s*$")


def _decimal(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _calendar_date(value):
    text = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return date.fromisoformat(text)
    timestamp = datetime.fromisoformat(text)
    if timestamp.utcoffset() is None:
        raise ValueError("Timestamp has no timezone")
    # Calendar-date constraints refer to the stated local date, not UTC.
    return timestamp.date()


def validate_constraint(constraint: dict, record: dict) -> ConstraintResult:
    """Evaluate supported equality/range/date constraints against explicit fields.

    Adapters may validate domain-specific semantics themselves. This helper
    intentionally returns unknown for absent fields, incompatible units,
    timezone ambiguity, and unsupported operators.
    """
    if not isinstance(constraint, dict) or not isinstance(record, dict):
        return ConstraintResult("unknown", "Constraint or record is not structured.")
    field = constraint.get("field")
    if not isinstance(field, str) or not field.strip() or field not in record:
        return ConstraintResult("unknown", "Required field is missing.")
    actual = record[field]
    operator = constraint.get("operator", "eq")
    if not isinstance(operator, str):
        return ConstraintResult("unknown", "Operator must be a string.", actual)
    expected = constraint.get("value")
    if actual is None or (expected is None and operator != "range"):
        return ConstraintResult("unknown", "Constraint value is missing.", actual)
    if operator in {"eq", "equals"}:
        if isinstance(actual, str) and isinstance(expected, str):
            ok = actual.strip().casefold() == expected.strip().casefold()
        else:
            ok = actual == expected
        return ConstraintResult("satisfied" if ok else "violated", "Equality comparison.", actual)
    if operator in {"date_eq", "date_before", "date_after"}:
        try:
            actual_date = _calendar_date(actual)
            expected_date = _calendar_date(expected)
        except ValueError:
            return ConstraintResult("unknown", "Date is not an unambiguous ISO calendar date.", actual)
        ok = {
            "date_eq": actual_date == expected_date,
            "date_before": actual_date < expected_date,
            "date_after": actual_date > expected_date,
        }[operator]
        return ConstraintResult("satisfied" if ok else "violated", "Calendar-date comparison.", actual)
    if operator in {"min", "max", "range"}:
        actual_match = _NUMBER_UNIT.fullmatch(str(actual))
        if not actual_match:
            return ConstraintResult("unknown", "Actual value is not a number with an optional unit.", actual)
        actual_number, actual_unit = Decimal(actual_match[1]), (actual_match[2] or "").casefold()
        expected_unit = str(constraint.get("unit") or "").casefold()
        if actual_unit != expected_unit:
            return ConstraintResult("unknown", "Units do not match; no implicit conversion was applied.", actual)
        low, high = constraint.get("min"), constraint.get("max")
        if operator == "min":
            low = expected
        elif operator == "max":
            high = expected
        low_number, high_number = (_decimal(low), _decimal(high))
        if (low is not None and low_number is None) or (high is not None and high_number is None):
            return ConstraintResult("unknown", "Range bound is invalid.", actual)
        if low_number is None and high_number is None:
            return ConstraintResult("unknown", "Range has no bounds.", actual)
        if low_number is not None and high_number is not None and low_number > high_number:
            return ConstraintResult("unknown", "Range bounds are reversed.", actual)
        ok = (low_number is None or actual_number >= low_number) and (
            high_number is None or actual_number <= high_number
        )
        return ConstraintResult("satisfied" if ok else "violated", "Numeric range comparison.", actual)
    return ConstraintResult("unknown", f"Unsupported operator: {operator!r}.", actual)


def validate_constraint_records(constraint: dict, records: list, *, validator=None) -> dict:
    """Validate one typed constraint against a bounded set of explicit records.

    Returns a deterministic summary: ``violated`` when any record fails a
    comparable check, ``satisfied`` when at least one record passes and none
    fail, otherwise ``unknown``. ``unknown`` is always retained as-is — it is
    never converted into satisfaction. Domain-specific adapters may extend the
    per-record checks, but they must return ConstraintResult objects so unknown
    evidence cannot be mistaken for a pass.
    """
    records = records if isinstance(records, list) else []
    check = validator or validate_constraint
    results = []
    for record in records[:32]:
        if not isinstance(record, dict):
            continue
        try:
            result = check(constraint, record)
        except Exception:
            result = ConstraintResult("unknown", "Constraint adapter could not evaluate this record.")
        if not isinstance(result, ConstraintResult) or result.status not in {"satisfied", "violated", "unknown"}:
            result = ConstraintResult("unknown", "Constraint adapter returned an invalid result.")
        results.append(result)
    statuses = {result.status for result in results}
    if "violated" in statuses:
        first_violation = next(result for result in results if result.status == "violated")
        return {"status": "violated", "reason": first_violation.reason,
                "checked": len(results), "actual": first_violation.actual}
    if "satisfied" in statuses:
        if len(records) > 32:
            return {"status": "unknown", "reason": "Additional records exceed the bounded check set.",
                    "checked": len(results)}
        first_satisfaction = next(result for result in results if result.status == "satisfied")
        return {"status": "satisfied", "reason": first_satisfaction.reason,
                "checked": len(results), "actual": first_satisfaction.actual}
    if results:
        return {"status": "unknown", "reason": results[0].reason, "checked": len(results)}
    return {"status": "unknown", "reason": "No structured records available for this constraint.",
            "checked": 0}

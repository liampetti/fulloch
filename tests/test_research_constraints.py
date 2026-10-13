import pytest

from core.research_constraints import validate_constraint, validate_constraint_records


def test_constraint_evaluator_preserves_missing_incompatible_and_supported_states():
    assert validate_constraint({"field": "price", "operator": "max", "value": 20, "unit": "USD"},
                              {"price": "19.99 USD"}).status == "satisfied"
    assert validate_constraint({"field": "price", "operator": "range", "min": 10, "max": 20,
                                "unit": "USD"}, {"price": "21 USD"}).status == "violated"
    assert validate_constraint({"field": "price", "operator": "min", "value": 10, "unit": "USD"},
                              {"price": "12 EUR"}).status == "unknown"
    assert validate_constraint({"field": "departure", "operator": "date_eq", "value": "2027-01-17"},
                              {"departure": "2027-01-17T09:30:00+13:00"}).status == "satisfied"
    assert validate_constraint({"field": "departure", "operator": "date_eq", "value": "2027-01-17"},
                              {"departure": "tomorrow"}).status == "unknown"
    assert validate_constraint({"field": "currency", "value": "USD"}, {}).status == "unknown"


@pytest.mark.parametrize("bounds", [
    {}, {"min": "NaN"}, {"max": "Infinity"}, {"min": "-Infinity"},
    {"min": 20, "max": 10}, {"min": True},
])
def test_invalid_ranges_never_succeed_or_raise(bounds):
    result = validate_constraint({"field": "size", "operator": "range", **bounds}, {"size": 15})
    assert result.status == "unknown"


@pytest.mark.parametrize("value", ["2027-01-17garbage", "2027-01-17T99:00:00Z", "2027-01-17T09:00:00"])
def test_dates_validate_entire_value_and_require_timestamp_timezone(value):
    assert validate_constraint({"field": "date", "operator": "date_eq", "value": "2027-01-17"},
                               {"date": value}).status == "unknown"


def test_numeric_units_require_both_sides_and_malformed_operator_is_unknown():
    assert validate_constraint({"field": "size", "operator": "max", "value": 20},
                               {"size": "15 kg"}).status == "unknown"
    assert validate_constraint({"field": "size", "operator": [], "value": 20},
                               {"size": 15}).status == "unknown"


def test_one_sided_ranges_and_local_calendar_date_are_supported():
    assert validate_constraint({"field": "size", "operator": "range", "min": 10},
                               {"size": 15}).status == "satisfied"
    assert validate_constraint({"field": "date", "operator": "date_eq", "value": "2027-01-17"},
                               {"date": "2027-01-17T00:30:00+13:00"}).status == "satisfied"


def test_constraint_records_keep_unknown_and_let_violations_win():
    constraint = {"field": "price", "operator": "max", "value": 20, "unit": "USD"}

    assert validate_constraint_records(constraint, []) == {
        "status": "unknown",
        "reason": "No structured records available for this constraint.",
        "checked": 0,
    }
    # A single comparable passing record is satisfaction; unknowns never count
    # as a pass even when a sibling record passes.
    assert validate_constraint_records(
        constraint, [{"price": "19.99 USD"}, {"price": "25 USD"}]) == {
        "status": "violated", "reason": "Numeric range comparison.",
        "checked": 2, "actual": "25 USD",
    }
    assert validate_constraint_records(
        constraint, [{"price": "19.99 USD"}, {"price": "no unit"}])["status"] == "satisfied"
    assert validate_constraint_records(constraint, [{"price": "no unit"}])["status"] == "unknown"
    # The first violation's actual value is reported, not a later record's.
    assert validate_constraint_records(
        constraint, [{"price": "21 USD"}, {"price": "22 USD"}])["actual"] == "21 USD"
    # Non-dict records are ignored, and the check set is bounded.
    assert validate_constraint_records(constraint, [None, 7, {"price": "19.99 USD"}])["status"] == "satisfied"
    many = [{"price": f"{10 + index} USD"} for index in range(40)]
    result = validate_constraint_records(constraint, many)
    assert result["status"] == "violated"
    assert result["checked"] == 32

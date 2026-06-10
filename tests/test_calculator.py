"""Tests for tools/calculator.py — the local calculator / date / unit tools.

These run pure Python (simpleeval / pint / dateutil), no models, so no
stubbing is needed. Covers arithmetic + sandbox safety, speech-friendly
number formatting across magnitudes, date maths, and unit conversion.
"""

from tools.calculator import (
    calculate,
    convert_units,
    date_of,
    days_between,
    _fmt_number,
)


class TestCalculate:
    def test_basic_arithmetic(self):
        assert calculate("3 * 8 + 7") == "That comes to 31."

    def test_percentage_phrasing(self):
        assert calculate("15% of 340") == "That comes to 51."

    def test_function_and_rounding(self):
        # sqrt(2) rounds to 2 dp.
        assert calculate("sqrt(2)") == "That comes to 1.41."

    def test_divide_by_zero_is_friendly(self):
        assert "dividing by zero" in calculate("1/0")

    def test_sandbox_blocks_code(self):
        # simpleeval must not execute attribute/name access into Python.
        out = calculate("__import__('os').system('echo hi')")
        assert out.startswith("Reactive question:")

    def test_empty_expression_asks(self):
        assert calculate("   ").startswith("Reactive question:")


class TestFormatNumber:
    def test_integer_stays_whole(self):
        assert _fmt_number(51) == "51"
        assert _fmt_number(100.0) == "100"

    def test_two_decimal_places(self):
        assert _fmt_number(8.0467) == "8.05"
        assert _fmt_number(1 / 3) == "0.33"

    def test_scale_words_for_large(self):
        assert _fmt_number(1_234_567) == "1.23 million"
        assert _fmt_number(1_234_567_000) == "1.23 billion"
        assert _fmt_number(-1_234_567) == "-1.23 million"

    def test_scientific_for_tiny(self):
        assert _fmt_number(0.00012) == "1.2 times ten to the power of minus 4"

    def test_scientific_for_huge(self):
        assert "times ten to the power of 15" in _fmt_number(1.0995e15)

    def test_small_decimal_keeps_significance(self):
        # 2 dp would round this to zero; sig-figs keep it.
        assert _fmt_number(0.0033) == "0.0033"

    def test_zero(self):
        assert _fmt_number(0) == "0"


class TestDays:
    def test_days_between_exact(self):
        assert days_between("2026-01-01", "2026-01-11") == (
            "There are 10 days between those two dates."
        )

    def test_same_day(self):
        assert days_between("2026-06-10", "2026-06-10") == "Those are the same day."

    def test_singular_day(self):
        assert "1 day between" in days_between("2026-06-10", "2026-06-11")

    def test_bad_date_asks(self):
        assert days_between("not-a-date", "2026-06-10").startswith("Reactive question:")


class TestDateOf:
    def test_resolves_to_correct_weekday(self):
        # The whole point of the tool: the resolved date must be that weekday.
        assert date_of("Sunday", "next").startswith("That falls on Sunday ")
        assert date_of("thursday", "this").startswith("That falls on Thursday ")

    def test_unknown_weekday_asks(self):
        assert date_of("blursday").startswith("Reactive question:")


class TestConvertUnits:
    def test_length(self):
        assert convert_units(5, "miles", "kilometres") == "5 miles is 8.05 kilometres."

    def test_temperature_offset(self):
        assert convert_units(20, "celsius", "fahrenheit") == "20 celsius is 68 fahrenheit."

    def test_imperial_gallon_default(self):
        # UK gallon ≈ 4.546 L → 3 gal ≈ 13.64 L.
        assert convert_units(3, "gallons", "litres") == "3 gallons is 13.64 litres."

    def test_us_gallon_opt_in(self):
        # US gallon ≈ 3.785 L → 3 gal ≈ 11.36 L.
        assert convert_units(3, "US gallons", "litres") == "3 US gallons is 11.36 litres."

    def test_dimensionality_mismatch_asks(self):
        assert convert_units(5, "kg", "miles").startswith("Reactive question:")

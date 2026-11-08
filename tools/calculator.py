"""Built-in calculator, date-resolver, and unit-conversion tools.

Smaller quantised models (9B and below) are unreliable at arithmetic and
calendar maths — they pattern-match rather than count, so "what date is next
Sunday?" produces off-by-one errors and "15% of 340" can come out wrong. These
tools offload the computation to Python so it is always correct, with zero
latency and no network call — everything runs on-device:

  - arithmetic via `simpleeval` (a sandboxed expression evaluator; never `eval`)
  - unit conversion via `pint`'s unit registry
  - date maths via `python-dateutil`
"""

import ast
import logging
import math
import re
from datetime import date, datetime, timedelta

import pint
import simpleeval
from dateutil import parser as _dateparser
from dateutil.relativedelta import (
    FR,
    MO,
    SA,
    SU,
    TH,
    TU,
    WE,
    relativedelta,
)

import utils.local_time as _local_time
from core.satellite_context import current_response_language
from utils.locale import household_locale
from utils.response_language import response

from .thinking_playbooks import thinking_playbook
from .tool_registry import tool

logger = logging.getLogger(__name__)

thinking_playbook(
    name="calculation",
    triggers=(r"\b(calculate|convert|percentage|how many|how much|difference between)\b",),
    capabilities=("calculate", "convert_units", "days_between"),
)


# Scale words make big results speakable: "1.23 million" beats a 7-digit run.
_SCALE_WORDS = ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"))


def _trim2(v: float) -> str:
    """Round to 2 dp and drop trailing zeros: 8.0 -> '8', 8.05 -> '8.05'."""
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def _sig2(x: float) -> str:
    """Two significant figures, fixed-point — for small sub-0.01 magnitudes
    that 2 dp would round away to zero (0.0033 stays '0.0033')."""
    d = 1 - int(math.floor(math.log10(abs(x))))
    return f"{round(x, d):.{max(d, 0)}f}".rstrip("0").rstrip(".") or "0"


def _spoken_decimal(value: str) -> str:
    return value.replace(".", ",") if current_response_language.get() in {"fr", "de"} else value


def _scientific(x: float) -> str:
    """Spoken scientific notation: 1.2e-5 -> '1.2 times ten to the power of minus 5'."""
    mant, exp = f"{x:.2e}".split("e")
    e = int(exp)
    language = current_response_language.get()
    sign = ("moins " if language == "fr" else "minus ") if e < 0 else ""
    return response("calc_scientific", mantissa=_spoken_decimal(_trim2(float(mant))), exponent=f"{sign}{abs(e)}")


def _fmt_number(x) -> str:
    """Format a numeric result for natural speech.

    Rounds to 2 dp in the everyday range, switches to scale words in the
    millions–trillions, and to spoken scientific notation at the extremes so
    very large numbers aren't read as a digit wall and very small ones aren't
    rounded down to zero.
    """
    if isinstance(x, bool):
        return str(x)
    try:
        n = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(n):
        return response("calc_undefined")
    if math.isinf(n):
        return response("calc_infinity" if n > 0 else "calc_negative_infinity")
    if n == 0:
        return "0"

    mag = abs(n)
    if mag >= 1e15:
        return _scientific(n)
    if mag >= 1e6:
        for scale, name in _SCALE_WORDS:
            if mag >= scale:
                value = _trim2(n / scale)
                singular = "_one" if abs(float(value)) == 1 else ""
                return response(f"calc_{name}{singular}", value=_spoken_decimal(value))
    if mag < 0.001:
        return _scientific(n)
    if mag < 0.01:
        return _spoken_decimal(_sig2(n))
    return _spoken_decimal(_trim2(n))


# --------------------------------------------------------------------------
# Arithmetic — sandboxed expression evaluation via simpleeval
# --------------------------------------------------------------------------


def _safe_factorial(n):
    # simpleeval caps `**` itself, but factorial is unguarded — cap the input.
    if n > 170:
        raise ValueError("factorial too large")
    return math.factorial(int(n))


_FUNCS = dict(simpleeval.DEFAULT_FUNCTIONS)
_FUNCS.update(
    {
        "sqrt": math.sqrt,
        "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "ln": math.log,
        "exp": math.exp,
        "factorial": _safe_factorial,
        "gcd": math.gcd,
        "min": min,
        "max": max,
        "pow": pow,
        "degrees": math.degrees,
        "radians": math.radians,
        "hypot": math.hypot,
    }
)
_NAMES = dict(simpleeval.DEFAULT_NAMES)
_NAMES.update({"pi": math.pi, "e": math.e, "tau": math.tau})


_NESTED_DATE_TOOL = re.compile(r"\bdays_between\b", re.IGNORECASE)
_EXPR_TOKENS = re.compile(
    r"(?P<quoted>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    r"|(?P<grouped>(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\d,]))"
    r"|(?P<name>[a-zA-Z_]\w*)|(?P<open>\()|(?P<close>\))|(?P<other>.)",
    re.DOTALL,
)


def parse_nested_days_between(expression: str) -> tuple[str, str] | None:
    """Recognize only a complete date-tool call with two literal safe dates."""
    if not isinstance(expression, str) or len(expression) > 256:
        return None
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except (ValueError, SyntaxError, RecursionError):
        return None
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != "days_between" or node.keywords or len(node.args) != 2):
        return None
    values = []
    for arg in node.args:
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            return None
        value = arg.value.strip()
        if value.lower() not in {"today", "tomorrow", "yesterday"}:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value, flags=re.ASCII):
                return None
            try:
                date.fromisoformat(value)
            except ValueError:
                return None
        values.append(value)
    return values[0], values[1]


def _strip_thousands_separators(text: str) -> str:
    """Keep function argument commas; only unambiguous outside-call groups are stripped."""
    result = []
    parens = []
    last_token = None
    for match in _EXPR_TOKENS.finditer(text):
        kind, value = match.lastgroup, match.group()
        if kind == "open":
            parens.append(last_token == "name" or any(parens))
        elif kind == "close" and parens:
            parens.pop()
        if kind == "grouped" and not any(parens):
            value = value.replace(",", "")
        result.append(value)
        if not value.isspace():
            last_token = kind
    return "".join(result)


def _normalize_expr(text: str) -> str:
    """Lightly rewrite natural phrasing into a plain math expression.

    Handles "15% of 340", trailing "%", "^" → "**", thousands separators,
    currency symbols, and a few spelled-out operators. Anything it doesn't
    recognise is left for simpleeval to reject.
    """
    s = (text or "").strip().lower().rstrip("?.")
    s = re.sub(r"^(what(?:'s| is)?|whats|calculate|compute|evaluate|how much is)\s+", "", s)
    s = re.sub(r"[£$€]", "", s)
    s = _strip_thousands_separators(s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s+", r"(\1/100)*", s)  # "15% of 340"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)  # standalone "n%"
    s = s.replace("^", "**")
    s = re.sub(r"\bplus\b", "+", s)
    s = re.sub(r"\bminus\b", "-", s)
    s = re.sub(r"\b(?:times|multiplied by)\b", "*", s)
    s = re.sub(r"\bdivided by\b", "/", s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", s)  # "3 x 8"
    return s.strip()


@tool(
    name="calculate",
    description=(
        "Evaluate a maths expression and speak the result. Handles sums, "
        "percentages, powers and roots; supports + - * / ** %, parentheses "
        "and functions like sqrt, abs, round, log, sin, cos. Also resolves "
        "relative calendar dates and ranges such as next week, next month, "
        "next year, or in three days."
    ),
    aliases=["calc", "math", "evaluate_expression"],
)
def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression string."""
    if _NESTED_DATE_TOOL.search(expression or ""):
        return ("Reactive question: " + response("calc_bad_expression", expression=repr(expression))
                + " Call days_between directly with two date arguments; do not put tool calls inside calculate.")
    if calendar := _calendar_reference(expression):
        start, end = calendar
        if start == end:
            return response("calc_date", date=start.isoformat())
        return response("calc_date_range", start=start.isoformat(), end=end.isoformat())
    if offset := _iso_date_offset(expression):
        return response("calc_date", date=offset.isoformat())
    expr = _normalize_expr(expression)
    if not expr:
        return "Reactive question: " + response("calc_no_expression") + " Ask them what they'd like calculated."
    try:
        result = simpleeval.simple_eval(expr, functions=_FUNCS, names=_NAMES)
    except ZeroDivisionError:
        return response("calculation_divide_by_zero")
    except Exception as e:
        logger.debug(f"calculate failed for {expression!r} (normalised {expr!r}): {e}")
        return "Reactive question: " + response("calc_bad_expression", expression=repr(expression)) + " Tell the user you couldn't work that out and ask them to rephrase."
    return response("calculation_result", value=_fmt_number(result))


# --------------------------------------------------------------------------
# Date / calendar maths via dateutil
# --------------------------------------------------------------------------

_WEEKDAYS = {
    "monday": MO,
    "mon": MO,
    "tuesday": TU,
    "tue": TU,
    "tues": TU,
    "wednesday": WE,
    "wed": WE,
    "thursday": TH,
    "thu": TH,
    "thur": TH,
    "thurs": TH,
    "friday": FR,
    "fri": FR,
    "saturday": SA,
    "sat": SA,
    "sunday": SU,
    "sun": SU,
}


def _parse_date(value: str) -> date:
    """Parse an ISO/natural date or the words today/tomorrow/yesterday.

    Missing components (e.g. no year in "25 December") fill from today via
    dateutil's `default`, so "25 December" resolves to this year.
    """
    v = (value or "").strip().lower()
    today = _local_time.today()
    if v in ("today", "now"):
        return today
    if v == "tomorrow":
        return today + timedelta(days=1)
    if v == "yesterday":
        return today - timedelta(days=1)
    base = datetime(today.year, today.month, today.day)
    return _dateparser.parse(value.strip(), default=base).date()


def has_relative_calendar_reference(value: str) -> bool:
    """Whether text contains a calendar phrase that needs deterministic maths."""
    return bool(
        re.search(
            r"\b(?:today|tomorrow|yesterday|(?:this|next|last)\s+"
            r"(?:week|month|year|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"in\s+\d+\s+(?:day|week|month|year)s?)\b",
            value or "",
            re.I,
        )
    )


def _calendar_reference(value: str) -> tuple[date, date] | None:
    """Resolve common relative dates to one date or an inclusive date range."""
    text = (value or "").lower()
    today = _local_time.today()
    if match := re.search(
        r"\b(this|next|last)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        text,
    ):
        modifier, weekday = match.groups()
        target = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday").index(
            weekday
        )
        days = (target - today.weekday()) % 7
        if modifier == "next" and days == 0:
            days = 7
        elif modifier == "last":
            days -= 7 if days == 0 else 7
        result = today + timedelta(days=days)
        return result, result
    if re.search(r"\bnext week\b", text):
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if re.search(r"\blast week\b", text):
        start = today - timedelta(days=today.weekday() + 7)
        return start, start + timedelta(days=6)
    if re.search(r"\bthis week\b", text):
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6)
    if re.search(r"\bnext month\b", text):
        start = today.replace(day=1) + relativedelta(months=1)
        return start, start + relativedelta(months=1, days=-1)
    if re.search(r"\blast month\b", text):
        start = today.replace(day=1) - relativedelta(months=1)
        return start, start + relativedelta(months=1, days=-1)
    if re.search(r"\bthis month\b", text):
        start = today.replace(day=1)
        return start, start + relativedelta(months=1, days=-1)
    if re.search(r"\bnext year\b", text):
        start = today.replace(year=today.year + 1, month=1, day=1)
        return start, start.replace(year=start.year + 1) - timedelta(days=1)
    if re.search(r"\blast year\b", text):
        start = today.replace(year=today.year - 1, month=1, day=1)
        return start, start.replace(year=start.year + 1) - timedelta(days=1)
    if re.search(r"\bthis year\b", text):
        start = today.replace(month=1, day=1)
        return start, start.replace(year=start.year + 1) - timedelta(days=1)
    if re.search(r"\bnext weekend\b", text):
        start = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
        return start, start + timedelta(days=1)
    if re.search(r"\bthis weekend\b", text):
        start = today + timedelta(days=(5 - today.weekday()) % 7)
        return start, start + timedelta(days=1)
    if match := re.search(r"\bin\s+(\d+)\s+(day|week|month|year)s?\b", text):
        count, unit = int(match.group(1)), match.group(2)
        result = today + relativedelta(**{unit + "s": count})
        return result, result
    if "tomorrow" in text:
        return today + timedelta(days=1), today + timedelta(days=1)
    if "yesterday" in text:
        return today - timedelta(days=1), today - timedelta(days=1)
    if "today" in text:
        return today, today
    return None


def _iso_date_offset(value: str) -> date | None:
    """Resolve ISO date offsets emitted by planning workers without arithmetic eval."""
    match = re.fullmatch(
        r"\s*(\d{4}-\d{2}-\d{2})\s*([+-])\s*(\d+)\s*(hour|day|week|month|year)s?\s*",
        value or "",
        re.I,
    )
    if match is None:
        return None
    value, operator, amount, unit = match.groups()
    try:
        start = date.fromisoformat(value)
    except ValueError:
        return None
    delta = relativedelta(**{unit.lower() + "s": int(amount)})
    result = start + delta if operator == "+" else start - delta
    return result.date() if isinstance(result, datetime) else result


def _friendly_date(d: date) -> str:
    """Speakable date, e.g. 'Sunday 14 June 2026' (no leading zero on the day)."""
    language = current_response_language.get()
    if language in {"fr", "de"}:
        from babel.dates import format_date

        return format_date(d, "full", locale=language)
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


@tool(
    name="days_between",
    description=(
        "Count whole days between two dates. Dates are ISO (YYYY-MM-DD) or "
        "today/tomorrow/yesterday; for 'days until X', pass today as date_a "
        "and the target as date_b."
    ),
    aliases=["date_difference", "days_until"],
)
def days_between(date_a: str, date_b: str) -> str:
    """Number of days between two dates, spoken-friendly."""
    try:
        a = _parse_date(date_a)
        b = _parse_date(date_b)
    except Exception:
        return "Reactive question: " + response("calc_bad_dates") + " Ask the user to clarify the dates."
    n = abs((b - a).days)
    if n == 0:
        return response("calc_same_day")
    language = current_response_language.get()
    if language == "fr":
        unit = "jour" if n == 1 else "jours"
    elif language == "de":
        unit = "Tag" if n == 1 else "Tage"
    else:
        unit = "day" if n == 1 else "days"
    return response("calc_days_between", count=n, unit=unit)


@tool(
    name="date_of",
    description=(
        "Resolve a weekday to an absolute calendar date. which: 'next' "
        "(default — the upcoming one, never today), 'this'/'coming' (nearest "
        "upcoming including today), or 'last'."
    ),
    aliases=["resolve_weekday", "what_date_is"],
)
def date_of(weekday: str, which: str = "next") -> str:
    """Return the absolute date of a named weekday relative to today."""
    wd = _WEEKDAYS.get((weekday or "").strip().lower())
    if wd is None:
        return "Reactive question: " + response("calc_bad_weekday") + " Ask the user which day they mean."
    mod = (which or "next").strip().lower()
    today = _local_time.today()
    if mod in ("this", "coming", "upcoming"):
        result = today + relativedelta(weekday=wd(+1))  # nearest, incl. today
    elif mod in ("last", "previous"):
        result = today + relativedelta(weekday=wd(-1))  # nearest past, incl. today
        if result == today:
            result = today + relativedelta(weeks=-1)
    else:  # "next" (default) — the upcoming occurrence, never today
        result = today + relativedelta(weekday=wd(+1))
        if result == today:
            result = today + relativedelta(weeks=+1)
    return response("calc_date_of", date=_friendly_date(result))


# --------------------------------------------------------------------------
# Unit conversion via pint
# --------------------------------------------------------------------------

_UREG = pint.UnitRegistry()

# Pint's bare gallon/pint/quart are US measures. Map ambiguous volume words to
# the household's timezone-derived convention; explicit US/American wording
# still always selects Pint's native US units.
_UK_VOLUME = {
    "gallon": "imperial_gallon",
    "gallons": "imperial_gallon",
    "pint": "imperial_pint",
    "pints": "imperial_pint",
    "quart": "imperial_quart",
    "quarts": "imperial_quart",
    "fluidounce": "imperial_fluid_ounce",
    "fluidounces": "imperial_fluid_ounce",
    "floz": "imperial_fluid_ounce",
}


def _pint_unit(u: str) -> str:
    """Map a spoken unit name to one pint understands.

    Strips the word 'degrees', honours an explicit 'US'/'American' prefix,
    and uses the household locale for ambiguous volume words.
    """
    s = (u or "").strip().lower()
    s = re.sub(r"\bdegrees?\b", "", s).strip()
    m = re.match(r"^(?:us|u\.?s\.?|american)\s+(.+)$", s)
    if m:
        return m.group(1).strip()
    key = s.replace(" ", "").replace("-", "")
    if household_locale().volume_system == "imperial":
        return _UK_VOLUME.get(key, s)
    return s


@tool(
    name="convert_units",
    description=(
        "Convert a value between units of length, mass, volume, temperature "
        "or speed (miles, km, kg, lb, litres, gallons, celsius, fahrenheit, "
        "mph). When to_unit is omitted, use the household's timezone-derived "
        "distance, temperature, or speed unit."
    ),
    aliases=["unit_conversion", "convert"],
)
def convert_units(value: float, from_unit: str, to_unit: str | None = None) -> str:
    """Convert `value` from one unit to another within the same dimension."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Reactive question: " + response("calc_bad_amount") + " Ask the user to clarify."

    frm = _pint_unit(from_unit)
    try:
        source = _UREG.Quantity(v, frm)
        display_to = to_unit
        if not isinstance(display_to, str) or not display_to.strip():
            locale = household_locale()
            if source.check("[length]"):
                display_to = locale.distance_unit
            elif source.check("[temperature]"):
                display_to = locale.temperature_unit
            elif source.check("[length] / [time]"):
                display_to = f"{locale.distance_unit} / hour"
            else:
                return "Reactive question: " + response("calc_missing_unit")
        to = _pint_unit(display_to)
        result = source.to(to)
    except pint.DimensionalityError:
        return "Reactive question: " + response("calc_incompatible_units", source=repr(from_unit), target=repr(to_unit)) + " Tell the user that."
    except Exception as e:
        logger.debug(f"convert_units failed ({from_unit!r}->{to_unit!r}): {e}")
        return "Reactive question: " + response("calc_unknown_units", source=repr(from_unit), target=repr(to_unit)) + " Ask the user to rephrase."
    language = current_response_language.get()
    if language in {"fr", "de"}:
        from babel.units import UnknownUnitError, format_unit

        def spoken_quantity(value, unit, fallback):
            name = str(unit).replace("degree_Celsius", "celsius").replace("degree_Fahrenheit", "fahrenheit")
            try:
                return format_unit(value, name, locale=language)
            except (UnknownUnitError, ValueError, KeyError):
                return f"{_fmt_number(value)} {fallback}"

        source_text = spoken_quantity(v, source.units, from_unit)
        result_text = spoken_quantity(result.magnitude, result.units, display_to)
    else:
        source_text = f"{_fmt_number(v)} {from_unit}"
        result_text = f"{_fmt_number(result.magnitude)} {display_to}"
    return response("calc_conversion", source=source_text, result=result_text)

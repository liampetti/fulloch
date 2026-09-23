"""Shared raw-value coercion used by Laya-routed tools."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.value_parsing import parse_percentage


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(70, 70), ("seventy percent", 70), ("a hundred percent", 100), ("150 percent", 100), ("full", 100)],
)
def test_parse_percentage_accepts_raw_spoken_values(raw, expected):
    assert parse_percentage(raw) == expected


def test_parse_percentage_rejects_non_percentage_text():
    with pytest.raises(ValueError):
        parse_percentage("rather bright")

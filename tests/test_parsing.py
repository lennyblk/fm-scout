from datetime import datetime

import pytest

from fm_scout.parsing import (parse_currency, parse_date, parse_money, parse_positions,
                              parse_range, parse_wage, query_positions)


@pytest.mark.parametrize("text,expected", [
    ("€5.2M", (5.2e6, 5.2e6)), ("5,2 M€", (5.2e6, 5.2e6)), ("£750K", (75e4, 75e4)),
    ("€65,000,000", (65e6, 65e6)), ("12 500 €", (12500, 12500)),
    ("12 500 €", (12500, 12500)), ("€12.500", (12500, 12500)),
    ("€1,234.50", (1234.5, 1234.5)), ("€1.5M - €3.1M", (1.5e6, 3.1e6)),
    ("4 - 6M€", (4e6, 6e6)), ("900K - 1,2M €", (9e5, 1.2e6)), ("€500 - €1K", (500, 1000)),
    ("€1.5Md", (1.5e9, 1.5e9)), ("5M", (5e6, 5e6)), ("€0", (0, 0)), ("Free", (0, 0)),
    ("€12K p/w", (12e3, 12e3)),
    ("—", None), ("-", None), ("", None), ("N/A", None), ("Not for Sale", None),
])
def test_money(text, expected):
    assert parse_money(text) == (pytest.approx(expected) if expected else None)


@pytest.mark.parametrize("text,weekly", [
    ("€12K p/w", 12000), ("12K € p/s", 12000), ("12 000 € par semaine", 12000),
    ("12 000 € / semaine", 12000), ("€52K p/a", 1000), ("52 000 €/an", 1000),
    ("€13K p/m", 3000), ("13K€/mois", 3000), ("€500", 500),
])
def test_wage(text, weekly):
    assert parse_wage(text)[1] == pytest.approx(weekly)


@pytest.mark.parametrize("text,codes", [
    ("GK", ["GK"]), ("D (RC)", ["DC", "DR"]), ("D/WB (RL)", ["DL", "DR", "WBL", "WBR"]),
    ("DM, M (C)", ["DM", "MC"]), ("AM (RLC), ST (C)", ["AMC", "AML", "AMR", "ST"]),
    ("D (DG)", ["DL", "DR"]), ("MO (G), BT (C)", ["AML", "ST"]), ("MC/AML", ["MC", "AML"]), ("DL, STC", ["DL", "ST"]),
    ("XYZ", []),
])
def test_positions(text, codes):
    assert parse_positions(text) == codes


def test_query():
    assert query_positions("mdc") == ["DM"]
    assert query_positions("Latéral") == ["DL", "DR"]
    assert query_positions("zzz") is None


@pytest.mark.parametrize("text,expected", [
    ("14", (14, 14)), ("10-14", (10, 14)), ("3,5", (3.5, 3.5)), ("-", None)])
def test_range(text, expected):
    assert parse_range(text) == expected


@pytest.mark.parametrize("text", [
    "30/6/2028", "30.06.2028", "2028-06-30", "30 juin 2028", "30 Jun 2028"])
def test_date(text):
    assert parse_date(text) == datetime(2028, 6, 30)


def test_bad_date():
    assert parse_date("N/A") is None and parse_date("40/1/2028") is None


def test_currency():
    assert [parse_currency(t) for t in ("€1M", "£2", "5")] == ["EUR", "GBP", None]

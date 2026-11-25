from pathlib import Path

import pytest

from fm_scout.config import load_config
from fm_scout.schema import ScoutError
from fm_scout.scouting.compare import compare
from fm_scout.scouting.roles import build_roles
from fm_scout.scouting.squad import get_squad_needs
from fm_scout.sources.csv_source import CsvSource

CONFIG = load_config(Path("absent.toml"))
ROLES = build_roles(CONFIG.roles)
ATTRS = list(CONFIG.columns["attributes"])


@pytest.fixture
def data():
    return CsvSource(Path(__file__).parent / "fixtures", CONFIG.columns).load()


def test_needs(data):
    r = get_squad_needs(data, ROLES, CONFIG.zones, expiring_months=24)
    zones = {z["zone"]: z for z in r["zones"]}
    assert r["players"] == 2 and r["average_age"] == 24
    assert zones["milieu defensif"]["depth"] == 1 and zones["milieu defensif"]["gap"] == 1
    assert zones["attaque"]["depth"] == 0  # le joueur prêté compte pour son club
    assert [p["name"] for p in r["expiring"]] == ["Moi Même"]


def test_needs_first_team_only(data):
    data.players["first_team"] = data.players["player_id"] == "8"
    assert get_squad_needs(data, ROLES, CONFIG.zones)["players"] == 1


def test_needs_without_club(data):
    data.my_club = None
    with pytest.raises(ScoutError):
        get_squad_needs(data, ROLES, CONFIG.zones)


def test_compare(data):
    a, b = compare(data, ROLES, ["1", "cheron"], ATTRS, role="sentinelle")
    assert (a["name"], b["name"]) == ("Léo Fictif", "Max Cherón")
    assert b["role_score"] > a["role_score"] and a["attributes"]["tackling"] == 14
    assert "dribbling" not in a["attributes"]


def test_compare_errors(data):
    with pytest.raises(ScoutError, match="introuvable"):
        compare(data, ROLES, ["zzz"], ATTRS)
    with pytest.raises(ScoutError, match="Plusieurs"):
        compare(data, ROLES, ["o"], ATTRS)

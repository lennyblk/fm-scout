from datetime import date
from pathlib import Path

import pytest

from fm_scout.config import load_config
from fm_scout.schema import ScoutError
from fm_scout.scouting.roles import build_roles
from fm_scout.scouting.search import search_players
from fm_scout.sources.csv_source import CsvSource

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_config(Path("absent.toml"))
ROLES = build_roles(CONFIG.roles)


@pytest.fixture
def data():
    return CsvSource(FIXTURES, CONFIG.columns).load()


def names(data, **kw):
    return [p["name"] for p in search_players(data, ROLES, **kw).players]


def test_load(data):
    assert data.my_club == "Mon Club" and data.currency == "EUR"
    assert data.players["player_id"].is_unique and len(data.players) == 8
    assert data.game_date == date(2027, 4, 15)
    assert set(data.players["nationality"].dropna()) == {"France", "Spain", "Morocco"}


def test_duplicate_keeps_known_values(data):
    me = data.players.set_index("player_id").loc["6"]
    assert me["value_max"] == 1e6 and me["tackling"] == 12  # absents de squad_sample


def test_main_request(data):
    r = search_players(data, ROLES, position="MDC", max_age=21, max_fee=5e6)
    # Max demande 9M malgré une valeur de 2M ; Jo n'a pas de prix ; Moi Même est à moi.
    assert [p["name"] for p in r.players] == ["Léo Fictif"]
    assert r.excluded_unknown == {"max_fee": 1}


def test_range_uses_upper_bound(data):
    assert "Léo Fictif" not in names(data, position="MDC", max_fee=4e6)


def test_free(data):
    assert names(data, free_agents_only=True) == ["Ali Libre"]
    assert "Ali Libre" in names(data, max_fee=0)


def test_contract(data):
    assert names(data, contract_max_months=6) == ["Tom Vieux"]
    data.game_date = None
    with pytest.raises(ScoutError, match="Date du jeu"):
        search_players(data, ROLES, contract_max_months=6)


def test_wage_nationality_age(data):
    assert names(data, max_wage=1000) == ["Tom Vieux"]
    assert names(data, nationality="france", min_age=25) == ["Tom Vieux"]


def test_role(data):
    r = search_players(data, ROLES, position="MDC", role="sentinelle")
    assert r.players[0]["name"] == "Max Cherón" and r.players[0]["role_coverage"] == 1
    assert r.excluded_unknown == {"role": 1}
    assert "Jo Inconnu" not in [p["name"] for p in r.players]


@pytest.mark.parametrize("kw", [{"min_ca": 100}, {"position": "zzz"}, {"role": "zzz"}])
def test_errors(data, kw):
    with pytest.raises(ScoutError):
        search_players(data, ROLES, **kw)


def test_role_without_position_uses_role_positions(data):
    assert names(data, role="attaquant_avance") == []
    assert "Ali Libre" not in names(data, role="sentinelle")


def test_position_without_role_ranks_by_best_role(data):
    first = search_players(data, ROLES, position="MDC").players[0]
    assert first["name"] == "Max Cherón" and first["role"] == "sentinelle"  # pas le moins cher
    # Sans filtre de rôle demandé, un joueur aux attributs inconnus reste visible.
    assert "Jo Inconnu" in names(data, position="MDC")


def test_no_result(data):
    assert search_players(data, ROLES, position="GK", role="gardien").total == 0


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252"])
@pytest.mark.parametrize("sep", [";", "\t"])
def test_encodings_french_headers(tmp_path, encoding, sep):
    f = tmp_path / "x.csv"
    f.write_text(f"Nom{sep}Âge{sep}Poste{sep}Valeur\nRené{sep}20{sep}MD{sep}5,2 M€\n",
                 encoding=encoding)
    row = CsvSource(f, CONFIG.columns).load().players.iloc[0]
    assert (row["name"], row["age"], row["positions"], row["value_max"]) == (
        "René", 20, ["DM"], 5.2e6)


def test_mixed_currency(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("Player,Transfer Value\nA,€1M\nB,£1M\n", encoding="utf-8")
    with pytest.raises(ScoutError, match="Devises"):
        CsvSource(f, CONFIG.columns).load()


def test_missing_path(tmp_path):
    with pytest.raises(ScoutError, match="Aucun export"):
        CsvSource(tmp_path, CONFIG.columns).load()

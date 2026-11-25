from datetime import date
from itertools import permutations
from pathlib import Path

import pandas as pd
import pytest

from fm_scout.config import load_config
from fm_scout.schema import Budget, Dataset, ScoutError
from fm_scout.scouting.lineup import assign
from fm_scout.scouting.recommend import _best_plan, diagnose, growth_bonus, recommend
from fm_scout.scouting.roles import Role

CONFIG = load_config(Path("absent.toml"))
ROLES = {"buteur": Role("buteur", ["ST"], {"finishing": 1}), "gardien": Role("gardien", ["GK"], {"reflexes": 1})}


def player(pid, club, positions, level, age=24, value=1e6, wage=1e3, rep=5000, loan=False):
    return {"player_id": pid, "name": f"J{pid}", "club": club, "positions": positions, "age": float(age),
            "finishing": float(level), "reflexes": float(level), "value_min": value, "value_max": value,
            "wage_min": wage, "wage_max": wage, "contract_end": pd.Timestamp(2030, 6, 30),
            "club_reputation": float(rep), "loan_from_me": loan, "on_loan": loan, "first_team": True}


@pytest.fixture
def data():
    rows = [player("g1", "Moi", ["GK"], 16, wage=9e3), player("g2", "Moi", ["GK"], 15),
            player("s1", "Moi", ["ST"], 10), player("s2", "Moi", ["ST"], 9)]
    rows += [player(f"p{i}", f"Pair{i}", ["ST", "GK"], 15, value=50e6) for i in range(3)]
    rows += [player("bon", "Petit", ["ST"], 16, value=4e6), player("cher", "Petit", ["ST"], 18, value=9e6),
             player("vieux", "Petit", ["ST"], 19, age=33), player("star", "Géant", ["ST"], 20, rep=9000),
             player("pret", "Petit", ["ST"], 20, loan=True), player("salaire", "Petit", ["ST"], 20, wage=50e3)]
    return Dataset(pd.DataFrame(rows), my_club="Moi", game_date=date(2026, 6, 1), formation=["GK", "ST"],
                   budget=Budget(None, 10e6, 20e3, 10e3))


def rec(data, **kw):
    return recommend(data, ROLES, CONFIG.recommend, **kw)


def test_assign_matches_brute_force():
    scores = [[5, 9, 1, 4], [8, 9, 3, None], [None, 2, 7, 6]]
    best = max(sum(scores[i][j] for i, j in enumerate(p)) for p in permutations(range(4), 3)
               if all(scores[i][j] is not None for i, j in enumerate(p)))
    got = assign(scores)
    assert sum(scores[i][j] for i, j in enumerate(got)) == best and len(set(got)) == 3


def test_assign_leaves_impossible_empty():
    assert assign([[None, None]]) == [None]
    assert assign([[5], [4]]) == [0, None]


def test_weak_position_first(data):
    assert diagnose(data, ROLES, CONFIG.recommend, 1)[0]["position"] == "ST"


def test_candidates_respect_every_limit(data):
    need = rec(data)["needs"][0]
    # cher : 9M x 1,25 > 10M ; vieux, star, prêté, salaire : écartés.
    assert [c["player_id"] for c in need["candidates"]] == ["bon"] and need["pick"] == "bon"


def test_budget_override(data):
    assert rec(data, budget=20e6)["needs"][0]["candidates"][0]["player_id"] == "cher"
    assert rec(data, budget=0)["needs"][0]["candidates"] == []


def test_total_never_exceeds_budget(data):
    r = rec(data)
    assert r["spent"] <= r["budget"]


def test_plan_spreads_budget_instead_of_one_star():
    def c(pid, fee, gain):
        return {"player_id": pid, "planned_fee": fee, "wage_max": 1, "upgrade": gain, "growth": 0, "for": "titulaire"}
    needs = [{"candidates": [c("star", 10, 9), c("ok", 5, 6)], "pick": None},
             {"candidates": [c("autre", 5, 6), c("ok", 1, 5)], "pick": None}]
    _best_plan(needs, 10, None)
    assert [n["pick"] for n in needs] == ["ok", "autre"]
    _best_plan(needs, 10, 1)  # marge salariale pour un seul joueur
    assert [n["pick"] for n in needs] == ["star", None]


def test_planner_defines_squad(data):
    data.plan = {1: {"GK": ["g1", "g2"], "ST": ["s2"]}}
    r = rec(data)
    need = next(n for n in r["needs"] if n["position"] == "ST")
    assert r["from_planner"] and need["starter"]["player_id"] == "s2" and need["backup"] is None
    assert "pas de doublure" in need["reasons"]


def test_future_signing_not_recommended(data):
    data.plan = {1: {"GK": ["g1", "g2"], "ST": ["s1"]}, 2: {"ST": ["bon"]}}
    need = next(n for n in rec(data)["needs"] if n["position"] == "ST")
    assert "bon" not in [c["player_id"] for c in need["candidates"]]


def with_potential(data, pid, age, ca, pa):
    if "ca" not in data.players:
        data.players[["ca", "pa"]] = 120.0
    data.players.loc[data.players["player_id"] == pid, ["age", "ca", "pa"]] = [age, ca, pa]


def test_growth_bonus_by_age():
    df = pd.DataFrame({"age": [18.0, 23.0, 26.0, 18.0], "ca": [100.0] * 4, "pa": [150.0, 150.0, 150.0, None]})
    assert list(growth_bonus(df, CONFIG.recommend)) == pytest.approx([4.0, 2.0, 0.0, 0.0])
    assert list(growth_bonus(df.drop(columns="pa"), CONFIG.recommend)) == [0.0] * 4


def test_potential_changes_ranking(data):
    data.players.loc[len(data.players)] = player("espoir", "Petit", ["ST"], 15.8, value=4e6)
    ids = [c["player_id"] for c in rec(data)["needs"][0]["candidates"]]
    assert ids == ["bon", "espoir"]
    with_potential(data, "espoir", 18, 110, 170)
    need = rec(data)["needs"][0]
    assert [c["player_id"] for c in need["candidates"]] == ["espoir", "bon"] and need["pick"] == "espoir"


def test_own_prospect_lowers_priority(data):
    before = diagnose(data, ROLES, CONFIG.recommend, 1)[0]["priority"]
    with_potential(data, "s1", 18, 100, 170)
    assert diagnose(data, ROLES, CONFIG.recommend, 1)[0]["priority"] < before
    assert diagnose(data, ROLES, CONFIG.recommend, 0)[0]["priority"] == before


def test_unknown_budget(data):
    data.budget = None
    with pytest.raises(ScoutError, match="Budget"):
        rec(data)

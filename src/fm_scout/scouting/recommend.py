"""Diagnostic du plan d'effectif, puis recrues possibles. Entièrement déterministe."""
from itertools import product
from typing import Any

import pandas as pd

from fm_scout.schema import Dataset, ScoutError
from fm_scout.scouting.lineup import assign
from fm_scout.scouting.roles import Role, best_role_scores
from fm_scout.scouting.search import add_fee, to_record
from fm_scout.scouting.squad import my_squad

# Sans tactique lisible : 4-2-3-1.
DEFAULT_FORMATION = ["GK", "DR", "DC", "DC", "DL", "DM", "DM", "AMR", "AMC", "AML", "ST"]


def _can_play(players: pd.DataFrame, position: str) -> pd.Series:
    return pd.Series([isinstance(p, list) and position in p for p in players["positions"]],
                     index=players.index, dtype=bool)


def growth_bonus(players: pd.DataFrame, settings: dict[str, Any]) -> pd.Series:
    """Points de score ajoutés par le potentiel restant ; 0 si la CA/PA n'est pas connue."""
    if not {"ca", "pa", "age"} <= set(players.columns):
        return pd.Series(0.0, index=players.index)
    span = settings["growth_zero_age"] - settings["growth_full_age"]
    young = ((settings["growth_zero_age"] - players["age"]) / span).clip(0, 1)
    return ((players["pa"] - players["ca"]).clip(lower=0) * young * settings["growth_weight"]).fillna(0.0)


def _scores(players: pd.DataFrame, position: str, roles: dict[str, Role]) -> pd.DataFrame:
    return best_role_scores(players, [r for r in roles.values() if position in r.positions])


def _planned(data: Dataset, season: int) -> tuple[pd.DataFrame, dict[str, set[str]] | None]:
    """Joueurs comptant pour la saison, et leurs postes prévus si un plan existe."""
    plan = data.plan.get(season) if data.plan else None
    if not plan or not any(plan.values()):
        return my_squad(data), None
    ids = {pid for pids in plan.values() for pid in pids}
    squad = data.players[data.players["player_id"].isin(ids)]
    return squad, {position: set(pids) for position, pids in plan.items()}


def _reference(data: Dataset, position: str, roles: dict[str, Role], starters: int, peers: int) -> float | None:
    """Niveau médian des titulaires à ce poste dans les clubs de réputation proche."""
    df = data.players
    if "club_reputation" not in df:
        return None
    mine = df.loc[df["club"] == data.my_club, "club_reputation"].median()
    clubs = df[(df["club"] != data.my_club) & (df["club"] != "")].groupby("club")["club_reputation"].first()
    nearest = (clubs - mine).abs().dropna().sort_values(kind="stable").index[:peers]
    pool = df[df["club"].isin(nearest) & _can_play(df, position)]
    scores = _scores(pool, position, roles)["role_score"]
    best = [g.dropna().nlargest(starters).mean() for _, g in scores.groupby(pool["club"])]
    value = pd.Series(best, dtype=float).median()
    return None if pd.isna(value) else float(value)


def _summary(row: pd.Series, score: float | None, growth: float) -> dict[str, Any] | None:
    if score is None:
        return None
    end = row.get("contract_end")
    return {"player_id": row["player_id"], "name": row["name"],
            "age": None if pd.isna(row.get("age")) else int(row["age"]), "score": round(score, 1), "growth": round(growth, 1),
            "contract_end": None if end is None or pd.isna(end) else end.date().isoformat()}


def diagnose(data: Dataset, roles: dict[str, Role], settings: dict[str, Any], season: int) -> list[dict[str, Any]]:
    """Une ligne par place de la tactique, de la plus à la moins prioritaire."""
    squad, plan = _planned(data, season)
    if squad.empty:
        raise ScoutError("Aucun joueur dans le plan d'effectif pour cette saison.")
    formation = data.formation or DEFAULT_FORMATION
    matrix: dict[str, list[float | None]] = {}
    for position in set(formation):
        scores = _scores(squad, position, roles)["role_score"]
        allowed = (squad["player_id"].isin(plan.get(position, set())) if plan is not None
                   else _can_play(squad, position))
        matrix[position] = [float(s) if ok and pd.notna(s) else None for s, ok in zip(scores, allowed)]
    starters = assign([matrix[p] for p in formation])
    taken = {i for i in starters if i is not None}
    backups = assign([[None if j in taken else s for j, s in enumerate(matrix[p])] for p in formation])

    weights = settings["weights"]
    # Le potentiel n'aide que les saisons à venir.
    growth = growth_bonus(squad, settings) * (1.0 if season else 0.0)
    references = {p: _reference(data, p, roles, formation.count(p), settings["peer_clubs"])
                  for p in set(formation)}
    result = []
    for position, i, j in zip(formation, starters, backups):
        reference = references[position]
        starter = _summary(squad.iloc[i], matrix[position][i], float(growth.iloc[i])) if i is not None else None
        backup = _summary(squad.iloc[j], matrix[position][j], float(growth.iloc[j])) if j is not None else None
        points, reasons = 0.0, []

        def add(value: float, reason: str) -> None:
            nonlocal points
            points += value
            reasons.append(reason)
        if starter is None:
            add(weights["no_starter"], "aucun titulaire")
        else:
            gap = 0.0 if reference is None else reference - starter["score"] - starter["growth"]
            if gap > 0:
                note = f" (potentiel compté : +{starter['growth']})" if starter["growth"] else ""
                add(gap * weights["starter_gap"], f"titulaire {gap:.1f} points sous la référence{note}")
            if starter["age"] is not None and starter["age"] >= settings["aging_age"]:
                add(weights["starter_aging"], f"titulaire âgé de {starter['age']} ans")
            if data.game_date and starter["contract_end"]:
                last = pd.Timestamp(data.game_date) + pd.DateOffset(months=13)
                if pd.Timestamp(starter["contract_end"]) <= last:
                    add(weights["starter_expiring"], f"contrat du titulaire jusqu'au {starter['contract_end']}")
        if backup is None:
            add(weights["no_backup"], "pas de doublure")
        elif reference is not None:
            gap = reference - settings["backup_tolerance"] - backup["score"] - backup["growth"]
            if gap > 0:
                add(gap * weights["backup_gap"], f"doublure faible ({backup['score']})")
        result.append({"position": position, "priority": round(points, 1), "reasons": reasons,
                       "starter": starter, "backup": backup,
                       "reference": None if reference is None else round(reference, 1)})
    return sorted(result, key=lambda r: -r["priority"])


def _candidates(data: Dataset, roles: dict[str, Role], settings: dict[str, Any], need: dict[str, Any],
                budget: float, squad_ids: set[str], max_wage: float | None, strict: bool) -> list[dict[str, Any]]:
    df = data.players
    df = df[_can_play(df, need["position"]) & ~df["player_id"].isin(squad_ids)]
    df = df[(df["club"] != data.my_club) & (df["age"] <= settings["max_age"])]
    if "loan_from_me" in df:
        df = df[~df["loan_from_me"].eq(True) & ~df["on_loan"].eq(True)]
    if "club_reputation" in df:
        mine = data.players.loc[data.players["club"] == data.my_club, "club_reputation"].median()
        df = df[df["club_reputation"] <= mine + settings["reputation_margin"]]
    if max_wage is not None:
        df = df[df["wage_max"] <= max_wage]  # salaire inconnu : écarté
    df = add_fee(df)
    df["planned_fee"] = df["fee_max"] * settings["fee_margin"]
    df = df[df["planned_fee"] <= budget]
    df = df.join(_scores(df, need["position"], roles))
    # strict : mieux que le titulaire ; sinon mieux que la doublure.
    holder = need["starter"] if strict else need["backup"]
    floor = holder["score"] + settings["min_upgrade"] if holder else (need["reference"] or 0) - (
        0 if strict else settings["backup_tolerance"])
    df["growth"] = growth_bonus(df, settings)
    df["rank"] = df["role_score"] + df["growth"]
    df = df[(df["rank"] >= floor) & (df["role_score"] >= floor - settings["prospect_tolerance"])]
    df = df.sort_values(["rank", "planned_fee"], ascending=[False, True], kind="stable")
    records = []
    for _, row in df.head(settings["candidates"]).iterrows():
        record = to_record(row)
        record["planned_fee"] = round(float(row["planned_fee"]))
        record["upgrade"] = round(float(row["role_score"]) - (holder["score"] if holder else 0), 1) if holder else None
        record["growth"] = round(float(row["growth"]), 1)
        record["for"] = "titulaire" if strict else "doublure"
        best = roles[row["role"]].weights
        record["attributes"] = {a: float(row[a]) for a in best if a in row and pd.notna(row[a])}
        records.append(record)
    return records


def _best_plan(needs: list[dict[str, Any]], budget: float, room: float | None) -> None:
    """Choisit au plus une recrue par besoin : gain (niveau + potentiel) total maximal sans dépasser le budget ni les salaires."""
    best: tuple[float, float] = (-1.0, 0.0)
    for combo in product(*[[None, *n["candidates"]] for n in needs]):
        picks = [c for c in combo if c]
        cost = sum(c["planned_fee"] for c in picks)
        wages = sum(c["wage_max"] for c in picks)
        if (len({c["player_id"] for c in picks}) < len(picks) or cost > budget
                or (room is not None and wages > room)):
            continue
        # Une doublure compte moitié moins ; un poste vide vaut un gain forfaitaire de 10.
        gain = sum(((10.0 if c["upgrade"] is None else c["upgrade"]) + c["growth"])
                   * (1.0 if c["for"] == "titulaire" else 0.5) for c in picks)
        if (gain, -cost) > best:
            best = (gain, -cost)
            for need, c in zip(needs, combo):
                need["pick"] = c["player_id"] if c else None


def recommend(data: Dataset, roles: dict[str, Role], settings: dict[str, Any], *,
              season: int = 1, budget: float | None = None) -> dict[str, Any]:
    """season : 0 = en cours, 1 = prochaine, 2 = suivante."""
    missing = {"positions", "age", "club", "wage_max"} - set(data.players.columns)
    if missing or not data.my_club:
        raise ScoutError("Recommandations impossibles : données manquantes ("
                         f"{', '.join(sorted(missing)) or 'mon club'}). Utilisez la save.")
    if budget is None:
        budget = data.budget.transfer_remaining if data.budget else None
    if budget is None:
        raise ScoutError("Budget inconnu : précisez --budget.")
    needs = diagnose(data, roles, settings, season)
    squad, plan = _planned(data, season)
    squad_ids = set(squad["player_id"])
    for pids in (data.plan or {}).values():
        squad_ids |= {pid for ids in pids.values() for pid in ids}
    top = my_squad(data)["wage_max"].max()
    max_wage = None if pd.isna(top) else float(top) * settings["wage_factor"]
    room = None
    if data.budget and data.budget.wage_budget_weekly is not None and data.budget.wage_payroll_weekly is not None:
        room = max(0.0, data.budget.wage_budget_weekly - data.budget.wage_payroll_weekly)
        max_wage = room if max_wage is None else min(max_wage, room)

    targets = [n for n in needs if n["priority"] > 0][:settings["priorities"]]
    for need in targets:
        weak_starter = need["starter"] is None or any(r.startswith("titulaire") or "contrat" in r for r in need["reasons"])
        need["candidates"], need["pick"] = [], None
        for strict in ([True, False] if weak_starter else [False]):
            need["candidates"] = _candidates(data, roles, settings, need, budget, squad_ids, max_wage, strict)
            if need["candidates"]:
                break
    _best_plan(targets, budget, room)
    spent = sum(c["planned_fee"] for n in targets for c in n["candidates"] if c["player_id"] == n["pick"])
    return {"club": data.my_club, "season": season, "from_planner": plan is not None,
            "formation": data.formation or DEFAULT_FORMATION, "budget": round(budget),
            "spent": round(spent), "wage_room": None if room is None else round(room),
            "needs": targets,
            "later": [n for n in needs if n["priority"] > 0 and n not in targets],
            "ok": [n for n in needs if n["priority"] <= 0]}

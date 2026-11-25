from dataclasses import asdict
from typing import Any

import pandas as pd

from fm_scout.schema import Dataset, ScoutError
from fm_scout.scouting.roles import Role, role_scores

OLD_AGE = 30


def my_squad(data: Dataset) -> pd.DataFrame:
    """Équipe première si la source la distingue, sinon tous les joueurs du club."""
    if not data.my_club or "club" not in data.players:
        raise ScoutError("Club inconnu : impossible d'analyser l'effectif.")
    squad = data.players[data.players["club"] == data.my_club]
    if "first_team" in squad and squad["first_team"].eq(True).any():
        squad = squad[squad["first_team"].eq(True)]
    return squad


def _playing(squad: pd.DataFrame, positions: list[str]) -> pd.DataFrame:
    keep = [isinstance(p, list) and bool(set(p) & set(positions)) for p in squad["positions"]]
    return squad[keep]


def _best_scores(players: pd.DataFrame, positions: list[str], roles: dict[str, Role]) -> pd.Series:
    """Meilleur score de chaque joueur parmi les rôles de la zone."""
    scores = [role_scores(players, r)["role_score"] for r in roles.values()
              if set(r.positions) & set(positions)]
    if not scores:
        return pd.Series(float("nan"), index=players.index)
    return pd.concat(scores, axis=1).max(axis=1)


def get_squad_needs(data: Dataset, roles: dict[str, Role], zones: dict[str, Any],
                    expiring_months: int = 12) -> dict[str, Any]:
    squad = my_squad(data)
    if "positions" not in squad:
        raise ScoutError("Postes absents de la source : impossible d'analyser l'effectif.")
    result: dict[str, Any] = {
        "club": data.my_club, "players": len(squad),
        "average_age": round(float(squad["age"].mean()), 1) if "age" in squad else None,
        "budget": asdict(data.budget) if data.budget else None, "zones": [], "expiring": []}
    for name, zone in zones.items():
        players = _playing(squad, zone["positions"])
        scores = _best_scores(players, zone["positions"], roles)
        best = scores.dropna().sort_values(ascending=False).head(2)
        old = int((players["age"] >= OLD_AGE).sum()) if "age" in players else 0
        result["zones"].append({
            "zone": name, "depth": len(players), "target": zone["target"],
            "gap": max(0, zone["target"] - len(players)),
            "average_age": round(float(players["age"].mean()), 1) if len(players) and "age" in players else None,
            "aged_30_plus": old,
            "best": [{"name": str(players.at[i, "name"]), "score": float(s)} for i, s in best.items()]})
    if data.game_date and "contract_end" in squad:
        limit = pd.Timestamp(data.game_date) + pd.DateOffset(months=expiring_months)
        ending = squad[squad["contract_end"] <= limit].sort_values("contract_end")
        result["expiring"] = [{"name": r["name"], "positions": r["positions"], "age": int(r["age"]) if pd.notna(r["age"]) else None,
                               "contract_end": r["contract_end"].date().isoformat()}
                              for _, r in ending.iterrows()]
    return result

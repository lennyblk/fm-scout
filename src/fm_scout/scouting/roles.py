from dataclasses import dataclass
from typing import Any

import pandas as pd

from fm_scout.parsing import fold
from fm_scout.schema import ScoutError

MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class Role:
    name: str
    positions: list[str]
    weights: dict[str, float]


def build_roles(raw: dict[str, Any]) -> dict[str, Role]:
    return {name: Role(name, r["positions"], r["weights"]) for name, r in raw.items()}


def get_role(roles: dict[str, Role], name: str) -> Role:
    key = fold(name).replace(" ", "_")
    if key not in roles:
        raise ScoutError(f"Rôle inconnu « {name} ». Disponibles : {', '.join(roles)}")
    return roles[key]


def best_role_scores(players: pd.DataFrame, roles: list[Role]) -> pd.DataFrame:
    """Pour chaque joueur, son meilleur rôle parmi ceux proposés (colonne role)."""
    best = pd.DataFrame({"role": None, "role_score": float("nan"), "role_coverage": float("nan")},
                        index=players.index).astype({"role": object})
    for role in roles:
        scores = role_scores(players, role)
        better = scores["role_score"].notna() & ~(best["role_score"] >= scores["role_score"])
        best.loc[better, ["role_score", "role_coverage"]] = scores[better]
        best.loc[better, "role"] = role.name
    return best


def role_scores(players: pd.DataFrame, role: Role) -> pd.DataFrame:
    """role_score 0-100 (NaN si trop peu d'attributs connus) et role_coverage 0-1."""
    weights = pd.Series(role.weights, dtype=float)
    values = players.reindex(columns=weights.index).astype(float)
    known = values.notna().mul(weights).sum(axis=1)
    coverage = known / weights.sum()
    mean = values.mul(weights).sum(axis=1) / known.where(known > 0)
    score = ((mean - 1) / 19 * 100).where(coverage >= MIN_COVERAGE)
    return pd.DataFrame({"role_score": score.round(1), "role_coverage": coverage.round(2)})

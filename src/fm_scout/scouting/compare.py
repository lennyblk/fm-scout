from typing import Any

import pandas as pd

from fm_scout.parsing import fold
from fm_scout.schema import Dataset, ScoutError
from fm_scout.scouting.roles import Role, get_role, role_scores
from fm_scout.scouting.search import add_fee, to_record


def resolve(players: pd.DataFrame, wanted: str) -> int:
    """Index du joueur désigné par identifiant ou par nom."""
    by_id = players.index[players["player_id"] == wanted]
    if len(by_id):
        return by_id[0]
    key = fold(wanted)
    folded = players["name"].map(fold)
    found = players.index[folded == key]
    if not len(found):
        found = players.index[[key in f for f in folded]]
    if len(found) == 1:
        return found[0]
    if not len(found):
        raise ScoutError(f"Joueur introuvable : {wanted}")
    shown = "; ".join(f"{players.at[i, 'player_id']} = {players.at[i, 'name']}" for i in found[:8])
    raise ScoutError(f"Plusieurs joueurs pour « {wanted} », précisez l'identifiant : {shown}")


def compare(data: Dataset, roles: dict[str, Role], player_ids: list[str], attributes: list[str],
            role: str | None = None) -> list[dict[str, Any]]:
    """Fiches côte à côte. Avec un rôle : seulement ses attributs, et son score."""
    chosen = get_role(roles, role) if role else None
    rows = add_fee(data.players.loc[[resolve(data.players, p) for p in player_ids]])
    if chosen:
        rows = rows.join(role_scores(rows, chosen))
        attributes = list(chosen.weights)
    result = []
    for _, row in rows.iterrows():
        record = to_record(row)
        record["attributes"] = {a: float(row[a]) for a in attributes
                                if a in row and pd.notna(row[a])}
        result.append(record)
    return result

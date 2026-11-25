from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from fm_scout.parsing import fold, query_positions
from fm_scout.schema import Dataset, ScoutError
from fm_scout.scouting.roles import Role, best_role_scores, get_role


@dataclass
class SearchResult:
    total: int
    excluded_unknown: dict[str, int]
    players: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_fee(df: pd.DataFrame) -> pd.DataFrame:
    """fee_min/max : 0 si libre, sinon prix demandé, sinon valeur estimée."""
    df = df.copy()
    empty = pd.Series(float("nan"), index=df.index)
    asking = df.get("asking_price_max", empty)
    free = df["club"].eq("") if "club" in df else pd.Series(False, index=df.index)
    for side in ("min", "max"):
        fee = df.get(f"asking_price_{side}", empty).fillna(df.get(f"value_{side}", empty))
        df[f"fee_{side}"] = fee.mask(free, 0.0)
    source = pd.Series("valeur estimée", index=df.index, dtype=object)
    source[asking.notna()] = "prix demandé"
    source[free] = "libre"
    df["fee_source"] = source.where(df["fee_max"].notna(), None)
    return df


def search_players(
    data: Dataset, roles: dict[str, Role], *,
    position: str | None = None, min_age: int | None = None, max_age: int | None = None,
    max_fee: float | None = None, max_wage: float | None = None,
    min_ca: float | None = None, min_pa: float | None = None, role: str | None = None,
    contract_max_months: int | None = None, nationality: str | None = None,
    free_agents_only: bool = False, limit: int = 10,
) -> SearchResult:
    df = add_fee(data.players)
    unknown: dict[str, int] = {}

    def need(column: str, label: str) -> None:
        if column not in df:
            raise ScoutError(f"Filtre « {label} » impossible : la source ne fournit pas « {column} ».")

    def count_missing(column: str, label: str) -> None:
        if missing := int(df[column].isna().sum()):
            unknown[label] = missing

    if data.my_club and "club" in df:
        df = df[df["club"] != data.my_club]
    if "loan_from_me" in df:
        df = df[~df["loan_from_me"].eq(True)]
    codes: list[str] | None = None
    if position:
        codes = query_positions(position)
        if codes is None:
            raise ScoutError(f"Poste inconnu : {position}")
    elif role:  # un rôle sans poste : on se limite aux postes du rôle
        codes = get_role(roles, role).positions
    if codes:
        need("positions", "position")
        df = df[[isinstance(p, list) and bool(set(p) & set(codes)) for p in df["positions"]]]
    if nationality:
        need("nationality", "nationality")
        wanted = fold(nationality)
        df = df[[isinstance(n, str) and fold(n) == wanted for n in df["nationality"]]]
    if free_agents_only:
        need("club", "free_agents_only")
        df = df[df["club"].eq("")]

    if max_fee is not None and not {"value_max", "asking_price_max"} & set(df.columns):
        raise ScoutError("Filtre « max_fee » impossible : ni valeur ni prix dans la source.")
    bounds = [("min_age", "age", min_age, False), ("max_age", "age", max_age, True),
              ("max_fee", "fee_max", max_fee, True), ("max_wage", "wage_max", max_wage, True),
              ("min_ca", "ca", min_ca, False), ("min_pa", "pa", min_pa, False)]
    for label, column, bound, upper in bounds:
        if bound is not None:
            need(column, label)
            count_missing(column, label)
            df = df[(df[column] <= bound) if upper else (df[column] >= bound)]
    if contract_max_months is not None:
        need("contract_end", "contract_max_months")
        if data.game_date is None:
            raise ScoutError("Date du jeu inconnue : exportez les colonnes « DoB » et « Age », "
                             "ou passez --game-date AAAA-MM-JJ.")
        count_missing("contract_end", "contract_max_months")
        last_day = pd.Timestamp(data.game_date) + pd.DateOffset(months=contract_max_months)
        df = df[df["contract_end"] <= last_day]

    sort: list[tuple[str, bool]] = [("ca", False), ("ability_stars", False), ("fee_max", True)]
    if role:
        df = df.join(best_role_scores(df, [get_role(roles, role)]))
        count_missing("role_score", "role")
        df = df[df["role_score"].notna()]
        sort.insert(0, ("role_score", False))
    elif codes:  # pas de rôle : meilleur rôle du poste, sinon le tri favoriserait les moins chers
        df = df.join(best_role_scores(df, [r for r in roles.values() if set(r.positions) & set(codes)]))
        sort.insert(0, ("role_score", False))
    sort = [s for s in sort if s[0] in df]
    df = df.sort_values([c for c, _ in sort], ascending=[a for _, a in sort], kind="stable")
    return SearchResult(len(df), unknown, [to_record(r) for _, r in df.head(limit).iterrows()])


_OUTPUT = ("player_id", "name", "age", "positions", "club", "nationality", "fee_min", "fee_max",
           "fee_source", "wage_min", "wage_max", "contract_end", "ca", "pa", "ability_stars",
           "potential_stars", "role", "role_score", "role_coverage")


def to_record(row: pd.Series) -> dict[str, Any]:
    """JSON compact : les valeurs inconnues sont omises."""
    record: dict[str, Any] = {}
    for key in _OUTPUT:
        value = row.get(key)
        if isinstance(value, list):
            record[key] = value
        elif value is None or pd.isna(value):
            continue
        elif isinstance(value, pd.Timestamp):
            record[key] = value.date().isoformat()
        elif isinstance(value, (int, float)):
            number = float(value)
            whole = number.is_integer() or abs(number) >= 1000
            record[key] = round(number) if whole else round(number, 2)
        else:
            record[key] = str(value)
    return record

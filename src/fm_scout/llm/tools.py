"""Outils exposés au modèle : arguments validés, résultats issus des données seulement."""
import json
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from fm_scout.parsing import fold, parse_money, parse_wage, query_positions
from fm_scout.schema import POSITIONS, Dataset, ScoutError
from fm_scout.scouting.compare import compare, resolve
from fm_scout.scouting.recommend import recommend
from fm_scout.scouting.roles import Role, get_role, role_scores
from fm_scout.scouting.search import search_players
from fm_scout.scouting.squad import get_squad_needs


def _amount(value: Any) -> Any:
    """Accepte « 5M » : les petits modèles recopient volontiers le texte."""
    if isinstance(value, str) and (parsed := parse_money(value)):
        return parsed[1]
    return value


def _weekly(value: Any) -> Any:
    if isinstance(value, str) and (parsed := parse_wage(value)):
        return parsed[1]
    return value


def _position(value: Any) -> Any:
    if isinstance(value, str) and query_positions(value) is None:
        raise ValueError(f"poste inconnu ; valeurs : {', '.join(POSITIONS)}")
    return value


Money = Annotated[float, BeforeValidator(_amount), Field(ge=0)]
Wage = Annotated[float, BeforeValidator(_weekly), Field(ge=0)]
Position = Annotated[str, BeforeValidator(_position)]
Age = Annotated[int, Field(ge=14, le=50)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    position: Position | None = Field(None, description="Poste. GB=GK, DG=DL, DD=DR, MDC=DM, MOC=AMC, MG=ML, MD=MR, AG=AML, AD=AMR, BU=ST")
    min_age: Age | None = None
    max_age: Age | None = None
    max_fee: Money | None = Field(None, description="Prix maximum, ex. 5000000 pour 5M")
    max_wage: Wage | None = Field(None, description="Salaire max PAR SEMAINE ; si mensuel, écrire \"80K/mois\"")
    min_ca: float | None = None
    min_pa: float | None = None
    role: str | None = Field(None, description="Rôle tactique, pour noter et trier")
    contract_max_months: int | None = Field(None, ge=0, description="Contrat finissant dans N mois max")
    nationality: str | None = None
    free_agents_only: bool = Field(False, description="Uniquement les joueurs sans club")
    limit: int = Field(10, ge=1, le=20)


class NeedsArgs(_Args):
    expiring_months: int = Field(12, ge=0, le=60)


class RecommendArgs(_Args):
    season: int = Field(1, ge=0, le=2, description="0 saison en cours, 1 prochaine, 2 suivante")
    budget: Money | None = Field(None, description="Seulement si l'utilisateur donne un budget")


class CompareArgs(_Args):
    player_ids: list[str] = Field(min_length=2, max_length=6, description="Identifiants ou noms exacts")
    role: str | None = None


class RoleScoreArgs(_Args):
    player: str = Field(description="Identifiant ou nom exact")
    role: str


class Pick(_Args):
    player_id: str
    role: str | None = None
    justification: str = Field(description="2 lignes, uniquement des faits renvoyés par les outils")


class ShortlistArgs(_Args):
    picks: list[Pick] = Field(min_length=1, max_length=5)


_TOOLS: dict[str, tuple[type[_Args], str]] = {
    "search_players": (SearchArgs, "Cherche des joueurs à recruter (tous clubs, ou libres) : poste, âge, prix, contrat, rôle..."),
    "get_squad_needs": (NeedsArgs, "Analyse MON effectif uniquement, jamais pour trouver des recrues : profondeur, âge, fins de contrat, budget."),
    "recommend": (RecommendArgs, "Analyse mon plan d'effectif et conseille quels postes renforcer "
                  "et quelles recrues acheter selon mon budget."),
    "compare": (CompareArgs, "Compare des joueurs côte à côte."),
    "role_score": (RoleScoreArgs, "Note un joueur pour un rôle (0-100)."),
    "submit_shortlist": (ShortlistArgs, "Réponse finale : jusqu'à 5 joueurs trouvés par les outils."),
}


class InvalidCall(Exception):
    """Appel d'outil refusé ; le message est renvoyé au modèle."""


class ToolBox:
    def __init__(self, data: Dataset, roles: dict[str, Role], zones: dict[str, Any],
                 attributes: list[str], settings: dict[str, Any]):
        self._data = data
        self._roles = roles
        self._zones = zones
        self._attributes = attributes
        self._settings = settings
        self.report: dict[str, Any] | None = None  # dernière recommandation, affichée par le programme
        self.seen: dict[str, dict[str, Any]] = {}  # joueurs renvoyés par les outils
        self.facts: list[Any] = []  # autres résultats (effectif...)

    @property
    def all_names(self) -> set[str]:
        return set(self._data.players["name"])

    def schemas(self) -> list[dict[str, Any]]:
        result = []
        for name, (model, description) in _TOOLS.items():
            schema = model.model_json_schema()
            for part in [schema, *schema.get("$defs", {}).values()]:
                props = part.get("properties", {})
                if "role" in props:
                    props["role"] |= {"enum": list(self._roles)}
                if "position" in props:
                    props["position"] |= {"enum": list(POSITIONS)}
            result.append({"type": "function", "function": {
                "name": name, "description": description, "parameters": schema}})
        return result

    def parse(self, name: str, arguments: Any) -> _Args:
        if name not in _TOOLS:
            raise InvalidCall(f"Outil inconnu « {name} ». Outils : {', '.join(_TOOLS)}")
        try:
            raw = json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
            args = _TOOLS[name][0].model_validate(raw)
            for role in self._roles_in(args):
                get_role(self._roles, role)
        except json.JSONDecodeError as e:
            raise InvalidCall(f"Arguments JSON invalides : {e}") from e
        except ValidationError as e:
            details = "; ".join(f"{'.'.join(map(str, x['loc']))} : {x['msg']}" for x in e.errors())
            raise InvalidCall(f"Arguments invalides : {details}") from e
        except ScoutError as e:
            raise InvalidCall(str(e)) from e
        if isinstance(args, ShortlistArgs):
            unknown = [p.player_id for p in args.picks if p.player_id not in self.seen]
            if unknown:
                raise InvalidCall(f"player_id absents des résultats : {', '.join(unknown)}. "
                                  "Utilisez les player_id renvoyés par les outils.")
        return args

    @staticmethod
    def _roles_in(args: _Args) -> list[str]:
        items = args.picks if isinstance(args, ShortlistArgs) else [args]
        return [r for i in items if (r := getattr(i, "role", None))]

    def run(self, args: _Args) -> Any:
        try:
            match args:
                case SearchArgs():
                    result = search_players(self._data, self._roles, **args.model_dump()).to_dict()
                    self._remember(result["players"], args.role)
                    return result
                case NeedsArgs():
                    needs = get_squad_needs(self._data, self._roles, self._zones, args.expiring_months)
                    self.facts.append(needs)
                    return needs
                case RecommendArgs():
                    self.report = recommend(self._data, self._roles, self._settings,
                                            season=args.season, budget=args.budget)
                    for need in self.report["needs"]:
                        for c in need["candidates"]:
                            self.seen[c["player_id"]] = self.seen.get(c["player_id"], {}) | c
                    self.facts.append(self.report)
                    return compact(self.report)
                case CompareArgs():
                    records = compare(self._data, self._roles, args.player_ids, self._attributes, args.role)
                    for record in records:
                        self.seen[record["player_id"]] = self.seen.get(record["player_id"], {}) | record
                    return records
                case RoleScoreArgs():
                    record = compare(self._data, self._roles, [args.player], self._attributes, args.role)[0]
                    self.seen[record["player_id"]] = self.seen.get(record["player_id"], {}) | record
                    return record
        except ScoutError as e:
            raise InvalidCall(str(e)) from e
        raise InvalidCall("Outil non exécutable.")

    def score(self, player_id: str, role: str) -> dict[str, Any]:
        """Score calculé ici, jamais par le modèle."""
        players = self._data.players
        row = role_scores(players.loc[[resolve(players, player_id)]], get_role(self._roles, role)).iloc[0]
        return {k: float(v) for k, v in row.items() if v == v}

    def _remember(self, records: list[dict[str, Any]], role: str | None) -> None:
        """Ajoute les attributs du rôle, pour que le modèle ait des faits à citer."""
        players = self._data.players.set_index("player_id")
        for record in records:
            row = players.loc[record["player_id"]]
            if best := role or record.get("role"):
                names = get_role(self._roles, best).weights
                record["attributes"] = {a: float(row[a]) for a in names if a in row and row[a] == row[a]}
            self.seen[record["player_id"]] = self.seen.get(record["player_id"], {}) | record


def compact(report: dict[str, Any]) -> dict[str, Any]:
    """Version courte pour le modèle : ce qu'il doit commenter."""
    needs = []
    for n in report["needs"]:
        pick = next((c for c in n["candidates"] if c["player_id"] == n["pick"]), None)
        needs.append({k: n[k] for k in ("position", "reasons", "starter", "backup", "reference")} | {
            "recrue_conseillee": pick and {k: pick.get(k) for k in (
                "name", "age", "club", "for", "upgrade", "ca", "pa", "role", "role_score", "attributes")},
            "autres": [c["name"] for c in n["candidates"] if c is not pick]})
    return {"club": report["club"], "budget_M": round(report["budget"] / 1e6, 1),
            "depense_M": round(report["spent"] / 1e6, 1), "needs": needs}


def describe(name: str, args: _Args) -> str:
    shown = ", ".join(f"{k}={v}" for k, v in args.model_dump(exclude_unset=True).items())
    return f"{fold(name).replace(' ', '_')}({shown})"

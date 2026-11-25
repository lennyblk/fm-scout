"""12 requêtes types : le modèle appelle-t-il le bon outil avec les bons arguments ?

Usage : python eval_tools.py [modèle ...]   (défaut : modèle de config.toml)
Les données sont fictives ; seul le premier appel est jugé.
"""
import sys
import time
from pathlib import Path
from typing import Any

from fm_scout.config import load_config
from fm_scout.llm.agent import SYSTEM
from fm_scout.llm.client import OpenAICompatClient
from fm_scout.llm.tools import InvalidCall, ToolBox
from fm_scout.parsing import query_positions
from fm_scout.schema import ScoutError
from fm_scout.scouting.roles import build_roles
from fm_scout.sources.csv_source import CsvSource

ROOT = Path(__file__).parent
S = "search_players"
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("Il me faut un MDC de 21 ans max, 5M max", S, {"position": "DM", "max_age": 21, "max_fee": 5e6}),
    ("Trouve un avant-centre pivot à moins de 12M", S,
     {"position": "ST", "max_fee": 12e6, "role": "avant_centre_pivot"}),
    ("Un latéral droit entre 23 et 27 ans", S, {"position": "DR", "min_age": 23, "max_age": 27}),
    ("Des défenseurs centraux libres", S, {"position": "DC", "free_agents_only": True}),
    ("Un gardien dont le contrat finit dans 6 mois", S, {"position": "GK", "contract_max_months": 6}),
    ("Un ailier gauche payé 20K par semaine maximum", S, {"position": "AML", "max_wage": 20e3}),
    ("Un milieu central mezzala, 8M max, 24 ans max", S,
     {"position": "MC", "max_fee": 8e6, "max_age": 24, "role": "mezzala"}),
    ("Donne-moi 3 meneurs de jeu (MOC) de 20 ans max", S, {"position": "AMC", "max_age": 20, "limit": 3}),
    ("Quels contrats se terminent dans mon effectif ?", "get_squad_needs", {}),
    ("Regarde mon plan d'effectif et dis-moi quel poste renforcer", "recommend", {}),
    ("Propose-moi des recrues pour mon mercato avec 15M", "recommend", {"budget": 15e6}),
    ("Compare Léo Fictif et Max Cherón", "compare", {"player_ids": ["Léo Fictif", "Max Cherón"]}),
]
OPTIONAL = {"role", "limit"}  # ajout toléré par le prompt


def judge(args: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    problems = []
    for key, value in expected.items():
        got = args.get(key)
        if key == "position" and isinstance(got, str):
            got = (query_positions(got) or [got])[0]
        if key == "player_ids" and isinstance(got, list):
            got, value = sorted(got), sorted(value)
        if got != value:
            problems.append(f"{key}={got!r}, attendu {value!r}")
    problems += [f"{k} ajouté" for k in args if k not in expected and k not in OPTIONAL]
    return problems


def evaluate(model: str) -> None:
    config = load_config(ROOT / "config.toml")
    data = CsvSource(ROOT / "tests" / "fixtures", config.columns).load()
    box = ToolBox(data, build_roles(config.roles), config.zones, list(config.columns["attributes"]),
                  config.recommend)
    client = OpenAICompatClient(config.llm_url, model, config.llm_timeout, config.reasoning_effort)
    print(f"\n=== {model}")
    tools_ok = args_ok = 0
    start = time.perf_counter()
    for question, tool, expected in CASES:
        message = client.chat([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": question}], box.schemas())
        calls = message.get("tool_calls") or []
        if not calls:
            print(f"ÉCHEC  {question}\n       aucun appel d'outil")
            continue
        function = calls[0]["function"]
        problems = [] if function["name"] == tool else [f"outil {function['name']}, attendu {tool}"]
        tools_ok += not problems
        if not problems:
            try:
                args = box.parse(tool, function["arguments"])
                problems = judge(args.model_dump(exclude_defaults=True), expected)
            except InvalidCall as e:
                problems = [str(e)]
        args_ok += not problems
        print(f"{'ÉCHEC' if problems else 'OK   '}  {question}")
        for problem in problems:
            print(f"       {problem}")
    seconds = (time.perf_counter() - start) / len(CASES)
    print(f"Bon outil : {tools_ok}/{len(CASES)}   Outil + arguments : {args_ok}/{len(CASES)}   {seconds:.1f} s/requête")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        for name in sys.argv[1:] or [load_config(ROOT / "config.toml").model]:
            evaluate(name)
    except ScoutError as e:
        sys.exit(f"Erreur : {e}")

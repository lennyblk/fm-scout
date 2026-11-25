import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from fm_scout.config import Config, load_config
from fm_scout.llm.agent import Agent, Answer
from fm_scout.llm.client import OpenAICompatClient
from fm_scout.llm.tools import ToolBox
from fm_scout.parsing import parse_money, parse_wage
from fm_scout.schema import Dataset, ScoutError
from fm_scout.scouting.compare import compare
from fm_scout.scouting.recommend import recommend
from fm_scout.scouting.roles import build_roles
from fm_scout.scouting.search import SearchResult, search_players
from fm_scout.scouting.squad import get_squad_needs
from fm_scout.sources.base import DataSource
from fm_scout.sources.csv_source import CsvSource
from fm_scout.sources.fmsave_source import FmsaveSource


def _fee(text: str) -> float:
    if (value := parse_money(text)) is None:
        raise argparse.ArgumentTypeError(f"montant invalide : {text}")
    return value[1]


def _wage(text: str) -> float:
    if (value := parse_wage(text)) is None:
        raise argparse.ArgumentTypeError(f"salaire invalide : {text}")
    return value[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fm-scout")
    p.add_argument("--config", type=Path, default=Path("config.toml"))
    p.add_argument("--save", help="partie du nom ou chemin de la save (défaut : la plus récente)")
    p.add_argument("--csv", type=Path, help="fichier ou dossier d'exports, au lieu de la save")
    p.add_argument("--game-date", type=date.fromisoformat, help="AAAA-MM-JJ")
    p.add_argument("--no-llm", action="store_true",
                   help="commandes directes seulement")
    sub = p.add_subparsers(dest="command")
    a = sub.add_parser("ask", help="une question au scout (LLM local)")
    a.add_argument("question")
    sub.add_parser("inspect", help="ce qui a été lu et les colonnes non reconnues")
    sub.add_parser("roles", help="rôles disponibles")
    n = sub.add_parser("needs", help="analyser mon effectif")
    n.add_argument("--months", type=int, default=12, help="horizon des fins de contrat")
    n.add_argument("--json", action="store_true")
    r = sub.add_parser("recommend", help="quels postes renforcer, et avec qui")
    r.add_argument("--season", type=int, choices=(0, 1, 2), default=1, help="0 en cours, 1 prochaine, 2 suivante")
    r.add_argument("--budget", type=_fee, help="défaut : budget transferts de la partie")
    r.add_argument("--json", action="store_true")
    c = sub.add_parser("compare", help="comparer des joueurs (noms ou identifiants)")
    c.add_argument("players", nargs="+")
    c.add_argument("--role")
    c.add_argument("--json", action="store_true")
    s = sub.add_parser("search", help="chercher des joueurs")
    s.add_argument("--pos", dest="position")
    s.add_argument("--min-age", type=int)
    s.add_argument("--max-age", type=int)
    s.add_argument("--max-fee", type=_fee, help="ex. 5M")
    s.add_argument("--max-wage", type=_wage, help="hebdo, ex. 20K ; ou 80K/mois")
    s.add_argument("--min-ca", type=float)
    s.add_argument("--min-pa", type=float)
    s.add_argument("--role")
    s.add_argument("--contract-max-months", type=int)
    s.add_argument("--nationality")
    s.add_argument("--free", dest="free_agents_only", action="store_true")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--json", action="store_true")
    return p


_SYMBOL = {"EUR": "€", "GBP": "£", "USD": "$", None: ""}


def money(low: float | None, high: float | None, symbol: str) -> str:
    def short(x: float) -> str:
        for size, suffix in ((1e6, "M"), (1e3, "K")):
            if x >= size:
                return f"{x / size:.1f}".removesuffix(".0") + suffix
        return f"{x:.0f}"
    if low is None or high is None:
        return "?"
    return (short(low) if low == high else f"{short(low)}-{short(high)}") + symbol


def print_table(result: SearchResult, data: Dataset) -> None:
    sym = _SYMBOL[data.currency]
    rows = [("Nom", "Âge", "Postes", "Club", "Prix", "Source prix", "Sal./sem", "Contrat", "Rôle")]
    for p in result.players:
        score = f"{p.get('role', '')} {p['role_score']} ({p['role_coverage']:.0%})" if "role_score" in p else ""
        rows.append((p["name"], str(p.get("age", "?")), "/".join(p.get("positions", [])),
                     p.get("club", "?") or "Libre",
                     money(p.get("fee_min"), p.get("fee_max"), sym), p.get("fee_source", "?"),
                     money(p.get("wage_min"), p.get("wage_max"), sym),
                     p.get("contract_end", "?"), score))
    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    print(f"\n{result.total} joueur(s) correspondant(s).")
    for label, n in result.excluded_unknown.items():
        print(f"  {n} écarté(s) car donnée inconnue pour « {label} ».")


def print_inspect(data: Dataset) -> None:
    print("Fichiers :", *data.origin, sep="\n  ")
    print(f"Joueurs : {len(data.players)}")
    print(f"Mon club (détecté) : {data.my_club or 'inconnu, exportez la vue Effectif avec Club'}")
    print(f"Date du jeu (estimée) : {data.game_date or 'inconnue'}")
    print(f"Devise : {data.currency or 'inconnue'}")
    if data.budget:
        b, sym = data.budget, _SYMBOL[data.currency]
        print(f"Budget (relevé {b.month}) : transferts {money(b.transfer_remaining, b.transfer_remaining, sym)}, "
              f"salaires {money(b.wage_payroll_weekly, b.wage_payroll_weekly, sym)} utilisés sur "
              f"{money(b.wage_budget_weekly, b.wage_budget_weekly, sym)} par semaine")
    print("Colonnes lues (% rempli) :")
    for name, rate in data.players.notna().mean().items():
        if not str(name).endswith("_min"):
            print(f"  {name}: {rate:.0%}")
    print("Colonnes ignorées :", ", ".join(data.unmapped_columns) or "aucune")


def print_needs(needs: dict[str, Any]) -> None:
    print(f"{needs['club']} : {needs['players']} joueurs, âge moyen {needs['average_age']}")
    for z in needs["zones"]:
        best = ", ".join(f"{b['name']} {b['score']}" for b in z["best"])
        gap = f"  MANQUE {z['gap']}" if z["gap"] else ""
        print(f"  {z['zone']:<17} {z['depth']}/{z['target']}  âge {z['average_age'] or '-'}  {best}{gap}")
    print("Fins de contrat :")
    for p in needs["expiring"]:
        print(f"  {p['contract_end']}  {p['name']} ({p['age']} ans)")


def print_report(report: dict[str, Any], data: Dataset) -> None:
    sym = _SYMBOL[data.currency]

    def cash(x: float | None) -> str:
        return money(x, x, sym)

    def who(p: dict[str, Any] | None) -> str:
        return f"{p['name']} ({p['age']} ans, {p['score']})" if p else "personne"
    season = ("saison en cours", "saison prochaine", "dans deux saisons")[report["season"]]
    origin = "plan d'effectif" if report["from_planner"] else "effectif actuel (plan non lu)"
    print(f"{report['club']}, {season}, d'après ton {origin}. Budget {cash(report['budget'])}, "
          f"marge salariale {cash(report['wage_room'])}/sem")
    for i, n in enumerate(report["needs"], 1):
        print(f"\n{i}. {n['position']} : {', '.join(n['reasons'])}")
        print(f"   titulaire {who(n['starter'])}, doublure {who(n['backup'])}, référence {n['reference']}")
        if not n["candidates"]:
            print("   aucune recrue réaliste dans ce budget")
        for c in n["candidates"]:
            mark = "=>" if c["player_id"] == n["pick"] else "  "
            gain = f", {c['upgrade']:+}" if c["upgrade"] is not None else ""
            potential = f" | CA {c['ca']} PA {c['pa']}, potentiel +{c['growth']}" if "pa" in c else ""
            print(f"   {mark} {c['name']}, {c.get('age', '?')} ans, {c['club'] or 'Libre'} | "
                  f"~{cash(c['planned_fee'])}, {cash(c['wage_max'])}/sem | {c['role']} {c['role_score']}{gain} ({c['for']}){potential}")
    print(f"\nPlan d'achat (=>) : ~{cash(report['spent'])} sur {cash(report['budget'])}, marge de prix incluse.")
    print("Moins urgent : " + (", ".join(n["position"] for n in report["later"]) or "rien"))
    print("Rien à signaler : " + (", ".join(n["position"] for n in report["ok"]) or "aucun poste"))


def print_compare(records: list[dict[str, Any]], data: Dataset) -> None:
    sym = _SYMBOL[data.currency]

    def line(label: str, cells: list[str]) -> None:
        print(f"{label:<17}" + "".join(f"{c:<24}" for c in cells).rstrip())
    line("", [r["name"] for r in records])
    line("Âge", [str(r.get("age", "?")) for r in records])
    line("Club", [r.get("club", "?") or "Libre" for r in records])
    line("Postes", ["/".join(r.get("positions", [])) for r in records])
    line("Prix", [money(r.get("fee_min"), r.get("fee_max"), sym) for r in records])
    line("Source prix", [r.get("fee_source", "?") for r in records])
    line("Sal./sem", [money(r.get("wage_min"), r.get("wage_max"), sym) for r in records])
    line("Contrat", [r.get("contract_end", "?") for r in records])
    if any("role_score" in r for r in records):
        line("Score rôle", [str(r.get("role_score", "?")) for r in records])
    for name in dict.fromkeys(a for r in records for a in r["attributes"]):
        line(name, [f"{r['attributes'][name]:g}" if name in r["attributes"] else "?" for r in records])


def print_answer(answer: Answer, data: Dataset) -> None:
    sym = _SYMBOL[data.currency]
    if answer.report:
        print_report(answer.report, data)
        print()
    for i, pick in enumerate(answer.picks, 1):
        p = pick.player
        print(f"{i}. {p['name']}, {p.get('age', '?')} ans, {p.get('club', '?') or 'Libre'}, "
              f"{'/'.join(p.get('positions', []))}")
        print(f"   {money(p.get('fee_min'), p.get('fee_max'), sym)} ({p.get('fee_source', '?')}), "
              f"salaire {money(p.get('wage_min'), p.get('wage_max'), sym)}/sem, "
              f"rôle : {pick.role or 'non précisé'} {p.get('role_score', '')}")
        print(f"   {pick.justification}")
    if answer.text:
        print(answer.text)
    for warning in answer.warnings:
        print(f"Attention : {warning}")


class Session:
    """Discussion qui recharge les données quand une nouvelle sauvegarde apparaît."""

    def __init__(self, source: DataSource, build: Callable[[Dataset], Agent], data: Dataset, stamp: object):
        self._source = source
        self._build = build
        self._stamp = stamp
        self.data = data
        self.agent = build(data)

    def refresh(self) -> str | None:
        """Message à afficher si quelque chose a changé."""
        try:
            stamp = self._source.stamp()
            if stamp == self._stamp:
                return None
            data = self._source.load()
        except (ScoutError, OSError) as e:  # FM est peut-être en train d'écrire : on réessaiera
            return f"Nouvelle sauvegarde illisible pour l'instant, je garde l'ancienne ({e})"
        self._stamp, self.data = stamp, data
        self.agent = self._build(data)  # l'historique parlait des anciennes données
        return f"Sauvegarde rechargée ({data.my_club}, {data.game_date})."


def chat(session: Session) -> None:
    data = session.data
    print(f"Scout prêt ({data.my_club or 'club inconnu'}). Ligne vide ou Ctrl+C pour quitter.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not question:
            return
        try:
            if message := session.refresh():
                print(f"  [{message}]", file=sys.stderr)
            print_answer(session.agent.ask(question), session.data)
        except ScoutError as e:
            print(f"Erreur : {e}", file=sys.stderr)


def make_source(args: argparse.Namespace, config: Config) -> DataSource:
    if args.csv or (config.csv_path and not args.save):
        return CsvSource(args.csv or config.csv_path, config.columns)
    return FmsaveSource(args.save or config.save, config.money_rate, config.currency,
                        config.show_hidden)


def run(args: argparse.Namespace) -> None:
    if args.no_llm and args.command in (None, "ask"):
        raise ScoutError("--no-llm : utilisez search, needs, compare, inspect ou roles.")
    config = load_config(args.config)
    roles = build_roles(config.roles)
    if args.command == "roles":
        for role in roles.values():
            print(f"{role.name}: {', '.join(role.positions)}")
        return
    source = make_source(args, config)
    stamp = source.stamp()  # avant la lecture : une save faite pendant sera vue
    data = source.load()
    if args.game_date:
        data.game_date = args.game_date
    for warning in data.warnings:
        print(f"Attention : {warning}", file=sys.stderr)
    if args.command == "inspect":
        print_inspect(data)
        return
    if args.command in (None, "ask"):
        client = OpenAICompatClient(config.llm_url, config.model, config.llm_timeout,
                                    config.reasoning_effort)

        def build(current: Dataset) -> Agent:
            if args.game_date:
                current.game_date = args.game_date
            return Agent(client, ToolBox(current, roles, config.zones, list(config.columns["attributes"]),
                                      config.recommend),
                      on_call=lambda call: print(f"  [{call}]", file=sys.stderr))
        if args.command == "ask":
            print_answer(build(data).ask(args.question), data)
        else:
            chat(Session(source, build, data, stamp))
        return
    if args.command in ("needs", "compare", "recommend"):
        if args.command == "needs":
            output: Any = get_squad_needs(data, roles, config.zones, args.months)
        elif args.command == "recommend":
            output = recommend(data, roles, config.recommend, season=args.season, budget=args.budget)
        else:
            output = compare(data, roles, args.players, list(config.columns["attributes"]), args.role)
        if args.json:
            print(json.dumps(output, ensure_ascii=False, separators=(",", ":"), default=str))
        elif args.command == "needs":
            print_needs(output)
        elif args.command == "recommend":
            print_report(output, data)
        else:
            print_compare(output, data)
        return
    shared = ("command", "config", "csv", "save", "game_date", "no_llm", "json")
    filters = {k: v for k, v in vars(args).items() if k not in shared}
    result = search_players(data, roles, **filters)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ":")))
    else:
        print_table(result, data)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # console Windows en cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        run(build_parser().parse_args())
    except ScoutError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

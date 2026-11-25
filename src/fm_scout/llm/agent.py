import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fm_scout.llm.client import LLMClient, Message
from fm_scout.llm.tools import InvalidCall, ShortlistArgs, ToolBox, compact, describe
from fm_scout.llm.verify import unknown_players, unsupported_numbers
from fm_scout.schema import ScoutError

MAX_RETRIES = 2
MAX_STEPS = 8
KEPT_TURNS = 3

SYSTEM = """Tu es recruteur pour un club de football. Réponds en français.
- Pour trouver des joueurs, appelle search_players avec TOUS les critères demandés, et seulement eux.
- Si un poste est cité ou sous-entendu, remplis toujours position, même avec un rôle.
- Si un nombre de joueurs est demandé, remplis limit.
- N'ajoute jamais de critère que l'utilisateur n'a pas donné. Tu peux choisir un rôle adapté au poste.
- Pour savoir quels postes renforcer, qui recruter ou des conseils de mercato : appelle recommend.
  Le tableau est déjà affiché à l'utilisateur : commente-le en 6 phrases au plus, sans tableau,
  en expliquant pour chaque poste pourquoi le renforcer et ce que la recrue apporte.
- Pour la simple profondeur ou les fins de contrat de l'effectif, appelle get_squad_needs.
- Termine une recherche par submit_shortlist avec 5 joueurs au plus, pris dans les résultats.
- Justification : 2 phrases sur ses points forts et faibles d'après attributes ; ne répète pas
  le nom, l'âge, le club ni le prix, déjà affichés.
- Interdit d'inventer un joueur ou une statistique : cite uniquement ce que les outils ont renvoyé.
- ca = niveau actuel, pa = potentiel (0 à 200) : parle du potentiel d'un jeune s'il est connu.
- Si aucun joueur ne correspond, dis-le.
Postes : GK gardien, DC central, DL/DR latéral gauche/droit, WBL/WBR piston gauche/droit,
DM milieu défensif (MDC), MC milieu central, ML/MR milieu gauche/droit, AMC meneur (MOC),
AML/AMR ailier gauche/droit, ST avant-centre (BU)."""


@dataclass
class PickedPlayer:
    player: dict[str, Any]
    role: str | None
    justification: str


@dataclass
class Answer:
    picks: list[PickedPlayer] = field(default_factory=list)
    text: str = ""
    warnings: list[str] = field(default_factory=list)
    report: dict[str, Any] | None = None


class Agent:
    def __init__(self, client: LLMClient, tools: ToolBox,
                 on_call: Callable[[str], None] = lambda call: None):
        self._client = client
        self._tools = tools
        self._on_call = on_call
        self._turns: list[list[Message]] = []

    def ask(self, question: str) -> Answer:
        turn: list[Message] = [{"role": "user", "content": question}]
        self._tools.report = None
        answer = self._loop(turn)
        answer.report = self._tools.report
        self._turns = [*self._turns, turn][-KEPT_TURNS:]
        return answer

    def _loop(self, turn: list[Message]) -> Answer:
        retries = 0
        past = [m for t in self._turns for m in t]
        for _ in range(MAX_STEPS):
            message = self._client.chat(
                [{"role": "system", "content": SYSTEM}, *past, *turn], self._tools.schemas())
            calls = message.get("tool_calls") or []
            kept: Message = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                kept["tool_calls"] = calls
            turn.append(kept)
            if not calls:
                return self._text_answer(kept["content"])
            answer: Answer | None = None
            failed = False
            for call in calls:
                content, submitted, ok = self._execute(call)
                answer = answer or submitted
                failed = failed or not ok
                turn.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
            if answer:
                return answer
            retries = retries + 1 if failed else 0
            if retries > MAX_RETRIES:
                raise ScoutError("Le modèle n'arrive pas à appeler les outils correctement. "
                                 "Reformulez, ou utilisez les commandes --no-llm.")
        raise ScoutError("Le modèle tourne en rond. Reformulez, ou utilisez --no-llm.")

    def _execute(self, call: dict[str, Any]) -> tuple[str, Answer | None, bool]:
        function = call.get("function") or {}
        name = function.get("name", "")
        try:
            args = self._tools.parse(name, function.get("arguments"))
            if isinstance(args, ShortlistArgs):
                return "Liste affichée.", self._shortlist(args), True
            self._on_call(describe(name, args))
            result = self._tools.run(args)
        except InvalidCall as e:
            self._on_call(f"appel refusé : {e}")
            return f"ERREUR : {e}", None, False
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str), None, True

    def _shortlist(self, args: ShortlistArgs) -> Answer:
        answer = Answer()
        for pick in args.picks:
            player = dict(self._tools.seen[pick.player_id])
            if pick.role:
                player |= self._tools.score(pick.player_id, pick.role)
            answer.picks.append(PickedPlayer(player, pick.role or player.get("role"), pick.justification.strip()))
            answer.warnings += self._check(pick.justification, [player], player["name"] + " : ")
        return answer

    def _text_answer(self, content: str) -> Answer:
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        facts = [*self._tools.seen.values(), *self._tools.facts]
        if self._tools.report:
            facts.append(compact(self._tools.report))
        answer = Answer(text=text or "(réponse vide du modèle)", warnings=self._check(text, facts))
        if not facts:
            answer.warnings.append("Réponse donnée sans consulter les données de la partie.")
        return answer

    def _check(self, text: str, facts: list[Any], prefix: str = "") -> list[str]:
        warnings = []
        if names := unknown_players(text, facts, self._tools.all_names):
            warnings.append(f"{prefix}nom non vérifié : {', '.join(names)}")
        if numbers := unsupported_numbers(text, facts):
            warnings.append(f"{prefix}nombre absent des données : {', '.join(numbers)}")
        return warnings

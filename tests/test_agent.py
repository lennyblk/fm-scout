import json
from pathlib import Path

import pytest

from fm_scout.config import load_config
from fm_scout.llm.agent import Agent
from fm_scout.llm.tools import InvalidCall, SearchArgs, ToolBox
from fm_scout.llm.verify import unknown_players, unsupported_numbers
from fm_scout.schema import ScoutError
from fm_scout.scouting.roles import build_roles
from fm_scout.sources.csv_source import CsvSource

CONFIG = load_config(Path("absent.toml"))


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append(list(messages))
        return self.replies.pop(0)


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def text(content):
    return {"role": "assistant", "content": content}


def pick(pid, why="Solide.", **kw):
    return call("submit_shortlist", picks=[{"player_id": pid, "justification": why, **kw}])


SEARCH = call("search_players", position="MDC", max_age=21, max_fee="5M", role="sentinelle")


@pytest.fixture
def box():
    data = CsvSource(Path(__file__).parent / "fixtures", CONFIG.columns).load()
    return ToolBox(data, build_roles(CONFIG.roles), CONFIG.zones, list(CONFIG.columns["attributes"]),
                   CONFIG.recommend)


def test_full_loop(box):
    answer = Agent(FakeClient(SEARCH, pick("1", "Tacle 14, 20 ans.", role="sentinelle")), box).ask("MDC")
    assert answer.picks[0].player["name"] == "Léo Fictif"
    assert answer.picks[0].player["role_score"] > 0 and answer.warnings == []


def test_invented_number_and_name_flagged(box):
    answer = Agent(FakeClient(SEARCH, pick("1", "Passe 19, mieux que Zinedine Inventé.")), box).ask("x")
    joined = " ".join(answer.warnings)
    assert "19" in joined and "Zinedine Inventé (introuvable)" in joined


def test_unseen_player_is_refused_then_retried(box):
    client = FakeClient(SEARCH, pick("2"), pick("1"))  # 2 existe mais hors résultats
    assert Agent(client, box).ask("x").picks[0].player["player_id"] == "1"
    assert "absents des résultats" in client.requests[-1][-1]["content"]


def test_gives_up_after_two_retries(box):
    client = FakeClient(*[call("outil_bidon") for _ in range(5)])
    with pytest.raises(ScoutError):
        Agent(client, box).ask("x")
    assert len(client.requests) == 3  # 1 essai + 2 relances


def test_text_without_data_is_flagged(box):
    answer = Agent(FakeClient(text("Prends Kylian Mbappé.")), box).ask("x")
    assert len(answer.warnings) == 2


def test_infinite_loop_stops(box):
    with pytest.raises(ScoutError, match="rond"):
        Agent(FakeClient(*[SEARCH] * 20), box).ask("x")


def test_history_kept_whole_turns(box):
    client = FakeClient(*[text("ok")] * 6)
    agent = Agent(client, box)
    for i in range(6):
        agent.ask(f"q{i}")
    assert [m["content"] for m in client.requests[-1] if m["role"] == "user"] == ["q2", "q3", "q4", "q5"]


@pytest.mark.parametrize("name,args", [
    ("search_players", '{"position":"ZZZ"}'), ("search_players", '{"foo":1}'),
    ("search_players", "{pas du json"), ("search_players", '{"role":"libero_magique"}'),
    ("search_players", '{"max_age":210}'), ("nope", "{}"), ("compare", '{"player_ids":["1"]}')])
def test_invalid_calls(box, name, args):
    with pytest.raises(InvalidCall):
        box.parse(name, args)


def test_amount_text(box):
    args = box.parse("search_players", '{"max_fee":"5M","max_wage":"52K/mois"}')
    assert isinstance(args, SearchArgs) and args.max_fee == 5e6 and args.max_wage == 12000


def test_missing_column_goes_back_to_model(box):
    with pytest.raises(InvalidCall, match="ca"):
        box.run(box.parse("search_players", '{"min_ca":100}'))


def test_recommend_attaches_report(box, monkeypatch):
    pick_ = {"player_id": "1", "name": "Léo Fictif", "club": "FC Exemple", "age": 20, "role_score": 62.9}
    report = {"club": "Mon Club", "budget": 10e6, "spent": 4e6, "needs": [
        {"position": "DM", "reasons": ["pas de doublure"], "starter": None, "backup": None,
         "reference": 60.0, "candidates": [pick_], "pick": "1"}]}
    monkeypatch.setattr("fm_scout.llm.tools.recommend", lambda *a, **k: report)
    agent = Agent(FakeClient(call("recommend"), text("Léo Fictif, 62,9 : renforce le DM pour 4M."),
                             text("ok")), box)
    answer = agent.ask("qui recruter ?")
    assert answer.report is report and answer.warnings == []
    assert agent.ask("merci").report is None


def test_verify():
    facts = [{"name": "Léo Fictif", "club": "FC Exemple", "fee_max": 4.5e6, "wage_max": 12000, "passing": 13.0}]
    assert unsupported_numbers("4,5M, 12K, 52K par mois, passe 13/20", facts) == []
    assert unsupported_numbers("passe 14, 9M", facts) == ["14", "9M"]
    assert unknown_players("Léo Fictif du FC Exemple", facts, set()) == []

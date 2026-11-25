from fm_scout.cli import Session
from fm_scout.schema import Dataset, ScoutError


class FakeSource:
    def __init__(self):
        self.version = 1
        self.broken = False

    def stamp(self):
        return self.version

    def load(self):
        if self.broken:
            raise ScoutError("en cours d'écriture")
        return Dataset(players=None, my_club=f"v{self.version}")


def make():
    source = FakeSource()
    return source, Session(source, lambda data: data.my_club, source.load(), source.stamp())


def test_no_change():
    _, session = make()
    assert session.refresh() is None and session.agent == "v1"


def test_new_save_is_reloaded_once():
    source, session = make()
    source.version = 2
    assert "rechargée" in session.refresh() and session.agent == "v2"
    assert session.refresh() is None


def test_unreadable_save_keeps_old_data_and_retries():
    source, session = make()
    source.version, source.broken = 2, True
    assert "illisible" in session.refresh() and session.agent == "v1"
    source.broken = False
    assert "rechargée" in session.refresh() and session.agent == "v2"

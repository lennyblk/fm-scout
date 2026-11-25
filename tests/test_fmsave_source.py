import os

import pytest

from fm_scout.schema import ScoutError
from fm_scout.sources.fmsave_source import FmsaveSource, find_save


@pytest.fixture
def games(tmp_path):
    names = ["Moi - Ajax", "Moi - Como", "last save overwrite backup"]
    for i, name in enumerate(names):
        f = tmp_path / f"{name}.fm"
        f.write_bytes(b"pas une save")
        os.utime(f, (1000 + i, 1000 + i))
    return tmp_path


def test_latest_ignores_backup(games):
    assert find_save(None, [games]).stem == "Moi - Como"


def test_by_name(games):
    assert find_save("ajax", [games]).stem == "Moi - Ajax"


def test_by_path(games):
    path = games / "Moi - Ajax.fm"
    assert find_save(str(path), []) == path


def test_not_found(games):
    with pytest.raises(ScoutError, match="Moi - Como"):
        find_save("psg", [games])


def test_unreadable_save_is_a_clear_error(games):
    pytest.importorskip("fmsave")
    with pytest.raises(ScoutError, match="fmsave ne peut pas lire"):
        FmsaveSource("ajax", folders=[games]).load()

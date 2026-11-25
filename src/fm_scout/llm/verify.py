"""Contrôle a posteriori du texte écrit par le modèle."""
import re
from typing import Any

from fm_scout.parsing import fold

_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)\s*(Md|M|K|k|m)?(?![A-Za-zÀ-ÿ\d])")
_PROPER = re.compile(r"[A-ZÀ-Ý][\wÀ-ÿ'’.-]+(?: [A-ZÀ-Ý][\wÀ-ÿ'’.-]+)+")
_SCALE = {"": 1.0, "k": 1e3, "m": 1e6, "md": 1e9}


def _values(data: Any) -> tuple[set[float], set[str]]:
    """Nombres et textes présents dans les résultats d'outils."""
    if isinstance(data, dict):
        parts = [_values(v) for v in data.values()]
    elif isinstance(data, (list, tuple)):
        parts = [_values(v) for v in data]
    elif isinstance(data, bool) or data is None:
        return set(), set()
    elif isinstance(data, (int, float)):
        return {float(data)}, set()
    else:
        return {float(n) for n in re.findall(r"\d+", str(data))}, {str(data)}
    return (set().union(*(p[0] for p in parts)) if parts else set(),
            set().union(*(p[1] for p in parts)) if parts else set())


def _close(x: float, known: set[float]) -> bool:
    monthly = {k * 52 / 12 for k in known if k > 100}
    return any(abs(x - k) <= max(0.5 if k > 20 else 0.05, 0.03 * abs(k) if k > 100 else 0)
               for k in known | monthly)


def unsupported_numbers(text: str, data: Any) -> list[str]:
    known, texts = _values(data)
    for name in texts:  # un nom de club peut contenir un chiffre
        if any(c.isalpha() for c in name):  # pas les identifiants : « 1 » dans « 14 »
            text = text.replace(name, " ")
    bad = []
    for number, suffix in _NUMBER.findall(text):
        value = float(number.replace(",", ".")) * _SCALE[suffix.lower()]
        if not _close(value, known | {1.0, 20.0, 100.0}):  # « 15 sur 20 », « 70/100 »
            bad.append(number + suffix)
    return bad


def unknown_players(text: str, data: Any, all_names: set[str]) -> list[str]:
    """Noms de joueurs absents des résultats : existants dans la base, ou inventés."""
    texts = [fold(t) for t in _values(data)[1]]
    folded_names = {fold(n) for n in all_names}
    bad = []
    for match in _PROPER.findall(text):
        match = match.rstrip(".'’-")
        key = fold(match)
        if any(key in t or (len(t) > 3 and t in key) for t in texts):
            continue
        bad.append(f"{match} ({'existe, mais hors résultats' if key in folded_names else 'introuvable'})")
    return list(dict.fromkeys(bad))

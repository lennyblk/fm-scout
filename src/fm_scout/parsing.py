"""Conversion des textes affichés par FM (EN et FR) en valeurs typées."""
import re
import unicodedata
from datetime import datetime

from fm_scout.schema import POSITIONS

Range = tuple[float, float]
EMPTY = {"", "-", "--", "—", "–", "n/a", "na", "?", "unknown", "inconnu", "inconnue"}
FREE_CLUB = {"", "-", "—", "–", "free agent", "free", "libre", "sans club", "aucun", "none"}


def fold(text: str) -> str:
    """Minuscules, sans accents, ponctuation réduite à des espaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9#%]+", " ", text.casefold()).strip()


def is_empty(text: str) -> bool:
    return text.strip().casefold() in EMPTY


def _number(text: str, scaled: bool) -> float:
    """scaled : suivi de K/M, donc un séparateur unique est décimal."""
    text = re.sub(r"\s", "", text)
    seps = [c for c in text if c in ".,"]
    if not seps:
        return float(text)
    if len(set(seps)) == 2:
        decimal = seps[-1]
        other = "," if decimal == "." else "."
        return float(text.replace(other, "").replace(decimal, "."))
    three_digits = len(text) - text.index(seps[0]) == 4
    if len(seps) > 1 or (three_digits and not scaled):
        return float(text.replace(seps[0], ""))
    return float(text.replace(seps[0], "."))


_SCALE = {"": 1, "k": 1e3, "m": 1e6, "md": 1e9, "b": 1e9, "bn": 1e9}
_AMOUNT = re.compile(r"(\d[\d.,\s]*)\s*(md|bn|b|m|k)?(?![a-z])")
_CURRENCY = re.compile(r"[€£$]|\beur\b|\bgbp\b|\busd\b")
_PERIOD = re.compile(
    r"(?:p\s*/|/|\bpar\b|\bper\b)\s*"
    r"(semaine|week|sem|mois|month|annee|année|year|an|w|s|m|a|y)\b.*$")


def parse_money(text: str) -> Range | None:
    """(min, max), ou None si inconnu / illisible."""
    if is_empty(text):
        return None
    low = text.casefold()
    if low.strip() in {"free", "gratuit", "free transfer"}:
        return (0.0, 0.0)
    low, symbols = _CURRENCY.subn("", _PERIOD.sub("", low))
    found = _AMOUNT.findall(low)
    if not 1 <= len(found) <= 2:
        return None
    # « 4 - 6M€ » : le 4 hérite du M ; mais pas dans « €500 - €1K ».
    inherited = found[-1][1] if symbols < 2 else ""
    try:
        values = [_number(n.strip().rstrip(".,"), bool(s or inherited)) * _SCALE[s or inherited]
                  for n, s in found]
    except ValueError:
        return None
    return (min(values), max(values))


def parse_wage(text: str) -> Range | None:
    """Comme parse_money, ramené à la semaine (hebdo si période absente)."""
    amount = parse_money(text)
    period = _PERIOD.search(text.casefold())
    if amount is None or period is None:
        return amount
    unit = period.group(1)
    per_year = (12 if unit in {"m", "mois", "month"}
                else 1 if unit in {"a", "an", "annee", "année", "y", "year"} else 52)
    return (amount[0] * per_year / 52, amount[1] * per_year / 52)


def parse_currency(text: str) -> str | None:
    low = text.casefold()
    for code, marks in {"EUR": ("€", "eur"), "GBP": ("£", "gbp"), "USD": ("$", "usd")}.items():
        if any(m in low for m in marks):
            return code
    return None


def parse_range(text: str) -> Range | None:
    """Attribut « 14 » ou « 10-14 »."""
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:[-–—]\s*(\d+(?:[.,]\d+)?))?\s*", text)
    if not match:
        return None
    low = float(match[1].replace(",", "."))
    high = float(match[2].replace(",", ".")) if match[2] else low
    return (min(low, high), max(low, high))


_MONTHS = {m: i for i, names in enumerate((
    "jan janv janvier january", "feb fev fevr fevrier february", "mar mars march",
    "apr avr avril april", "may mai", "jun juin june", "jul juil juillet july",
    "aug aou aout august", "sep sept septembre september", "oct octobre october",
    "nov novembre november", "dec decembre december"), 1) for m in names.split()}


def parse_date(text: str) -> datetime | None:
    """Jour en premier, sauf format ISO."""
    folded = fold(text)
    try:
        if m := re.match(r"(\d{4}) (\d{1,2}) (\d{1,2})\b", folded):
            return datetime(int(m[1]), int(m[2]), int(m[3]))
        if m := re.match(r"(\d{1,2}) (\w+) (\d{4})\b", folded):
            month = int(m[2]) if m[2].isdigit() else _MONTHS.get(m[2], 0)
            return datetime(int(m[3]), month, int(m[1]))
    except ValueError:
        pass
    return None


# Lettres telles qu'affichées dans FM (hors parenthèses).
_BASES = {"GK": "GK", "GB": "GK", "G": "GK", "D": "D", "WB": "WB", "DL": "WB", "AL": "WB",
          "DLO": "WB", "DM": "DM", "MD": "DM", "MDC": "DM", "M": "M", "AM": "AM", "MO": "AM",
          "ST": "ST", "BT": "ST", "BU": "ST", "AT": "ST", "S": "ST"}
_SIDES = {"L": "L", "G": "L", "R": "R", "D": "R", "C": "C"}


def parse_positions(text: str) -> list[str]:
    """« D/WB (RL), DM » -> [DL, DR, WBL, WBR, DM]. Codes inconnus ignorés."""
    result: list[str] = []
    for chunk in text.upper().split(","):
        match = re.fullmatch(r"\s*([A-Z/ ]+?)\s*(?:\(([A-Z/ ]+)\))?\s*", chunk)
        if not match:
            continue
        sides = [_SIDES[c] for c in match[2] or "" if c in _SIDES]
        for token in match[1].split("/"):
            token = "ST" if token.strip() in {"STC", "SC"} else token.strip()
            if match[2] is None and token in POSITIONS:  # « MC », « AML »
                result += [token] if token not in result else []
                continue
            base = _BASES.get(token)
            if base is None:
                continue
            codes = [base] if base in POSITIONS else [base + s for s in sides or ["C"]]
            result += [c for c in codes if c in POSITIONS and c not in result]
    return sorted(result, key=POSITIONS.index)


# Saisie utilisateur : le sens français courant prime (MD = milieu droit).
_QUERY = {"GB": ["GK"], "G": ["GK"], "GARDIEN": ["GK"], "DG": ["DL"], "DD": ["DR"],
          "ARG": ["DL"], "ARD": ["DR"], "LATERAL": ["DL", "DR"], "PG": ["WBL"], "PD": ["WBR"],
          "MDC": ["DM"], "MG": ["ML"], "MD": ["MR"], "MOC": ["AMC"], "MOG": ["AML"],
          "MOD": ["AMR"], "AILIER": ["AML", "AMR", "ML", "MR"], "AG": ["AML"], "AD": ["AMR"],
          "BU": ["ST"], "BT": ["ST"], "AT": ["ST"], "AC": ["ST"]}


def query_positions(text: str) -> list[str] | None:
    key = fold(text).upper()
    return [key] if key in POSITIONS else _QUERY.get(key)

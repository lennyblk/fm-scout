import csv
import io
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fm_scout.parsing import (FREE_CLUB, Range, fold, is_empty, parse_currency, parse_date,
                              parse_money, parse_positions, parse_range, parse_wage)
from fm_scout.schema import Dataset, ScoutError

Mapping = dict[str, dict[str, list[str]]]
_SQUAD_NAME = re.compile(r"squad|effectif", re.I)
_STAMP = re.compile(r"_\d{8}_\d{6}$")
_TEXT = {"name", "player_id", "nationality", "club", "positions"}
_DATES = {"birth_date", "contract_end"}


def default_dirs() -> list[Path]:
    home = Path.home()
    docs = [home / "Documents", home / "OneDrive" / "Documents"]
    return [d / "Sports Interactive" / "FM View Exporter" for d in docs] + [
        home / "Sports Interactive" / "Football Manager 27" / "FM26PlayerExport by vinteset"
        / "Exports CSV"]


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Détecte encodage et séparateur. Les en-têtes répétés deviennent « X#2 »."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
    first_line = text.split("\n", 1)[0]
    delimiter = max(";,\t|", key=first_line.count)
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if not rows:
        raise ScoutError(f"Fichier vide : {path}")
    seen: Counter[str] = Counter()
    headers = []
    for header in rows[0]:
        key = fold(header)
        seen[key] += 1
        headers.append(header.strip() if seen[key] == 1 else f"{header.strip()}#{seen[key]}")
    body = [(r + [""] * len(headers))[:len(headers)] for r in rows[1:] if any(c.strip() for c in r)]
    return headers, body


def _add_range(out: dict[str, list[Any]], field: str, ranges: list[Range | None]) -> None:
    out[f"{field}_min"] = [None if r is None else r[0] for r in ranges]
    out[f"{field}_max"] = [None if r is None else r[1] for r in ranges]


def _middle(ranges: list[Range | None]) -> list[float | None]:
    return [None if r is None else sum(r) / 2 for r in ranges]


class CsvSource:
    def __init__(self, path: Path | None, mapping: Mapping):
        self._path = path
        self._mapping = mapping

    def stamp(self) -> object:
        return [(f, f.stat().st_mtime_ns) for f in self._files()]

    def load(self) -> Dataset:
        files = self._files()
        data = Dataset(players=pd.DataFrame(), origin=[str(f) for f in files])
        frames: list[pd.DataFrame] = []
        currencies: set[str] = set()
        for file in files:  # du plus récent au plus ancien
            frame = self._load_file(file, data, currencies)
            frames.append(frame)
            if data.my_club is None and _SQUAD_NAME.search(file.stem) and "club" in frame:
                clubs = frame.loc[frame["club"] != "", "club"]
                # Majoritaire : des joueurs prêtés peuvent afficher un autre club.
                data.my_club = None if clubs.empty else str(clubs.mode().iloc[0])
        if len(currencies) > 1:
            raise ScoutError(f"Devises mélangées ({', '.join(sorted(currencies))}) : refaites les "
                             "exports avec une seule devise.")
        data.currency = next(iter(currencies), None)
        times = [f.stat().st_mtime for f in files]
        if max(times) - min(times) > 86400:
            data.warnings.append("Exports espacés de plus de 24 h : vérifiez qu'ils viennent de la "
                                 "même partie.")
        # Joueur présent dans plusieurs exports : première valeur connue, la plus récente.
        data.players = (pd.concat(frames, ignore_index=True)
                        .groupby("player_id", sort=False).first().reset_index())
        data.game_date = infer_game_date(data.players)
        return data

    def _files(self) -> list[Path]:
        """Le dernier export de chaque vue."""
        roots = [self._path] if self._path else default_dirs()
        for root in roots:
            if root.is_file():
                return [root]
            if root.is_dir():
                latest: dict[str, Path] = {}
                for f in sorted(root.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
                    latest.setdefault(_STAMP.sub("", f.stem), f)
                if latest:
                    return list(latest.values())
        raise ScoutError("Aucun export CSV trouvé dans :\n  " + "\n  ".join(map(str, roots))
                         + "\nExportez une liste depuis FM27 (F9) ou indiquez --csv.")

    def _columns(self, headers: list[str]) -> dict[str, int]:
        index = {fold(h): i for i, h in enumerate(headers)}
        columns: dict[str, int] = {}
        for group in self._mapping.values():
            for field, aliases in group.items():
                match = next((index[fold(a)] for a in aliases if fold(a) in index), None)
                if match is not None:
                    columns[field] = match
        return columns

    def _load_file(self, file: Path, data: Dataset, currencies: set[str]) -> pd.DataFrame:
        headers, rows = read_table(file)
        columns = self._columns(headers)
        if "name" not in columns:
            raise ScoutError(f"Colonne du nom introuvable dans {file}.\nEn-têtes : "
                             f"{', '.join(headers)}\nComplétez columns.yaml.")
        used = set(columns.values())
        data.unmapped_columns += [h for i, h in enumerate(headers)
                                  if i not in used and h not in data.unmapped_columns]
        cells = {f: [row[i].strip() for row in rows] for f, i in columns.items()}

        out: dict[str, list[Any]] = {"name": cells["name"]}
        if "nationality" in cells:
            out["nationality"] = [None if is_empty(c) else c for c in cells["nationality"]]
        if "club" in cells:
            out["club"] = ["" if c.casefold() in FREE_CLUB else c for c in cells["club"]]
        if "position" in cells:
            raw = cells["position"]
            out["positions"] = [parse_positions(c) for c in raw]
            bad = sorted({c for c, p in zip(raw, out["positions"]) if not p and not is_empty(c)})
            if bad:
                data.warnings.append(f"Postes non reconnus : {', '.join(bad[:5])}")
        for field in _DATES & cells.keys():
            out[field] = [parse_date(c) for c in cells[field]]
        for field in ("value", "asking_price", "wage"):
            if field in cells:
                currencies.update(c for c in map(parse_currency, cells[field]) if c)
                parser = parse_wage if field == "wage" else parse_money
                _add_range(out, field, [parser(c) for c in cells[field]])
        for field in ("age", "ca", "pa", "ability_stars", "potential_stars"):
            if field in cells:
                out[field] = _middle([parse_range(c) for c in cells[field]])
        for field in self._mapping["attributes"]:
            if field in cells:
                ranges = [parse_range(c) for c in cells[field]]
                _add_range(out, field, ranges)
                out[field] = _middle(ranges)

        frame = pd.DataFrame(out)
        for col in frame.columns:
            if col in _DATES:
                frame[col] = pd.to_datetime(frame[col])
            elif col not in _TEXT:
                frame[col] = pd.to_numeric(frame[col]).astype(float)
        ids = cells.get("player_id", [""] * len(rows))
        frame["player_id"] = [f"{file.stem}:{i}" if is_empty(uid) else uid
                              for i, uid in enumerate(ids)]
        return frame


def infer_game_date(players: pd.DataFrame) -> date | None:
    """Le dernier anniversaire fêté donne la date du jeu, à quelques jours près."""
    if not {"birth_date", "age"} <= set(players.columns):
        return None
    latest: datetime | None = None
    for born, age in players[["birth_date", "age"]].dropna().itertuples(index=False):
        day = 28 if (born.month, born.day) == (2, 29) else born.day
        birthday = datetime(born.year + int(age), born.month, day)
        latest = max(latest, birthday) if latest else birthday
    return latest.date() if latest else None

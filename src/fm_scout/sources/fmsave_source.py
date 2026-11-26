"""Lit une sauvegarde .fm avec fmsave, sur une copie temporaire."""
import shutil
import tempfile
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from fm_scout.parsing import fold
from fm_scout.schema import Budget, Dataset, ScoutError

# Attributs que le jeu n'affiche jamais : pas lus.
_HIDDEN = {"consistency", "dirtiness", "important_matches", "injury_proneness", "versatility"}
# Cases du plan d'effectif, déduites des postes des joueurs d'une vraie save FM27
# (23 et 27 : inconnues, vides ; 2 : libéro).
_PLANNER = {1: "GK", 3: "DR", 4: "DC", 5: "DC", 6: "DC", 7: "DL", 8: "WBR", 9: "DM", 10: "DM", 11: "DM",
            12: "WBL", 13: "MR", 14: "MC", 15: "MC", 16: "MC", 17: "ML", 18: "AMR", 19: "AMC",
            20: "AMC", 21: "AMC", 22: "AML", 24: "ST", 25: "ST", 26: "ST"}
# Bits de poste d'une tactique (5 et 6 déduits de l'ordre, non vérifiés).
_TACTIC = {0: "GK", 2: "DR", 3: "DL", 4: "DC", 5: "WBR", 6: "WBL", 7: "DM", 8: "MR", 9: "ML", 10: "MC",
           11: "AMR", 12: "AML", 13: "AMC", 14: "ST"}
_BACKUP = "last save overwrite backup"


def default_games_dirs() -> list[Path]:
    home = Path.home()
    tail = Path("Sports Interactive") / "Football Manager 27" / "games"
    return [home / "Documents" / tail, home / "OneDrive" / "Documents" / tail,
            home / "Library" / "Application Support" / tail]


def find_save(wanted: str | None, folders: list[Path]) -> Path:
    """Chemin exact, ou save la plus récente dont le nom contient `wanted`."""
    if wanted and (path := Path(wanted).expanduser()).is_file():
        return path
    saves = [f for d in folders if d.is_dir() for f in d.glob("*.fm") if f.stem != _BACKUP]
    matches = [f for f in saves if not wanted or fold(wanted) in fold(f.stem)]
    if not matches:
        listing = ", ".join(f.stem for f in saves) or "aucune"
        raise ScoutError(f"Sauvegarde introuvable{f' pour « {wanted} »' if wanted else ''}. "
                         f"Saves vues : {listing}. Indiquez --save.")
    return max(matches, key=lambda f: f.stat().st_mtime)


class FmsaveSource:
    def __init__(self, save: str | None, money_rate: float | None = None,
                 currency: str | None = None, show_hidden: bool = False,
                 folders: list[Path] | None = None):
        self._save = save
        self._rate = money_rate
        self._currency = currency
        self._show_hidden = show_hidden
        self._folders = default_games_dirs() if folders is None else folders

    def stamp(self) -> object:
        path = find_save(self._save, self._folders)
        info = path.stat()
        return path, info.st_mtime_ns, info.st_size

    def load(self) -> Dataset:
        try:
            import fmsave
        except ImportError as e:
            raise ScoutError('fmsave absent : pip install -e ".[fmsave]"') from e
        path = find_save(self._save, self._folders)
        try:
            with (tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(record=True) as caught):
                warnings.simplefilter("always")
                copy = shutil.copy2(path, tmp)
                with fmsave.open(copy) as save:
                    data = self._read(save, fmsave)
        except (fmsave.FmsaveError, OSError) as e:
            raise ScoutError(f"fmsave ne peut pas lire {path.name} : {e}\n"
                             "Mettez fmsave à jour ou utilisez un export CSV (--csv).") from e
        data.origin = [str(path)]
        data.warnings += sorted({str(w.message) for w in caught
                                 if issubclass(w.category, fmsave.ReaderCheckWarning)})
        if self._rate is None:
            data.warnings.append("Montants dans la devise interne de la save (peut différer des euros "
                                 "affichés). Réglez money_rate dans config.toml.")
        return data

    def _read(self, save: Any, fmsave: Any) -> Dataset:
        rate = self._rate or 1.0
        managed = list(save.managed_clubs())
        mine = managed[0] if managed else None
        genders = {c.uid: c.gender.label for c in save.clubs() if c.gender is not None}
        my_gender = genders.get(mine.club_uid) if mine else None
        rows = []
        for p in save.players():
            gender = genders.get(p.club_uid)
            if not p.name or (my_gender and gender and gender != my_gender):
                continue  # entrée vide, ou football féminin/masculin différent du mien
            row = self._row(p, rate)
            row["loan_from_me"] = bool(mine and p.on_loan and p.loan_parent_club_uid == mine.club_uid)
            row["on_loan"] = bool(p.on_loan)
            rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ScoutError("Aucun joueur lu dans la save.")
        for col in ("birth_date", "contract_end"):
            frame[col] = pd.to_datetime(frame[col])
        numeric = frame.columns.difference(
            ["player_id", "name", "club", "positions", "birth_date", "contract_end", "first_team",
             "loan_from_me", "on_loan"])
        frame[numeric] = frame[numeric].apply(pd.to_numeric).astype(float)
        data = Dataset(
            players=frame, my_club=mine.club_name if mine else None,
            game_date=save.info.game_date,
            currency=self._currency if self._rate is not None else None,
            budget=self._budget(save, mine.club_uid, rate) if mine else None)
        if mine:
            try:
                data.plan = self._plan(save)
                data.formation = self._formation(save, mine.club_uid, data)
            except fmsave.FmsaveError as e:
                data.warnings.append(f"Plan d'effectif ou tactique illisible : {e}")
        return data

    def _row(self, p: Any, rate: float) -> dict[str, Any]:
        contract = p.contract
        # Libre = ni club ni contrat. Club illisible mais contrat présent = inconnu.
        club = p.club_name or ("" if p.club_uid is None and contract is None else None)
        value = None if p.transfer_value is None else p.transfer_value * rate
        # Salaire 0 = souvent non renseigné : inconnu plutôt que gratuit.
        wage = contract.wage * rate if contract is not None and contract.wage else None
        codes = [*p.natural_positions, *p.accomplished_positions]
        row: dict[str, Any] = {
            "player_id": str(p.uid), "name": p.name, "age": p.age, "birth_date": p.birth_date,
            "club": club, "positions": ["ST" if c == "STC" else c for c in codes],
            "value_min": value, "value_max": value, "wage_min": wage, "wage_max": wage,
            "contract_end": contract.end if contract is not None else None,
            "first_team": p.team_slot == 0,
            "club_reputation": p.club_reputation,
        }
        if self._show_hidden:
            row |= {"ca": p.ability.current, "pa": p.ability.potential}
        for name in p.attributes.__dataclass_fields__:
            if name not in _HIDDEN:
                value = getattr(p.attributes, name)
                row[name] = row[f"{name}_min"] = row[f"{name}_max"] = value
        return row

    @staticmethod
    def _plan(save: Any) -> dict[int, dict[str, list[str]]] | None:
        plan: dict[int, dict[str, list[str]]] = {}
        for slot in save.squad_planner():
            if position := _PLANNER.get(slot.slot):
                ids = plan.setdefault(slot.season_offset, {}).setdefault(position, [])
                ids += [str(u) for u in slot.player_uids if u is not None and str(u) not in ids]
        return plan or None

    @staticmethod
    def _formation(save: Any, club_uid: int, data: Dataset) -> list[str] | None:
        found: dict[str, list[str]] = {}
        for tactic in save.tactics():
            if tactic.club_uid == club_uid and tactic.team_slot == 0:
                codes = [[_TACTIC.get(p.raw) for p in s.in_possession_positions] for s in tactic.slots]
                if len(codes) == 11 and all(len(c) == 1 and c[0] for c in codes):
                    found.setdefault(tactic.name, [c[0] for c in codes])
        if len(found) > 1:
            data.warnings.append(f"Plusieurs tactiques ({', '.join(found)}), la première est utilisée.")
        return next(iter(found.values()), None)

    @staticmethod
    def _budget(save: Any, club_uid: int, rate: float) -> Budget | None:
        months = [m for m in save.finances() if m.club_uid == club_uid]
        if not months:
            return None
        last = max(months, key=lambda m: m.month or date.min)

        def convert(amount: int | None) -> float | None:
            return None if amount is None else amount * rate
        return Budget(last.month, convert(last.transfer_budget_remaining),
                      convert(last.wage_budget_weekly), convert(last.wage_payroll_weekly))

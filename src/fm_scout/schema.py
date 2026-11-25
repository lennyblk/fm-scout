"""Schéma normalisé partagé par toutes les sources.

Une ligne = un joueur. Inconnu = NaN/None/NaT, jamais 0.
  player_id, name, nationality : texte
  age                          : années
  club                         : texte ; None = inconnu ; "" = libre
  positions                    : liste de codes canoniques (GK, DC, DL, ...)
  value_min/max, asking_price_min/max, wage_min/max : monnaie (salaire HEBDO)
  birth_date, contract_end     : datetime
  ca, pa                       : 0-200 ; absents d'un export CSV FM27
  ability_stars, potential_stars : 0-5, relatifs à l'effectif
  <attribut>, <attribut>_min, <attribut>_max : 1-20, <attribut> = milieu
  club_reputation              : nombre
  on_loan, loan_from_me        : bool, en prêt ; prêté par mon club
  first_team                   : bool, équipe première
Une colonne absente signifie que la source ne la fournit pas.
"""
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

POSITIONS = ("GK", "DC", "DL", "DR", "WBL", "WBR", "DM", "MC", "ML", "MR",
             "AMC", "AML", "AMR", "ST")


@dataclass(frozen=True)
class Budget:
    month: date | None
    transfer_remaining: float | None
    wage_budget_weekly: float | None
    wage_payroll_weekly: float | None


@dataclass
class Dataset:
    players: pd.DataFrame
    my_club: str | None = None
    game_date: date | None = None
    currency: str | None = None
    budget: Budget | None = None
    plan: dict[int, dict[str, list[str]]] | None = None  # saison -> poste -> player_id
    formation: list[str] | None = None  # 11 postes de la tactique
    origin: list[str] = field(default_factory=list)
    unmapped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ScoutError(Exception):
    """Erreur attendue, à montrer telle quelle à l'utilisateur."""

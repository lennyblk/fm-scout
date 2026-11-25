from typing import Protocol

from fm_scout.schema import Dataset


class DataSource(Protocol):
    def load(self) -> Dataset:
        """Lit la source, en lecture seule. Lève ScoutError si impossible."""
        ...

    def stamp(self) -> object:
        """Change quand les données ont changé. Lève ScoutError si introuvable."""
        ...

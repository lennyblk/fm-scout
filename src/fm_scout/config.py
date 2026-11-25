import os
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from fm_scout.schema import ScoutError


@dataclass(frozen=True)
class Config:
    csv_path: Path | None
    save: str | None
    money_rate: float | None
    currency: str
    show_hidden: bool
    llm_url: str
    model: str
    llm_timeout: float
    reasoning_effort: str
    columns: dict[str, Any]
    roles: dict[str, Any]
    zones: dict[str, Any]
    recommend: dict[str, Any]


def _yaml(path: str, default: str) -> dict[str, Any]:
    try:
        source = Path(path).expanduser() if path else files("fm_scout") / "data" / default
        return yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ScoutError(f"Lecture impossible de {path or default} : {e}") from e


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except tomllib.TOMLDecodeError as e:
        raise ScoutError(f"{path} invalide : {e}") from e
    llm = raw.get("llm", {})
    section, fm = raw.get("csv", {}), raw.get("fmsave", {})
    csv_path = os.environ.get("FM_SCOUT_CSV") or section.get("path")
    return Config(Path(csv_path).expanduser() if csv_path else None,
                  os.environ.get("FM_SCOUT_SAVE") or fm.get("save") or None,
                  fm.get("money_rate") or None, fm.get("currency", "EUR"),
                  fm.get("show_hidden", False),
                  os.environ.get("FM_SCOUT_LLM_URL") or llm.get("base_url", "http://localhost:11434/v1"),
                  os.environ.get("FM_SCOUT_MODEL") or llm.get("model", "qwen3.5:4b"),
                  float(llm.get("timeout", 300)), llm.get("reasoning_effort", ""),
                  _yaml(section.get("columns_file", ""), "columns.yaml"),
                  _yaml(section.get("roles_file", ""), "roles.yaml"),
                  _yaml(section.get("zones_file", ""), "zones.yaml"),
                  _yaml(section.get("recommend_file", ""), "recommend.yaml"))

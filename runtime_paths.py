from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
VAP_HOME = Path(os.getenv("VAP_HOME", "~/.vap")).expanduser().resolve()
VAP_BIN_DIR = VAP_HOME / "bin"
VAP_LOGS_DIR = VAP_HOME / "logs"
VAP_TMP_DIR = VAP_HOME / "tmp"
VAP_TEMP_CONFIG_DIR = VAP_TMP_DIR / "configs"
VAP_CONFIG_PATH = VAP_HOME / "config.json"
VAP_PERFETTO_HOME = VAP_HOME / "perfetto-home"
VAP_CACHE_DIR = VAP_HOME / "cache"


def ensure_vap_home() -> None:
    for path in (
        VAP_HOME,
        VAP_BIN_DIR,
        VAP_LOGS_DIR,
        VAP_TMP_DIR,
        VAP_TEMP_CONFIG_DIR,
        VAP_PERFETTO_HOME,
        VAP_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_under_vap_home(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = VAP_HOME / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(VAP_HOME):
        raise ValueError("Path must be under VAP_HOME")
    return resolved

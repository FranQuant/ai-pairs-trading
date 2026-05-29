"""Shared configuration: paths, environment, API credentials.

Importing this module guarantees data/raw, data/processed, data/static, data/cache,
and notebooks/img exist. Credentials are loaded from .env (override=True so the
dotenv beats any stale exported keys).
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv


def _find_project_root(markers=(".git", "requirements.txt")) -> Path:
    """Walk up from cwd looking for a repo-root marker."""
    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    raise RuntimeError("Could not locate project root")


ROOT: Path = _find_project_root()


class PATHS:
    """Repo-anchored paths. All non-trivial subdirectories are mkdir'd on import."""
    ROOT      = ROOT
    DATA      = ROOT / "data"
    RAW       = ROOT / "data" / "raw"
    PROCESSED = ROOT / "data" / "processed"
    STATIC    = ROOT / "data" / "static"
    ARTIFACTS = ROOT / "artifacts"
    CACHE     = ROOT / "data" / "cache"
    DOCS      = ROOT / "docs"
    IMG       = ROOT / "notebooks" / "img"


for _p in (PATHS.RAW, PATHS.PROCESSED, PATHS.STATIC, PATHS.CACHE, PATHS.IMG):
    _p.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env", override=True)

EODHD_API_KEY: str  = os.getenv("EODHD_API_KEY", "")
OFFLINE_MODE: bool  = os.getenv("OFFLINE_MODE", "0") == "1"
EODHD_BASE_URL: str = "https://eodhd.com/api"

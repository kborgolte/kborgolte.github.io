"""Zentrale Pfade und Einstellungen.

Alles lässt sich per Umgebungsvariable überschreiben, damit das Modell und die
Daten nicht zwingend im Repo liegen müssen (der Checkpoint ist 4,6 GB).
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Kartograph"

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


# Wohin Uploads, Zwischenergebnisse und fertige Szenen geschrieben werden.
DATA_DIR = _path_from_env("KARTOGRAPH_DATA", PROJECT_ROOT / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
SCENE_DIR = DATA_DIR / "scenes"

# Das geklonte LingBot-Map-Repository und der Modell-Checkpoint.
LINGBOT_REPO = _path_from_env("LINGBOT_REPO", PROJECT_ROOT / "vendor" / "lingbot-map")
MODEL_PATH = _path_from_env("LINGBOT_MODEL", PROJECT_ROOT / "models" / "lingbot-map.pt")

HOST = os.environ.get("KARTOGRAPH_HOST", "0.0.0.0")
PORT = int(os.environ.get("KARTOGRAPH_PORT", "8765"))

# 4 GB. Handyvideos sind selten größer; verhindert, dass ein Fehlgriff die Platte füllt.
MAX_UPLOAD_BYTES = int(os.environ.get("KARTOGRAPH_MAX_UPLOAD", str(4 * 1024**3)))

# Voreinstellungen für die Rekonstruktion. Bewusst konservativ: lieber ein
# Ergebnis nach ein paar Minuten als ein Speicherfehler nach zwanzig.
DEFAULT_FPS = int(os.environ.get("KARTOGRAPH_FPS", "4"))
DEFAULT_MAX_FRAMES = int(os.environ.get("KARTOGRAPH_MAX_FRAMES", "240"))
DEFAULT_IMAGE_SIZE = 518
DEFAULT_PATCH_SIZE = 14

# Punktwolken-Budget für den Browser-Viewer. Drei Millionen Punkte rendert
# WebGL noch flüssig, darüber wird es auf integrierter Grafik zäh.
VIEWER_MAX_POINTS = int(os.environ.get("KARTOGRAPH_VIEWER_POINTS", "3000000"))

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def ensure_dirs() -> None:
    for directory in (DATA_DIR, UPLOAD_DIR, SCENE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

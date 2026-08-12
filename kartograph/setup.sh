#!/usr/bin/env bash
# Einmalige Einrichtung von Kartograph.
#
# Holt LingBot-Map, den Modell-Checkpoint (4,6 GB) und three.js, und legt eine
# virtuelle Umgebung an. Der Aufruf ist wiederholbar: was schon da ist, wird
# übersprungen.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv"
LINGBOT_DIR="${ROOT}/vendor/lingbot-map"
MODEL_DIR="${ROOT}/models"
VENDOR_JS="${ROOT}/kartograph/static/vendor"
THREE_VERSION="0.169.0"

say()  { printf '\n\033[1;36m▸ %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '\n  \033[31m✗ %s\033[0m\n\n' "$1" >&2; exit 1; }

# ── Voraussetzungen ──────────────────────────────────────────────────────────

say "Umgebung prüfen"

command -v python3 >/dev/null || die "python3 nicht gefunden."

PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 - <<'EOF' || die "Python 3.10 oder neuer nötig."
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
ok "Python ${PY_VERSION}"

command -v git >/dev/null || die "git nicht gefunden."

if [[ "$(uname -s)" == "Darwin" ]]; then
  ok "macOS $(sw_vers -productVersion) auf $(uname -m)"
  [[ "$(uname -m)" == "arm64" ]] || warn "Kein Apple Silicon — die GPU-Beschleunigung entfällt."
fi

# ── Virtuelle Umgebung ───────────────────────────────────────────────────────

say "Virtuelle Umgebung"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
  ok "angelegt unter .venv"
else
  ok "schon vorhanden"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip

say "Pakete installieren (dauert beim ersten Mal ein paar Minuten)"
python -m pip install --quiet -r requirements.txt
ok "Abhängigkeiten installiert"

# ── LingBot-Map ──────────────────────────────────────────────────────────────

say "LingBot-Map"

if [[ ! -d "$LINGBOT_DIR" ]]; then
  mkdir -p "$(dirname "$LINGBOT_DIR")"
  git clone --depth 1 https://github.com/robbyant/lingbot-map.git "$LINGBOT_DIR"
  ok "geklont"
else
  ok "schon geklont"
fi

# Ohne -e findet der Runner die Paketmodule nicht. FlashInfer und Kaolin aus der
# offiziellen Anleitung bleiben bewusst außen vor: beides gibt es nur für CUDA,
# und beides brauchen wir nicht — SDPA ersetzt FlashInfer, und die Punktwolke
# exportiert Kartograph selbst statt über die CUDA-Render-Pipeline.
python -m pip install --quiet -e "$LINGBOT_DIR" --no-deps
ok "als Paket eingebunden"

# ── Checkpoint ───────────────────────────────────────────────────────────────

say "Modell-Checkpoint (4,6 GB)"

mkdir -p "$MODEL_DIR"
if [[ -f "${MODEL_DIR}/lingbot-map.pt" ]]; then
  ok "schon vorhanden"
else
  warn "Der Download dauert je nach Leitung 5-20 Minuten."
  python - <<EOF
from huggingface_hub import hf_hub_download
import shutil, pathlib

path = hf_hub_download("robbyant/lingbot-map", "lingbot-map.pt")
target = pathlib.Path("${MODEL_DIR}") / "lingbot-map.pt"
# Kopieren statt symlinken: der HF-Cache wird gern mal aufgeräumt.
shutil.copy(path, target)
print(f"  gespeichert unter {target}")
EOF
  ok "geladen"
fi

# ── three.js ─────────────────────────────────────────────────────────────────

say "three.js für den Viewer"

mkdir -p "$VENDOR_JS"
fetch_js() {
  local url="$1" target="$2"
  if [[ -f "$target" ]]; then
    ok "$(basename "$target") schon da"
    return
  fi
  if curl -fsSL "$url" -o "$target"; then
    ok "$(basename "$target") geladen"
  else
    warn "$(basename "$target") nicht ladbar — der Viewer bleibt leer, der Rest läuft."
  fi
}

BASE="https://unpkg.com/three@${THREE_VERSION}"
fetch_js "${BASE}/build/three.module.js" "${VENDOR_JS}/three.module.js"
fetch_js "${BASE}/examples/jsm/controls/OrbitControls.js" "${VENDOR_JS}/OrbitControls.js"

# ── Fertig ───────────────────────────────────────────────────────────────────

cat <<'EOF'

  Fertig eingerichtet.

  Starten:
      source .venv/bin/activate
      python -m kartograph

  Dann http://localhost:8765 im Browser öffnen. Der QR-Code dort führt das
  Handy auf die Aufnahmeseite — beide Geräte müssen im selben WLAN sein.

EOF

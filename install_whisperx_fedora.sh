#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Installateur WhisperX + diarisation (Fedora / NVIDIA)
# Configuration testée :
#   - Python 3.12
#   - PyTorch 2.8.0 + CUDA 12.8
#   - WhisperX 3.8.6
#   - pyannote speaker-diarization-community-1
#
# Ce script :
#   1. installe les dépendances système utiles ;
#   2. crée ~/.venvs/whisperx ;
#   3. installe PyTorch CUDA puis WhisperX ;
#   4. vérifie que le GPU NVIDIA est visible ;
#   5. configure l'authentification Hugging Face ;
#   6. vérifie l'accès à pyannote ;
#   7. crée ~/bin/transcrire-reunion avec de bons réglages par défaut.
#
# Important :
#   - l'acceptation des conditions du modèle pyannote doit être faite une fois
#     dans le navigateur ;
#   - le script n'installe PAS le pilote NVIDIA du noyau.
# =============================================================================

VENV="${HOME}/.venvs/whisperx"
WHISPERX_VERSION="${WHISPERX_VERSION:-3.8.6}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
BIN_DIR="${HOME}/bin"
RUNNER="${BIN_DIR}/transcrire-reunion"
PYANNOTE_REPO="pyannote/speaker-diarization-community-1"
PYANNOTE_URL="https://huggingface.co/${PYANNOTE_REPO}"

say() {
    printf '\n\033[1;34m==> %s\033[0m\n' "$*"
}

warn() {
    printf '\n\033[1;33mATTENTION: %s\033[0m\n' "$*" >&2
}

die() {
    printf '\n\033[1;31mERREUR: %s\033[0m\n' "$*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 0. Vérifications de base
# ---------------------------------------------------------------------------
if [[ ! -f /etc/fedora-release ]]; then
    die "Ce script est prévu pour Fedora."
fi

command -v sudo >/dev/null || die "sudo est requis."
command -v dnf >/dev/null || die "dnf est introuvable."

say "Système détecté"
cat /etc/fedora-release || true

# ---------------------------------------------------------------------------
# 1. Dépendances système
# ---------------------------------------------------------------------------
say "Installation des dépendances système"

sudo dnf install -y \
    python3.12 \
    python3.12-devel \
    gcc \
    gcc-c++ \
    git \
    curl

# ffmpeg : ne rien modifier s'il est déjà installé.
if ! command -v ffmpeg >/dev/null 2>&1; then
    warn "FFmpeg n'est pas installé. Tentative avec ffmpeg-free."
    if ! sudo dnf install -y ffmpeg-free; then
        warn "ffmpeg-free indisponible. Tentative avec le paquet ffmpeg."
        sudo dnf install -y ffmpeg
    fi
fi

ffmpeg -version | head -n 1

# ---------------------------------------------------------------------------
# 2. Vérification NVIDIA
# ---------------------------------------------------------------------------
say "Vérification du pilote NVIDIA"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi est absent. Installe d'abord un pilote NVIDIA fonctionnel."
fi

if ! nvidia-smi >/dev/null 2>&1; then
    warn "Le pilote NVIDIA est présent mais le GPU ne répond pas."
    warn "Sur cette machine, un arrêt complet puis redémarrage a déjà été nécessaire après un Xid 79."
    die "Corrige d'abord nvidia-smi, puis relance ce script."
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ---------------------------------------------------------------------------
# 3. Environnement Python 3.12
# ---------------------------------------------------------------------------
say "Création de l'environnement Python : ${VENV}"

mkdir -p "$(dirname "${VENV}")"

if [[ ! -d "${VENV}" ]]; then
    python3.12 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 4. PyTorch CUDA 12.8
# ---------------------------------------------------------------------------
say "Installation de PyTorch CUDA 12.8"

python -m pip install --upgrade \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchvision==0.23.0 \
    --index-url "${PYTORCH_INDEX}"

# ---------------------------------------------------------------------------
# 5. WhisperX stable + CLI Hugging Face
# ---------------------------------------------------------------------------
say "Installation de WhisperX ${WHISPERX_VERSION}"

python -m pip install --upgrade \
    "whisperx==${WHISPERX_VERSION}" \
    "huggingface_hub>=0.28.1"

say "Versions installées"

python - <<'PY'
import sys
import torch
import ctranslate2
import whisperx

print("Python       :", sys.version.split()[0])
print("WhisperX     :", getattr(whisperx, "__version__", "installé"))
print("PyTorch      :", torch.__version__)
print("CUDA runtime :", torch.version.cuda)
print("CUDA dispo   :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "\nERREUR : PyTorch ne voit pas CUDA alors que nvidia-smi fonctionne."
    )

print("GPU          :", torch.cuda.get_device_name(0))
print("CTranslate2  :", ctranslate2.get_cuda_device_count(), "GPU(s)")
print(
    "Compute types:",
    ctranslate2.get_supported_compute_types("cuda", 0)
)

if "int8" not in ctranslate2.get_supported_compute_types("cuda", 0):
    raise SystemExit("ERREUR : CTranslate2 ne propose pas int8 sur ce GPU.")
PY

# ---------------------------------------------------------------------------
# 6. Hugging Face / pyannote
# ---------------------------------------------------------------------------
say "Configuration Hugging Face"

echo
echo "Le modèle de diarisation est soumis à des conditions d'utilisation."
echo "Il faut les accepter UNE SEULE FOIS ici :"
echo
echo "  ${PYANNOTE_URL}"
echo

if command -v xdg-open >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    xdg-open "${PYANNOTE_URL}" >/dev/null 2>&1 || true
fi

read -r -p "Après avoir accepté les conditions dans le navigateur, appuie sur Entrée... "

# Si déjà connecté, whoami fonctionne et on ne redemande rien.
if ! hf auth whoami >/dev/null 2>&1; then
    say "Connexion à Hugging Face"
    hf auth login
fi

say "Compte Hugging Face"
hf auth whoami

# Le token est enregistré localement par `hf auth login`.
# huggingface_hub et pyannote le réutilisent automatiquement : il n'est pas
# nécessaire de l'extraire ni de l'afficher dans le shell.
say "Vérification de l'accès à pyannote"

python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="pyannote/speaker-diarization-community-1",
    filename="config.yaml",
)
print("Accès pyannote OK :", path)
PY

# ---------------------------------------------------------------------------
# 7. Commande pratique de transcription
# ---------------------------------------------------------------------------
say "Création de ${RUNNER}"

mkdir -p "${BIN_DIR}"

cat > "${RUNNER}" <<'RUNNER_EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

VENV="${HOME}/.venvs/whisperx"

usage() {
    cat <<'EOF'
Usage:
  transcrire-reunion FICHIER_AUDIO_OU_VIDEO [DOSSIER_SORTIE] [MIN_LOCUTEURS] [MAX_LOCUTEURS]

Exemple:
  transcrire-reunion \
    "$HOME/Téléchargements/reunion.mp4" \
    "$HOME/transcriptions/reunion" \
    1 \
    5

Réglages par défaut :
  modèle        : large-v3
  langue        : fr
  GPU           : cuda
  compute_type  : int8
  batch_size    : 2
  min speakers  : 1
  max speakers  : 5

Variables optionnelles :
  WHISPERX_MODEL=large-v3
  WHISPERX_LANGUAGE=fr
  WHISPERX_BATCH=2
  WHISPERX_COMPUTE=int8
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${HOME}/transcriptions/$(basename "${INPUT%.*}")}"
MIN_SPEAKERS="${3:-1}"
MAX_SPEAKERS="${4:-5}"

MODEL="${WHISPERX_MODEL:-large-v3}"
LANGUAGE="${WHISPERX_LANGUAGE:-fr}"
BATCH="${WHISPERX_BATCH:-2}"
COMPUTE="${WHISPERX_COMPUTE:-int8}"

[[ -f "${INPUT}" ]] || {
    echo "Fichier introuvable : ${INPUT}" >&2
    exit 1
}

[[ -d "${VENV}" ]] || {
    echo "Environnement WhisperX absent : ${VENV}" >&2
    exit 1
}

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if ! nvidia-smi >/dev/null 2>&1; then
    echo "Le GPU NVIDIA n'est pas disponible." >&2
    echo "Vérifie d'abord : nvidia-smi" >&2
    exit 1
fi

# Vérifie simplement qu'une session Hugging Face est disponible.
# Le token sauvegardé par `hf auth login` sera utilisé automatiquement par
# huggingface_hub/pyannote.
if ! command -v hf >/dev/null 2>&1 || ! hf auth whoami >/dev/null 2>&1; then
    echo "Authentification Hugging Face absente." >&2
    echo "Exécute : source ~/.venvs/whisperx/bin/activate && hf auth login" >&2
    exit 1
fi

mkdir -p "${OUTPUT}"

# Sur les portables hybrides NVIDIA, empêcher temporairement le dGPU de
# s'endormir pendant une longue transcription peut éviter un échec au réveil.
# Pour la machine concernée, le GPU est à 0000:01:00.0.
POWER_CONTROL="/sys/bus/pci/devices/0000:01:00.0/power/control"
ORIGINAL_POWER=""

restore_power() {
    if [[ -n "${ORIGINAL_POWER}" && -w "${POWER_CONTROL}" ]]; then
        echo "${ORIGINAL_POWER}" | sudo tee "${POWER_CONTROL}" >/dev/null || true
    elif [[ -n "${ORIGINAL_POWER}" && -e "${POWER_CONTROL}" ]]; then
        sudo sh -c "echo '${ORIGINAL_POWER}' > '${POWER_CONTROL}'" || true
    fi
}

if [[ -e "${POWER_CONTROL}" ]]; then
    ORIGINAL_POWER="$(cat "${POWER_CONTROL}" 2>/dev/null || true)"
    sudo sh -c "echo on > '${POWER_CONTROL}'"
    trap restore_power EXIT INT TERM
fi

echo
echo "Fichier      : ${INPUT}"
echo "Sortie       : ${OUTPUT}"
echo "Modèle       : ${MODEL}"
echo "Langue       : ${LANGUAGE}"
echo "Locuteurs    : ${MIN_SPEAKERS}-${MAX_SPEAKERS}"
echo "Batch        : ${BATCH}"
echo "Compute type : ${COMPUTE}"
echo

whisperx "${INPUT}" \
    --model "${MODEL}" \
    --language "${LANGUAGE}" \
    --diarize \
    --min_speakers "${MIN_SPEAKERS}" \
    --max_speakers "${MAX_SPEAKERS}" \
    --device cuda \
    --batch_size "${BATCH}" \
    --compute_type "${COMPUTE}" \
    --output_dir "${OUTPUT}"

echo
echo "Terminé. Résultats : ${OUTPUT}"
RUNNER_EOF

chmod +x "${RUNNER}"

# Ajouter ~/bin au PATH si nécessaire.
if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
    PROFILE="${HOME}/.bashrc"
    if ! grep -Fq 'export PATH="$HOME/bin:$PATH"' "${PROFILE}" 2>/dev/null; then
        {
            echo
            echo '# Commandes personnelles'
            echo 'export PATH="$HOME/bin:$PATH"'
        } >> "${PROFILE}"
    fi
    export PATH="${BIN_DIR}:${PATH}"
fi

# ---------------------------------------------------------------------------
# 8. Résumé
# ---------------------------------------------------------------------------
say "Installation terminée"

echo
echo "Environnement : ${VENV}"
echo "Commande      : ${RUNNER}"
echo
echo "Exemple :"
echo
echo '  transcrire-reunion "$HOME/Téléchargements/reunion.mp4" "$HOME/transcriptions/reunion" 1 5'
echo
echo "Au prochain terminal, ~/bin sera disponible automatiquement."
echo
echo "Note : avec FFmpeg 8, pyannote peut afficher un warning TorchCodec."
echo "Dans la configuration WhisperX actuelle, ce warning n'empêche pas la"
echo "transcription, l'alignement ni la diarisation, puisque WhisperX charge"
echo "l'audio lui-même avant de le transmettre à pyannote."

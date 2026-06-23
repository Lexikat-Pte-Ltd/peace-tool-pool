#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_URL="${MODEL_URL:-https://drive.google.com/uc?id=1f7dUdfA_W8He9czG6SoYQBmUsSPrA6MZ}"
DEST_DIR="dependencies"
FORCE=0

usage() {
  cat <<'USAGE'
Install the PEACE layout detection YOLO model files.

Usage:
  bash scripts/install_layout_models.sh [--dest dependencies] [--force]

Options:
  --dest DIR   Archive extraction root. Defaults to dependencies.
  --force      Download and extract even when expected weights already exist.
  -h, --help   Show this help text.

Environment:
  MODEL_URL    Override the Google Drive model archive URL.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --dest\n' >&2
        exit 2
      fi
      DEST_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required. Install uv first: https://docs.astral.sh/uv/\n' >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

case "$DEST_DIR" in
  /*) INSTALL_ROOT="$DEST_DIR" ;;
  *) INSTALL_ROOT="${REPO_ROOT}/${DEST_DIR}" ;;
esac

EXPECTED_WEIGHTS=(
  "${INSTALL_ROOT}/models/det_component/weights/best.pt"
  "${INSTALL_ROOT}/models/det_legend/weights/best.pt"
)

if [[ "$FORCE" -eq 0 ]]; then
  all_present=1
  for weight_path in "${EXPECTED_WEIGHTS[@]}"; do
    if [[ ! -f "$weight_path" ]]; then
      all_present=0
      break
    fi
  done

  if [[ "$all_present" -eq 1 ]]; then
    printf 'Layout models already installed under %s\n' "${INSTALL_ROOT}/models"
    printf 'Use --force to download and extract again.\n'
    exit 0
  fi
fi

mkdir -p "$INSTALL_ROOT"
ARCHIVE_PATH="$(mktemp -t peace-layout-models.XXXXXX.zip)"
trap 'rm -f "$ARCHIVE_PATH"' EXIT

printf 'Downloading PEACE layout model archive...\n'
uvx --from gdown gdown "$MODEL_URL" -O "$ARCHIVE_PATH"

printf 'Extracting model archive into %s...\n' "$INSTALL_ROOT"
uv run --no-project python - "$ARCHIVE_PATH" "$INSTALL_ROOT" <<'PY'
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import sys

archive_path = Path(sys.argv[1])
install_root = Path(sys.argv[2])
install_root_resolved = install_root.resolve()

try:
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = (install_root / member.filename).resolve()
            if target_path != install_root_resolved and install_root_resolved not in target_path.parents:
                raise SystemExit(f"Refusing to extract unsafe archive member: {member.filename}")
        archive.extractall(install_root)
except BadZipFile as exc:
    raise SystemExit(f"Downloaded file is not a valid zip archive: {archive_path}") from exc
PY

missing=()
for weight_path in "${EXPECTED_WEIGHTS[@]}"; do
  if [[ ! -f "$weight_path" ]]; then
    missing+=("$weight_path")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  printf 'Model archive extracted, but expected weight files were not found:\n' >&2
  for weight_path in "${missing[@]}"; do
    printf '  %s\n' "$weight_path" >&2
  done
  exit 1
fi

printf 'Installed layout models under %s\n' "${INSTALL_ROOT}/models"

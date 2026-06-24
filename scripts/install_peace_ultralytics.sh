#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT="${PEACE_SOURCE_ROOT:-$HOME/peace}"
DEST_DIR="dependencies/ultralytics"
FORCE=0

usage() {
  cat <<'USAGE'
Install PEACE's vendored Ultralytics YOLOv10 tree into this repo.

Usage:
  bash scripts/install_peace_ultralytics.sh [--source ~/peace] [--dest dependencies/ultralytics] [--force]

Options:
  --source DIR  PEACE source checkout root. Defaults to PEACE_SOURCE_ROOT or ~/peace.
  --dest DIR    Destination Ultralytics directory. Defaults to dependencies/ultralytics.
  --force       Replace an existing destination directory.
  -h, --help    Show this help text.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --source\n' >&2
        exit 2
      fi
      SOURCE_ROOT="$2"
      shift 2
      ;;
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

case "$SOURCE_ROOT" in
  /*) SOURCE_ROOT_ABS="$SOURCE_ROOT" ;;
  *) SOURCE_ROOT_ABS="${REPO_ROOT}/${SOURCE_ROOT}" ;;
esac

case "$DEST_DIR" in
  /*) DEST_DIR_ABS="$DEST_DIR" ;;
  *) DEST_DIR_ABS="${REPO_ROOT}/${DEST_DIR}" ;;
esac

SOURCE_ULTRALYTICS="${SOURCE_ROOT_ABS}/dependencies/ultralytics"

if [[ ! -d "$SOURCE_ULTRALYTICS" ]]; then
  printf 'PEACE Ultralytics source directory was not found: %s\n' "$SOURCE_ULTRALYTICS" >&2
  printf 'Set PEACE_SOURCE_ROOT or pass --source with a local PEACE checkout.\n' >&2
  exit 1
fi

if [[ -e "$DEST_DIR_ABS" && "$FORCE" -eq 0 ]]; then
  printf 'Ultralytics tree already exists at %s\n' "$DEST_DIR_ABS"
  printf 'Use --force to replace it.\n'
  exit 0
fi

uv run --no-project python - "$SOURCE_ULTRALYTICS" "$DEST_DIR_ABS" "$FORCE" <<'PY'
from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2]).resolve()
force = sys.argv[3] == "1"

if dest.exists():
    if not force:
        raise SystemExit(f"Destination already exists: {dest}")
    shutil.rmtree(dest)

dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, dest)
print(f"Installed PEACE Ultralytics tree at {dest}")
PY

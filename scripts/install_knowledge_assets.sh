#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_ROOT="${PEACE_SOURCE_ROOT:-$HOME/peace}"
DEST_DIR="dependencies/knowledge"
FORCE=0

usage() {
  cat <<'USAGE'
Install PEACE geological knowledge assets into this repo.

Usage:
  bash scripts/install_knowledge_assets.sh [--source ~/peace] [--dest dependencies/knowledge] [--force]

Options:
  --source DIR  PEACE source checkout root. Defaults to PEACE_SOURCE_ROOT or ~/peace.
  --dest DIR    Destination knowledge directory. Defaults to dependencies/knowledge.
  --force       Replace files that already exist at the destination.
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

SOURCE_KNOWLEDGE="${SOURCE_ROOT_ABS}/dependencies/knowledge"

if [[ ! -d "$SOURCE_KNOWLEDGE" ]]; then
  printf 'PEACE knowledge source directory was not found: %s\n' "$SOURCE_KNOWLEDGE" >&2
  printf 'Set PEACE_SOURCE_ROOT or pass --source with a local PEACE checkout.\n' >&2
  exit 1
fi

required_assets=(
  "k2_rock_type.json"
  "k2_rock_age.json"
  "earthquake_1970_4.5mag.csv"
  "gem_active_faults_harmonized.geojson"
)

optional_assets=(
  "k2_rock_detail.json"
  "k2_usage.json"
  "k2_expertise.json"
)

missing=()
for asset in "${required_assets[@]}"; do
  if [[ ! -f "${SOURCE_KNOWLEDGE}/${asset}" ]]; then
    missing+=("${SOURCE_KNOWLEDGE}/${asset}")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  printf 'Required knowledge assets were not found:\n' >&2
  for asset in "${missing[@]}"; do
    printf '  %s\n' "$asset" >&2
  done
  exit 1
fi

mkdir -p "$DEST_DIR_ABS"

copy_asset() {
  local asset="$1"
  local source_path="${SOURCE_KNOWLEDGE}/${asset}"
  local dest_path="${DEST_DIR_ABS}/${asset}"
  if [[ -e "$dest_path" && "$FORCE" -eq 0 ]]; then
    printf 'Knowledge asset already exists at %s\n' "$dest_path"
    printf 'Use --force to replace it.\n'
    return
  fi
  cp "$source_path" "$dest_path"
  printf 'Installed %s\n' "$dest_path"
}

for asset in "${required_assets[@]}"; do
  copy_asset "$asset"
done

for asset in "${optional_assets[@]}"; do
  if [[ -f "${SOURCE_KNOWLEDGE}/${asset}" ]]; then
    copy_asset "$asset"
  fi
done

printf 'Installed knowledge assets under %s\n' "$DEST_DIR_ABS"

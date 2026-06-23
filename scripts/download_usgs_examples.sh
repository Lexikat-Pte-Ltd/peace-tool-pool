#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ID="microsoft/PEACE"
REVISION="main"
REMOTE_PREFIX="usgs_images"
LOCAL_DIR="data/peace"
COUNT="${USGS_SAMPLE_COUNT:-5}"
DOWNLOAD_ALL=0
FORCE=0

usage() {
  cat <<'USAGE'
Download a small set of PEACE USGS map images from Hugging Face.

Usage:
  bash scripts/download_usgs_examples.sh [--count 5] [--dest data/peace]
  bash scripts/download_usgs_examples.sh --all [--dest data/peace]

Options:
  --count N      Number of USGS images to download. Defaults to 5.
  --all          Download every file under usgs_images/ instead of a sample.
  --dest DIR     Local Hugging Face download root. Defaults to data/peace.
  --revision REF Hugging Face revision. Defaults to main.
  --force        Re-download files even when present in the local cache.
  -h, --help     Show this help text.

Environment:
  USGS_SAMPLE_COUNT  Default count when --count is not provided.
  HF_TOKEN           Optional Hugging Face token if the dataset access changes.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --count\n' >&2
        exit 2
      fi
      COUNT="$2"
      DOWNLOAD_ALL=0
      shift 2
      ;;
    --all)
      DOWNLOAD_ALL=1
      shift
      ;;
    --dest)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --dest\n' >&2
        exit 2
      fi
      LOCAL_DIR="$2"
      shift 2
      ;;
    --revision)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --revision\n' >&2
        exit 2
      fi
      REVISION="$2"
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

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || [[ "$COUNT" -lt 1 ]]; then
  printf '%s\n' "--count must be a positive integer. Got: ${COUNT}" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is required. Install uv first: https://docs.astral.sh/uv/\n' >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

case "$LOCAL_DIR" in
  /*) LOCAL_DIR_ABS="$LOCAL_DIR" ;;
  *) LOCAL_DIR_ABS="${REPO_ROOT}/${LOCAL_DIR}" ;;
esac

mkdir -p "$LOCAL_DIR_ABS"

uv run --no-project --with 'huggingface_hub>=0.32' python - \
  "$REPO_ID" \
  "$REVISION" \
  "$REMOTE_PREFIX" \
  "$LOCAL_DIR_ABS" \
  "$COUNT" \
  "$DOWNLOAD_ALL" \
  "$FORCE" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download

repo_id, revision, remote_prefix, local_dir, count, download_all, force = sys.argv[1:]
count = int(count)
download_all = download_all == "1"
force = force == "1"
prefix = remote_prefix.rstrip("/") + "/"
image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

api = HfApi()
repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
image_files = sorted(
    path
    for path in repo_files
    if path.startswith(prefix) and path.lower().endswith(image_extensions)
)

if not image_files:
    raise SystemExit(f"No image files found under {prefix!r} in dataset {repo_id!r}")

selected_files = image_files if download_all else image_files[:count]
local_root = Path(local_dir)
local_root.mkdir(parents=True, exist_ok=True)

mode = "all" if download_all else f"{len(selected_files)} of {len(image_files)}"
print(f"Downloading {mode} USGS image files from {repo_id}@{revision} into {local_root}")

downloaded_paths = []
for index, remote_path in enumerate(selected_files, start=1):
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=remote_path,
        local_dir=local_root,
        force_download=force,
    )
    downloaded_paths.append(Path(local_path))
    print(f"[{index}/{len(selected_files)}] {remote_path}")

print(f"Installed USGS examples under {local_root / prefix.rstrip('/')}")
PY

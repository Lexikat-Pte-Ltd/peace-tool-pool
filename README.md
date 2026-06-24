# peace-tool-pool

Portable extraction scaffold for the PEACE GeoMap-Agent YOLO extraction tools.

This repository is intended to distill the YOLO-backed extraction utilities from [PEACE](https://github.com/microsoft/PEACE) into a package that can be used independently of the GeoMap-Bench evaluation harness. The longer-term direction is to expose stable local Python APIs first, then add an MCP surface over those APIs.

## Shape

```text
peace-tool-pool/
  src/peace_tool_pool/        # portable package code
  dependencies/models/        # detector weights, ignored by default
  dependencies/ultralytics/   # optional vendored PEACE Ultralytics tree, ignored by default
  data/                       # local samples or benchmark data, ignored by default
  .cache/                     # runtime caches, ignored by default
```

The package should prefer explicit constructor arguments and environment-backed config over the source repo's current `sys.path` mutation, repo-root-relative model paths, and hardcoded defaults.

## Quick Start

Install `uv`, then create the baseline environment:

```bash
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Verify the scaffold imports:

```bash
uv run python -c "import peace_tool_pool; print(peace_tool_pool.__version__)"
```

## Install Profiles

The default environment stays lightweight. Install extras only for the tool families you are actively extracting or testing.

| Command                            | Use case                             |
| ---------------------------------- | ------------------------------------ |
| `uv sync`                          | Package scaffold only.               |
| `uv sync --extra detectors`        | Component and legend detection work. |
| `uv sync --extra mcp`              | Future MCP server surface.           |
| `uv sync --all-extras --group dev` | Full local development environment.  |

## Local Assets

Install the detector weights with the local bootstrap script:

```bash
bash scripts/install_layout_models.sh
```

The script uses `uvx --from gdown gdown` as a transient dependency and extracts the
downloaded archive into `dependencies/` with Python's standard library.

Expected local asset paths after bootstrap:

```text
dependencies/models/det_component/weights/best.pt
dependencies/models/det_legend/weights/best.pt
```

Install PEACE's vendored Ultralytics YOLOv10 tree if it is not already present:

```bash
bash scripts/install_peace_ultralytics.sh --source "${PEACE_SOURCE_ROOT:-$HOME/peace}"
```

Expected local source path after bootstrap:

```text
dependencies/ultralytics/
```

Install a tiny set of USGS example maps from the Hugging Face dataset without cloning
the full benchmark repository:

```bash
bash scripts/download_usgs_examples.sh --count 5
```

This writes files under `data/peace/usgs_images/` and uses `huggingface_hub` as a
transient `uv` dependency. The full `usgs_images/` folder is about 52.6 MB, so it can
also be installed without `git-lfs` when needed:

```bash
bash scripts/download_usgs_examples.sh --all
```

## Environment Variables

Use `.env.example` as the starting point for local configuration. Do not commit `.env`.

| Variable            | Purpose                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `GEOMAP_DATA_ROOT`  | Local data or sample maps.                                             |
| `GEOMAP_MODEL_ROOT` | Detector model root.                                                   |
| `GEOMAP_CACHE_ROOT` | Cache root for detector outputs and derived artifacts.                 |
| `PEACE_SOURCE_ROOT` | Optional pointer to the source PEACE checkout for local bootstrapping. |

## References

- PEACE paper: <https://arxiv.org/abs/2501.06184>
- PEACE source repo: <https://github.com/microsoft/PEACE>
- GeoMap-Bench dataset: <https://huggingface.co/datasets/microsoft/PEACE>
- MCP tools documentation: <https://modelcontextprotocol.io/docs/concepts/tools>

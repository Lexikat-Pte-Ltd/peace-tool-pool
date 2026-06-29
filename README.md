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

| Command                                  | Use case                                      |
| ---------------------------------------- | --------------------------------------------- |
| `uv sync`                                | Package scaffold only.                        |
| `uv sync --extra detectors`              | Component and legend detection work.          |
| `uv sync --extra knowledge-local`        | Optimized local geological knowledge queries. |
| `uv sync --extra knowledge-network`      | Explicit knowledge source sync/live HTTP.     |
| `uv sync --extra knowledge-earthengine`  | Live Earth Engine geographic providers.       |
| `uv sync --extra knowledge-semantic`     | Semantic K2 retrieval with embedding models.  |
| `uv sync --extra mcp`                    | Future MCP server surface.                    |
| `uv sync --all-extras --group dev`       | Full local development environment.           |

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

Install PEACE's local geological knowledge assets if you want offline knowledge
queries for legend lithology, stratigraphic age, earthquakes, and active faults:

```bash
bash scripts/install_knowledge_assets.sh --source "${PEACE_SOURCE_ROOT:-$HOME/peace}"
```

Expected local asset path after bootstrap:

```text
dependencies/knowledge/
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

## Map Processing Outputs

`MapProcessingService.process_image(...)` writes derived artifacts under
`${GEOMAP_CACHE_ROOT:-.cache}/${GEOMAP_DATASET_SOURCE:-usgs}/map_processing/`:

```text
det/<map_name>/                  # component crops
meta/<map_name>.json             # PEACE-compatible metadata
vis/<map_name>_detections.png    # original image with category-colored detections
```

## Knowledge Services

`KnowledgeService` exposes deterministic, local geological lookups without Earth
Engine credentials or live LLM calls by default:

```python
from peace_tool_pool.knowledge import Bounds, KnowledgeService

service = KnowledgeService.from_env()
bundle = service.query_bounds(
    Bounds(min_lon=-122.5, min_lat=37.0, max_lon=-121.5, max_lat=38.0),
    include=("active_faults", "earthquake_history"),
)
legend = service.enrich_legend_label("sandstone")
```

Provider outputs are written under `${GEOMAP_CACHE_ROOT:-.cache}/knowledge/v2/`
when caching is enabled.

Earthquake and active-fault providers first look for normalized source mirrors under
`${GEOMAP_KNOWLEDGE_SOURCES_ROOT:-data/knowledge/sources}`. If no mirror is present,
they fall back to PEACE's legacy local assets under `dependencies/knowledge/` and add
a warning to the returned bundle. Live network behavior is never enabled by default;
it requires per-request `provider_options={"earthquake_history": {"source_mode": "live"}}`.

Sync source mirrors explicitly after installing the network extra:

```bash
uv run --extra knowledge-network python -m peace_tool_pool.knowledge.sources.sync usgs_fdsn_events \
  --profile-json docs/source-manifests/usgs_fdsn_events/default.json
uv run --extra knowledge-network python -m peace_tool_pool.knowledge.sources.sync gem_global_active_faults \
  --profile-json docs/source-manifests/gem_global_active_faults/default.json
```

The `knowledge-local` extra enables faster local engines where available:

```bash
uv sync --extra knowledge-local
```

Set `GEOMAP_KNOWLEDGE_EARTHQUAKE_ENGINE=pandas` to require pandas CSV filtering,
or keep `auto` to use pandas when installed and fall back to stdlib CSV. Set
`GEOMAP_KNOWLEDGE_FAULT_GEOMETRY_ENGINE=shapely` to require shapely STRtree
geometry filtering, or keep `auto` to fall back to bbox filtering.

Earth Engine and semantic K2 providers are registered but explicit-only. Request
them with `include=("landcover_distribution", "population_density")` or
`include=("rock_knowledge", "component_usage_knowledge")` after installing the
matching extra and configuring credentials/assets.

Semantic K2 retrieval defaults to CUDA when `torch.cuda.is_available()` is true
and falls back to CPU only when CUDA is unavailable or `GEOMAP_SEMANTIC_DEVICE=cpu`
is set. To require CUDA, set `GEOMAP_SEMANTIC_DEVICE=cuda` or `cuda:0`; startup
will fail for that provider if CUDA is not available. Install a CUDA-enabled
PyTorch build appropriate for the host before `knowledge-semantic` if the default
wheel is not CUDA-enabled.

## Environment Variables

Use `.env.example` as the starting point for local configuration. Do not commit `.env`.

| Variable            | Purpose                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `GEOMAP_DATA_ROOT`  | Local data or sample maps.                                             |
| `GEOMAP_MODEL_ROOT` | Detector model root.                                                   |
| `GEOMAP_KNOWLEDGE_ROOT` | Local geological knowledge asset root.                            |
| `GEOMAP_KNOWLEDGE_SOURCES_ROOT` | Normalized source mirror root.                         |
| `GEOMAP_EARTHQUAKE_SOURCE_ID` | Earthquake source id, defaults to `usgs_fdsn_events`.      |
| `GEOMAP_ACTIVE_FAULT_SOURCE_ID` | Active-fault source id, defaults to `gem_global_active_faults`. |
| `GEOMAP_GEM_ACTIVE_FAULT_VERSION` | Optional pinned local GEM mirror version.              |
| `GEOMAP_CACHE_ROOT` | Cache root for detector outputs and derived artifacts.                 |
| `GEOMAP_KNOWLEDGE_EARTHQUAKE_ENGINE` | `auto`, `csv`, or `pandas`.                    |
| `GEOMAP_KNOWLEDGE_FAULT_GEOMETRY_ENGINE` | `auto`, `bbox`, or `shapely`.             |
| `GEOMAP_EARTHENGINE_PROJECT` | Earth Engine project id for live providers.                  |
| `GEOMAP_SEMANTIC_MODEL` | SentenceTransformer model for semantic K2 providers.                |
| `GEOMAP_SEMANTIC_DEVICE` | `auto`, `cpu`, `cuda`, or `cuda:<index>`.                         |
| `GEOMAP_SEMANTIC_TOP_K` | Default semantic K2 result count.                                    |
| `GEOMAP_SEMANTIC_MIN_SCORE` | Optional semantic score threshold.                             |
| `PEACE_SOURCE_ROOT` | Optional pointer to the source PEACE checkout for local bootstrapping. |

## References

- PEACE paper: <https://arxiv.org/abs/2501.06184>
- PEACE source repo: <https://github.com/microsoft/PEACE>
- GeoMap-Bench dataset: <https://huggingface.co/datasets/microsoft/PEACE>
- MCP tools documentation: <https://modelcontextprotocol.io/docs/concepts/tools>

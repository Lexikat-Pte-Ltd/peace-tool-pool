# peace-tool-pool

Portable extraction scaffold for the PEACE GeoMap-Agent YOLO extraction tools and knowledge-query engines.

This repository distills the YOLO-backed extraction utilities from [PEACE](https://github.com/microsoft/PEACE) into a package that can be used independently of the GeoMap-Bench evaluation harness. It exposes local Python APIs and a Model Context Protocol (MCP) server adapter over those APIs.

We also generalise the knowledge-base to support broader geographic usage.

## Recommended Workflow

The motivating [PEACE paper](https://arxiv.org/abs/2501.06184) shows that a
large geologic map is hard for a multimodal model to understand in one pass. A
map can be high resolution, split across many semantic regions, and dependent on
geology-specific facts that are not visible in the pixels. PEACE addresses that
with three ideas: **HIE**, **DKI**, and **PEQA**. This package turns those ideas
into local tools that a Python program or vision-language model (VLM) agent can
call directly.

The recommended order is:

```text
HIE:  extract map structure and readable evidence
DKI:  add domain knowledge for the mapped area and legend labels
PEQA: answer from selected map evidence plus injected knowledge
```

### 1. HIE: Make The Map Addressable

**Hierarchical Information Extraction (HIE)** means dividing the map into useful
parts before asking a model to reason over it. Instead of sending one giant map
image to a VLM, first detect the title, scale, legend, main map, cross sections,
and legend units, then let the model inspect only the crops that matter.

In this repo, HIE starts with `MapProcessingService.process_image(...)` or the
MCP tool `geomap_process_image`. The output is structured metadata plus local
artifacts: component crops, legend crops, and a detection overlay. The current
tools detect and crop regions; a VLM or downstream OCR step still reads printed
labels, scale text, legend names, and coordinate ticks from those crops.

Use HIE when you need to answer questions such as:

- What is the map title or scale?
- Where is the legend, main map, cross section, or stratigraphic column?
- Which legend entries and colors should the model inspect?
- Which corner labels or grid ticks are needed for georeferencing?

After the VLM reads coordinate labels or grid control points, call
`geomap_georeference` or `peace_tool_pool.georef.georeference_bounds(...)` to
turn those observations into bounds, an affine transform, and a residual-quality
report showing ground-control-point fit error.

### 2. DKI: Add Geological Context

**Domain Knowledge Injection (DKI)** means adding external geoscience knowledge
after the map has been structured. The key inputs usually come from HIE: map
bounds, legend labels, and the user's question. The output is a bounded,
structured `KnowledgeBundle` that the agent can cite, filter, visualize, or use
as final-answer context.

In this repo, DKI is handled by `KnowledgeService` or the MCP tools
`geomap_query_map`, `geomap_query_knowledge`, and `geomap_enrich_legend`.
Default local providers support legend lithology and age enrichment, active
fault lookup, and earthquake history after the local knowledge assets are
installed. Optional extras add network source sync, mineral occurrences, Earth
Engine providers, and semantic K2 retrieval when those capabilities are
explicitly installed and configured.

Use DKI when you need to answer questions such as:

- What rock type or stratigraphic age does this legend label imply?
- Are active faults known inside this map extent?
- What earthquake history is relevant to this mapped region?
- Are known mineral occurrences inside this map extent?
- Which external facts should be considered before answering an analysis question?

### 3. Optional PEQA: Assemble The Answer

**Prompt-enhanced Question Answering (PEQA)** is the final answer-building step.
It combines the structured map metadata from HIE, the external knowledge from
DKI, and the specific crops or overlays that matter for the question. PEQA is
not a single standalone tool in this package today; it is the client or agent
behavior that sits on top of the SDK or MCP primitives.

For MCP clients, the pattern is to keep large images as `geomap://` resources,
read only the crop or overlay resources that are relevant to the question, and
then ask the VLM to answer with the structured JSON and selected visual evidence
in context. Use `geomap_render_knowledge_overlay` when a visual overlay will make
the injected knowledge easier to inspect.

This repository does not include a VLM client. Bring your own VLM or OCR flow to
read crops, scale text, legend labels, and coordinate labels, and to assemble the
final PEQA prompt.

### Tool Map

| Workflow step                     | Python SDK                                                                                   | MCP tool                                              |
| --------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Register or load a map            | Pass `image_path` directly to `MapProcessingService.process_image(...)`                      | `geomap_register_map(path)`                           |
| HIE layout extraction             | `MapProcessingService.process_image(...)`                                                    | `geomap_process_image(map_id)`                        |
| Georeference from VLM-read labels | `georeference_bounds(...)`                                                                   | `geomap_georeference(map_id, crs, gcps)`              |
| DKI by bounds or labels           | `KnowledgeService.query_bounds(...)`; advanced callers can use `KnowledgeService.query(...)` | `geomap_query_knowledge(...)`                         |
| DKI from registered map state     | `KnowledgeService.query_map(...)`                                                            | `geomap_query_map(map_id, question=...)`              |
| Legend lithology and age          | `KnowledgeService.enrich_legend_label(...)`                                                  | `geomap_enrich_legend(label)`                         |
| Visual evidence for PEQA          | `peace_tool_pool.knowledge.render_knowledge_overlay_svg(...)`                                | `geomap_render_knowledge_overlay(map_id, bundle_uri)` |

For a first complete local agent pass, run
`uv sync --extra mcp --extra detectors --extra geo --extra knowledge-local`.
Then install detector weights and local knowledge assets from the Local Assets
section before running HIE or DKI. Add the network, Earth Engine, or semantic
extras only when the question requires those providers.

## Shape

```text
peace-tool-pool/
  src/peace_tool_pool/        # portable package code
  dependencies/models/        # detector weights, ignored by default
  dependencies/ultralytics/   # optional vendored PEACE Ultralytics tree, ignored by default
  data/                       # local samples or benchmark data, ignored by default
  .cache/                     # runtime caches, ignored by default
```

The package uses explicit constructor arguments and environment-backed config instead of the source repo's `sys.path` mutation, repo-root-relative model paths, and hardcoded defaults.

## Quick Start

Requires Python 3.10 or 3.11. Install `uv`, then create the baseline environment:

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

| Command                                 | Use case                                                                           |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| `uv sync`                               | Package scaffold only.                                                             |
| `uv sync --extra detectors`             | Component and legend detection work.                                               |
| `uv sync --extra knowledge-local`       | Faster earthquake CSV and active-fault geometry filtering with pandas and shapely. |
| `uv sync --extra knowledge-network`     | Explicit knowledge source sync/live HTTP.                                          |
| `uv sync --extra knowledge-earthengine` | Live Earth Engine geographic providers.                                            |
| `uv sync --extra knowledge-semantic`    | Semantic K2 retrieval with embedding models.                                       |
| `uv sync --extra geo`                   | CRS resolution and affine georeferencing.                                          |
| `uv sync --extra mcp`                   | Local MCP server for VLM agent consumption.                                        |
| `uv sync --group dev`                   | Lightweight development tools.                                                     |

The `detectors` and `knowledge-semantic` extras intentionally use incompatible
PyTorch ranges. Install them in separate environments if you need both YOLO
layout extraction and semantic K2 retrieval.

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
meta/<map_name>.json             # structured metadata for regions, legend entries, and artifacts
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

Mineral, Earth Engine, and semantic K2 providers are registered but
explicit-only. Request them with `include=("mineral_occurrences",)`,
`include=("landcover_distribution", "population_density")`, or
`include=("rock_knowledge", "component_usage_knowledge", "downstream_task_knowledge")`
after installing the matching extra and configuring credentials/assets.

## MCP Agent Surface

Install the lightweight MCP adapter and run the local stdio server:

```bash
uv sync --extra mcp
uv run peace-tool-pool-mcp
```

The MCP server is a thin adapter over the local Python SDK. It registers map images
under configured allowed roots, returns structured JSON for tool calls, exposes
generated crops/overlays as `geomap://` resources, and avoids echoing absolute file
paths to the model.

`geomap://` resource URIs are MCP handles for local artifacts such as source
maps, crops, knowledge bundles, georeference JSON, and overlays. Clients read
them through MCP `resources/read`; the server resolves the local file without
exposing absolute paths to the model.

Tool `outputSchema` values currently validate only the common response envelope
(`trace_id`, `text_summary`, warnings, and resource links). Tool-specific payload
shapes remain permissive until the SDK schemas are promoted to stable MCP output
contracts.

The MCP form of the recommended HIE -> DKI -> optional PEQA loop is:

```text
geomap_register_map(path)
  -> geomap_process_image(map_id)                 # HIE: regions, legend, crops
  -> VLM reads selected crop resources             # labels, scale, lon/lat, grid ticks
  -> geomap_georeference(map_id, crs, gcps)        # map bounds and transform
  -> geomap_query_map(map_id, question=...)        # DKI: structured knowledge bundle
  -> geomap_render_knowledge_overlay(map_id, bundle_uri)
  -> VLM answers from metadata + knowledge + selected resources
```

Use `geomap_query_knowledge(...)` instead of `geomap_query_map(...)` when you
already have explicit bounds or legend labels and do not need registered map
state. Treat the final VLM answer as the PEQA step: the server supplies the
evidence, while the client decides which resources and knowledge records belong
in the final prompt.

Install additional extras for the capabilities you want the server to expose:
`--extra detectors` for image processing, `--extra geo` for georeferencing, and the
knowledge extras for local, network, Earth Engine, or semantic providers.

Knowledge query tools may write to the local cache and register returned bundles
as `geomap://bundles/...` resources. They do not call live providers or mutate
external services unless requested through provider options. Because of those
local cache/registry side effects, they are marked non-read-only in MCP
annotations.

Semantic K2 retrieval defaults to CUDA when `torch.cuda.is_available()` is true
and falls back to CPU only when CUDA is unavailable or `GEOMAP_SEMANTIC_DEVICE=cpu`
is set. To require CUDA, set `GEOMAP_SEMANTIC_DEVICE=cuda` or `cuda:0`; startup
will fail for that provider if CUDA is not available. Install a CUDA-enabled
PyTorch build appropriate for the host before `knowledge-semantic` if the default
wheel is not CUDA-enabled.

## Environment Variables

Use `.env.example` as the starting point for local configuration. Do not commit `.env`.

| Variable                                 | Purpose                                                                                                   |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `GEOMAP_DATA_ROOT`                       | Local data or sample maps.                                                                                |
| `GEOMAP_MODEL_ROOT`                      | Detector model root.                                                                                      |
| `GEOMAP_KNOWLEDGE_ROOT`                  | Local geological knowledge asset root.                                                                    |
| `GEOMAP_KNOWLEDGE_SOURCES_ROOT`          | Normalized source mirror root.                                                                            |
| `GEOMAP_MCP_ALLOWED_ROOTS`               | Path-list of local roots the MCP server may read. Defaults to `GEOMAP_DATA_ROOT` and `GEOMAP_CACHE_ROOT`. |
| `GEOMAP_EARTHQUAKE_SOURCE_ID`            | Earthquake source id, defaults to `usgs_fdsn_events`.                                                     |
| `GEOMAP_ACTIVE_FAULT_SOURCE_ID`          | Active-fault source id, defaults to `gem_global_active_faults`.                                           |
| `GEOMAP_GEM_ACTIVE_FAULT_VERSION`        | Optional pinned local GEM mirror version.                                                                 |
| `GEOMAP_CACHE_ROOT`                      | Cache root for detector outputs and derived artifacts.                                                    |
| `GEOMAP_KNOWLEDGE_EARTHQUAKE_ENGINE`     | `auto`, `csv`, or `pandas`.                                                                               |
| `GEOMAP_KNOWLEDGE_FAULT_GEOMETRY_ENGINE` | `auto`, `bbox`, or `shapely`.                                                                             |
| `GEOMAP_EARTHENGINE_PROJECT`             | Earth Engine project id for live providers.                                                               |
| `GEOMAP_SEMANTIC_MODEL`                  | SentenceTransformer model for semantic K2 providers.                                                      |
| `GEOMAP_SEMANTIC_DEVICE`                 | `auto`, `cpu`, `cuda`, or `cuda:<index>`.                                                                 |
| `GEOMAP_SEMANTIC_TOP_K`                  | Default semantic K2 result count.                                                                         |
| `GEOMAP_SEMANTIC_MIN_SCORE`              | Optional semantic score threshold.                                                                        |
| `PEACE_SOURCE_ROOT`                      | Optional pointer to the source PEACE checkout for local bootstrapping.                                    |

## References

- PEACE paper: <https://arxiv.org/abs/2501.06184>
- PEACE source repo: <https://github.com/microsoft/PEACE>
- GeoMap-Bench dataset: <https://huggingface.co/datasets/microsoft/PEACE>
- MCP tools documentation: <https://modelcontextprotocol.io/docs/concepts/tools>

# Examples — standard entry points

Runnable, minimal demonstrations of the tool pool, each importing the real APIs.
These are the place to start if you want to see how to *use* the tools. (Internal
quality benchmarks and evals live separately under [`scripts/benchmarks/`](../scripts/benchmarks/).)

All examples operate on the bundled maps in `data/test-inputs/` via a single
registry — `peace_tool_pool.examples.maps` — which is the one source of truth for
each map's georeferencing (CRS, ground control points, neat-line extent).

| # | Example | What it shows | Extras |
| - | --- | --- | --- |
| 1 | `01_process_map.py` | CV pipeline: detect components + legend on a map image | `detectors` |
| 2 | `02_georeference_map.py` | Fit GCPs → EPSG:4326 bounds; pixel↔lon/lat round-trip | `geo` |
| 3 | `03_knowledge_overlay_on_map.py` | Georef → query knowledge → annotate the map raster | `geo knowledge-local knowledge-network detectors` |

Run with `uv run --extra <...> python examples/<file>.py [--map <key>]`, e.g.:

```bash
uv run --extra geo --extra knowledge-local --extra knowledge-network --extra detectors \
    python examples/03_knowledge_overlay_on_map.py --map huronian
```

## The test-input maps

| key | scale | knowledge target? | notes |
| --- | --- | --- | --- |
| `osmani` | regional (1:100k, UTM 15N) | yes | Shebandowan belt; ~86 mineral occurrences |
| `huronian` | regional (1:100k, UTM 15N) | yes | Shebandowan/Quetico; 43 mineral occurrences |
| `harfang` | **channel (~40 m, UTM 18N)** | **no** | James Bay, QC; regional knowledge returns 0 at this footprint |

`harfang` is georeferenceable (use it with example 2), but a regional knowledge
overlay is **not** meaningful at a ~40 m channel exposure — example 3 only offers
the regional maps.

## Where the GCPs come from

A VLM is not wired up yet, so the ground control points were read by hand off each
map's printed UTM graticule (the regional grids were pinned computationally by
detecting the tick marks in the white margin just outside the neat-line). They
live in `src/peace_tool_pool/examples/maps.py`; swap in your own map there to run
the examples on it.

## Prerequisites

- Example 1 needs the YOLO weights — see `scripts/install_peace_ultralytics.sh`.
- Example 3 makes live calls to OGS MDI / SIGÉOM (the `knowledge-network` extra);
  results are cached under `.cache/knowledge/` so re-runs of the same map are offline.

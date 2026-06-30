"""Registry of the bundled test-input maps and their hand-read georeferencing.

Single source of truth for the example entry points. The ground control points
were read off each map's printed UTM graticule (no VLM is wired up yet); see
``examples/README.md`` for how they were obtained and validated.

GCPs are ``(pixel_x, pixel_y, world_easting, world_northing)``. ``pixel_extent``
is the ``(x0, y0, x1, y1)`` neat-line (main-map) box used both to reproject the
map corners and to crop the raster behind a knowledge overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_INPUTS_DIR = REPO_ROOT / "data" / "test-inputs"


@dataclass(frozen=True)
class MapGeoref:
    crs: str
    gcps: tuple[tuple[float, float, float, float], ...]
    pixel_extent: tuple[int, int, int, int]


@dataclass(frozen=True)
class TestMap:
    """A bundled test-input map plus the metadata needed to use the tools on it."""

    key: str
    image_filename: str
    title: str
    scale: str  # "regional" | "channel"
    knowledge_target: bool  # whether a regional knowledge overlay is meaningful here
    georef: MapGeoref | None = None
    notes: str = ""

    @property
    def image_path(self) -> Path:
        return TEST_INPUTS_DIR / self.image_filename


# Osmani regional geology (1:100k). GCPs from the printed UTM N83 Zone 15 grid.
OSMANI = TestMap(
    key="osmani",
    image_filename="7-1-regional-geology-osmani-1993.png",
    title="Osmani regional geology (Shebandowan belt, ON)",
    scale="regional",
    knowledge_target=True,
    georef=MapGeoref(
        crs="UTM N83 Zone 15",
        gcps=((167, 99, 660000, 5400000), (1175, 1238, 690000, 5360000)),
        pixel_extent=(23, 20, 1332, 1344),
    ),
)

# Huronian Gold Project bedrock geology (1:100k). GCP ticks were detected
# computationally off the UTM N83 Zone 15 graticule; bounds land in the
# Shebandowan/Quetico shield, overlapping Osmani.
HURONIAN = TestMap(
    key="huronian",
    image_filename="7-3-huronian-bedrock-geology-2025.png",
    title="Huronian Gold Project bedrock geology (Shebandowan/Quetico, ON)",
    scale="regional",
    knowledge_target=True,
    georef=MapGeoref(
        crs="UTM N83 Zone 15",
        gcps=((391, 478, 660000, 5385000), (1107, 1195, 670000, 5375000)),
        pixel_extent=(272, 319, 1328, 1398),
    ),
)

# Harfang Serpent-Radisson channel showing. A ~40 m channel exposure in NAD83 /
# UTM zone 18N (James Bay, QC). Georeferenceable, but NOT a regional-knowledge
# target: at its true footprint regional sources (e.g. SIGEOM) return 0 -- the
# "392" seen in earlier runs came from a 2x1 deg box ~60,000x the map. GCPs/extent
# are best-effort (faint neat-line); good enough for a georef-only example.
HARFANG = TestMap(
    key="harfang",
    image_filename="harfang-lithium-assay-2023.png",
    title="Harfang Serpent-Radisson channel showing (James Bay, QC)",
    scale="channel",
    knowledge_target=False,
    georef=MapGeoref(
        crs="NAD83 / UTM zone 18N",
        gcps=((694, 578, 360690, 5884250), (121, 864, 360670, 5884240)),
        pixel_extent=(105, 275, 1290, 990),
    ),
    notes=(
        "~40 m channel exposure; regional knowledge returns 0 at this footprint. "
        "Use for a georef-only example, not a knowledge overlay."
    ),
)

TEST_MAPS: dict[str, TestMap] = {m.key: m for m in (OSMANI, HURONIAN, HARFANG)}


def build_georeference(test_map: TestMap) -> Any:
    """Build a :class:`peace_tool_pool.georef.GeoReference` for a map (needs ``geo`` extra)."""
    from ..georef import GroundControlPoint, georeference_bounds

    if test_map.georef is None:
        raise ValueError(f"{test_map.key!r} has no georeferencing.")
    gcps = [
        GroundControlPoint(pixel_x=a, pixel_y=b, world_x=c, world_y=d)
        for a, b, c, d in test_map.georef.gcps
    ]
    return georeference_bounds(
        crs=test_map.georef.crs, gcps=gcps, pixel_extent=test_map.georef.pixel_extent
    )


def query_map_metadata(test_map: TestMap) -> dict[str, Any]:
    """``KnowledgeService.query_map``-ready metadata (georef + image_path)."""
    if test_map.georef is None:
        raise ValueError(f"{test_map.key!r} has no georeferencing.")
    return {
        "image_path": str(test_map.image_path),
        "georef": {
            "crs": test_map.georef.crs,
            "gcps": [list(gcp) for gcp in test_map.georef.gcps],
            "pixel_extent": list(test_map.georef.pixel_extent),
        },
    }

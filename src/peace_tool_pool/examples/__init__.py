"""Reusable assets backing the runnable examples in the top-level ``examples/``.

The example entry points (and tests/benchmarks) import the test-input map
registry from here so georeferencing control points live in exactly one place.
"""

from .maps import (
    HARFANG,
    HURONIAN,
    OSMANI,
    TEST_MAPS,
    MapGeoref,
    TestMap,
    build_georeference,
    query_map_metadata,
)

__all__ = [
    "TestMap",
    "MapGeoref",
    "OSMANI",
    "HURONIAN",
    "HARFANG",
    "TEST_MAPS",
    "build_georeference",
    "query_map_metadata",
]

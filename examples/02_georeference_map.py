"""Example 2 - georeference a map to EPSG:4326 bounds.

Fits a pixel->world affine from the map's ground control points (read off its
printed UTM graticule, stored in the registry), reprojects the extent to lon/lat,
and shows the pixel<->lon/lat round-trip the knowledge overlay relies on.

    uv run --extra geo python examples/02_georeference_map.py [--map osmani|huronian|harfang]
"""

from __future__ import annotations

import argparse

from peace_tool_pool.examples import TEST_MAPS, build_georeference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="huronian", choices=sorted(TEST_MAPS))
    args = parser.parse_args()
    test_map = TEST_MAPS[args.map]

    ref = build_georeference(test_map)
    bounds = ref.bounds
    print(f"Map: {test_map.title}  ({test_map.scale}-scale)")
    print(f"  CRS: {test_map.georef.crs!r} -> {ref.crs}")
    print(
        f"  bounds (EPSG:4326): lon [{bounds.min_lon:.4f}, {bounds.max_lon:.4f}], "
        f"lat [{bounds.min_lat:.4f}, {bounds.max_lat:.4f}]"
    )
    print(f"  affine fit residual: {ref.residual:.4f} m")

    # Round-trip a pixel out to lon/lat and back -- this inverse is what places
    # knowledge points back onto the raster in example 3.
    x0, y0, x1, y1 = test_map.georef.pixel_extent
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lon, lat = ref.pixel_to_lonlat(cx, cy)
    bx, by = ref.lonlat_to_pixel(lon, lat)
    print(f"  centre pixel ({cx:.0f}, {cy:.0f}) -> ({lon:.4f}, {lat:.4f}) -> ({bx:.1f}, {by:.1f})")


if __name__ == "__main__":
    main()

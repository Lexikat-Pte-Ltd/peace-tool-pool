"""Example 3 - overlay federated knowledge onto a georeferenced map.

The headline workflow: georeference a regional map, query the knowledge service
for its extent (mineral occurrences, active faults, seismicity), and annotate the
input image with the results in pixel space -- a reviewable, map-backed overlay.

    uv run --extra geo --extra knowledge-local --extra knowledge-network --extra detectors \
        python examples/03_knowledge_overlay_on_map.py [--map osmani|huronian] [--out PATH]

Only regional maps are valid targets: Harfang is a ~40 m channel exposure, so a
regional knowledge overlay is not meaningful (it would draw nothing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from peace_tool_pool.examples import TEST_MAPS, build_georeference, query_map_metadata
from peace_tool_pool.knowledge import KnowledgeService
from peace_tool_pool.knowledge.visualization import (
    extract_knowledge_overlay,
    render_knowledge_overlay_on_image,
)

PROVIDERS = ("mineral_occurrences", "active_faults", "earthquake_history")


def main() -> None:
    targets = sorted(key for key, m in TEST_MAPS.items() if m.knowledge_target)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="osmani", choices=targets)
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--max-records", type=int, default=200)
    args = parser.parse_args()
    test_map = TEST_MAPS[args.map]

    ref = build_georeference(test_map)
    service = KnowledgeService.from_env()
    service.config.max_records_per_provider = args.max_records
    bundle = service.query_map(query_map_metadata(test_map), include=PROVIDERS)
    overlay = extract_knowledge_overlay(bundle)

    out = Path(args.out) if args.out else (
        Path("docs") / "benchmarks" / f"knowledge_overlay_{test_map.key}_annotated.png"
    )
    render_knowledge_overlay_on_image(
        overlay, ref, test_map.image_path, out, title=f"{test_map.title} - knowledge lookup"
    )

    counts = {item.key: item.record_count for item in bundle.items}
    plotted = sum(1 for item in overlay.items if item.kind in {"result_point", "result_bbox"})
    print(f"Map: {test_map.title}")
    print(f"  knowledge counts: {counts}")
    print(f"  annotations drawn: {plotted}  (out of map: {len(overlay.out_of_bounds)})")
    print(f"  annotated map written: {out}")


if __name__ == "__main__":
    main()

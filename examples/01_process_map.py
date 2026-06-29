"""Example 1 - process a map image into structured components.

Runs the map-processing CV pipeline (component detection + legend extraction) on
a bundled test-input map and prints what it found, writing a detection overlay.

    uv run --extra detectors python examples/01_process_map.py [--map osmani|huronian|harfang]

See examples/README.md for prerequisites (YOLO weights via
scripts/install_peace_ultralytics.sh).
"""

from __future__ import annotations

import argparse
import os

from peace_tool_pool.examples import TEST_MAPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="osmani", choices=sorted(TEST_MAPS))
    args = parser.parse_args()
    test_map = TEST_MAPS[args.map]

    os.environ.setdefault("GEOMAP_DATASET_SOURCE", "test-inputs")
    from peace_tool_pool.map_processing import MapProcessingService

    result = MapProcessingService().process_image(test_map.image_path).to_dict()

    print(f"Map: {test_map.title}")
    print(f"  size: {result['size']['width']}x{result['size']['height']}")
    detected = {label: len(dets) for label, dets in result["regions"].items() if dets}
    print(f"  components detected: {detected}")
    print(f"  legend entries: {len(result.get('legend', []))}")
    roles = sorted({a.get("role") for a in result.get("artifacts", [])})
    print(f"  artifact roles: {roles}")


if __name__ == "__main__":
    main()

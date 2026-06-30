import json
from pathlib import Path

import pytest

from peace_tool_pool.map_processing.config import MapProcessingConfig
from peace_tool_pool.map_processing.service import MapProcessingService
from peace_tool_pool.map_processing.types import Detection
from peace_tool_pool.mcp.adapter import GeomapMcpAdapter
from peace_tool_pool.mcp.resources import ResourceRegistry

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


class FakeComponentDetector:
    def detect(self, image_path: str | Path):
        return {
            "main_map": [Detection(label="main_map", bbox=(0, 0, 60, 60), confidence=0.99)],
            "legend": [Detection(label="legend", bbox=(70, 10, 120, 40), confidence=0.95)],
        }


class FakeLegendDetector:
    def detect(self, image_path: str | Path):
        return {
            "color_bndbox": [[2, 2, 10, 10]],
            "text_bndbox": [[15, 1, 42, 11]],
        }


def test_process_image_returns_resource_uris_and_bounded_preview(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    cache_root.mkdir()
    monkeypatch.setenv("GEOMAP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("GEOMAP_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("GEOMAP_MCP_ALLOWED_ROOTS", f"{data_root}:{cache_root}")
    image_path = data_root / "sample.png"
    image = np.full((100, 130, 3), 255, dtype=np.uint8)
    image[5:25, 5:25] = (0, 0, 255)
    image[12:20, 72:80] = (0, 0, 255)
    image[12:20, 88:105] = 0
    assert cv2.imwrite(str(image_path), image)
    registry = ResourceRegistry.from_env(base_dir=tmp_path)
    map_info = registry.register_map(image_path)

    def service_factory():
        return MapProcessingService(
            config=MapProcessingConfig(
                data_root=data_root,
                model_root=tmp_path / "models",
                cache_root=cache_root,
                dataset_source="fixture",
            ),
            component_detector=FakeComponentDetector(),
            legend_detector=FakeLegendDetector(),
        )

    adapter = GeomapMcpAdapter(registry=registry, map_service_factory=service_factory)
    result = adapter.process_image(map_id=map_info["map_id"])
    structured = result["structuredContent"]

    assert structured["source_uri"] == map_info["source_uri"]
    assert structured["regions"]["main_map"][0]["artifact_uri"].startswith(
        "geomap://artifacts/"
    )
    assert all("path" not in artifact for artifact in structured["artifacts"])
    assert all(artifact["uri"].startswith("geomap://artifacts/") for artifact in structured["artifacts"])
    assert str(tmp_path) not in json.dumps(result)
    assert result["content"][0]["type"] == "text"
    assert any(part["type"] == "image" for part in result["content"])
    assert structured["preview"]["artifact_uri"].startswith("geomap://artifacts/")

    artifact_resource = adapter.read_resource(structured["artifacts"][0]["uri"])
    assert artifact_resource["mimeType"] == "image/png"

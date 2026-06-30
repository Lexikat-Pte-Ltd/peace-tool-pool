import base64
import json

import pytest

from peace_tool_pool.mcp.errors import McpToolError
from peace_tool_pool.mcp.resources import ResourceRegistry


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGA"
    "WjR9awAAAABJRU5ErkJggg=="
)


def _registry(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    cache_root.mkdir()
    monkeypatch.setenv("GEOMAP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("GEOMAP_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv(
        "GEOMAP_MCP_ALLOWED_ROOTS",
        f"{data_root}:{cache_root}",
    )
    return ResourceRegistry.from_env(base_dir=tmp_path), data_root, cache_root


def test_register_map_is_idempotent_and_redacts_source_path(tmp_path, monkeypatch):
    registry, data_root, _ = _registry(tmp_path, monkeypatch)
    image_path = data_root / "map.png"
    image_path.write_bytes(PNG_1X1)

    first = registry.register_map(image_path)
    second = registry.register_map(image_path)

    assert first["map_id"] == second["map_id"]
    assert first["map_uri"] == f"geomap://maps/{first['map_id']}"
    assert first["source_uri"] == f"geomap://maps/{first['map_id']}/source"
    assert str(image_path) not in json.dumps(first)

    content = registry.read_resource(first["source_uri"])
    assert content["uri"] == first["source_uri"]
    assert content["mimeType"] == "image/png"
    assert base64.b64decode(content["blob"]) == PNG_1X1


def test_registry_preserves_entries_from_stale_process_instances(tmp_path, monkeypatch):
    first_registry, data_root, _ = _registry(tmp_path, monkeypatch)
    stale_registry = ResourceRegistry.from_env(base_dir=tmp_path)
    first_path = data_root / "first.png"
    second_path = data_root / "second.png"
    first_path.write_bytes(PNG_1X1)
    second_path.write_bytes(PNG_1X1)

    first = first_registry.register_map(first_path)
    second = stale_registry.register_map(second_path)

    reloaded = ResourceRegistry.from_env(base_dir=tmp_path)
    assert reloaded.map_public(first["map_id"])["source_uri"] == first["source_uri"]
    assert reloaded.map_public(second["map_id"])["source_uri"] == second["source_uri"]


def test_registry_rejects_paths_outside_allowed_roots(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_1X1)

    with pytest.raises(McpToolError) as exc_info:
        registry.register_map(outside)

    assert exc_info.value.code == "disallowed_path"


def test_registry_rejects_symlink_escape(tmp_path, monkeypatch):
    registry, data_root, _ = _registry(tmp_path, monkeypatch)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_1X1)
    link = data_root / "linked.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this filesystem")

    with pytest.raises(McpToolError) as exc_info:
        registry.register_map(link)

    assert exc_info.value.code == "disallowed_path"


def test_reading_stale_artifact_returns_typed_error(tmp_path, monkeypatch):
    registry, _, cache_root = _registry(tmp_path, monkeypatch)
    artifact_path = cache_root / "overlay.png"
    artifact_path.write_bytes(PNG_1X1)
    artifact = registry.register_artifact(artifact_path, role="detection_overlay", stage="hie")
    artifact_path.unlink()

    with pytest.raises(McpToolError) as exc_info:
        registry.read_resource(artifact["uri"])

    assert exc_info.value.code == "artifact_not_found"

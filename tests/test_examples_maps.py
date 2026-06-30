import pytest

from peace_tool_pool.examples import (
    HARFANG,
    HURONIAN,
    OSMANI,
    TEST_MAPS,
    build_georeference,
    query_map_metadata,
)


def test_all_test_map_images_exist():
    for key, test_map in TEST_MAPS.items():
        assert test_map.image_path.exists(), f"{key} image missing: {test_map.image_path}"


def test_regional_maps_are_knowledge_targets_channel_is_not():
    assert OSMANI.knowledge_target and OSMANI.scale == "regional"
    assert HURONIAN.knowledge_target and HURONIAN.scale == "regional"
    # Harfang is a ~40 m channel exposure -> not a regional-knowledge target.
    assert not HARFANG.knowledge_target and HARFANG.scale == "channel"


def test_build_georeference_gives_sane_canadian_bounds():
    pytest.importorskip("pyproj")
    for key, test_map in TEST_MAPS.items():
        ref = build_georeference(test_map)
        b = ref.bounds
        assert b.min_lon < b.max_lon and b.min_lat < b.max_lat, key
        # all three sit in the Canadian shield/James Bay (neg lon, mid-high lat)
        assert -95 < b.min_lon < -70 and 45 < b.min_lat < 60, key


def test_query_map_metadata_is_query_map_ready():
    metadata = query_map_metadata(HURONIAN)
    assert set(metadata) >= {"image_path", "georef"}
    assert set(metadata["georef"]) >= {"crs", "gcps", "pixel_extent"}
    assert metadata["image_path"].endswith(".png")

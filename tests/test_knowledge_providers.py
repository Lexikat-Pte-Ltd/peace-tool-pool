from pathlib import Path

import pytest

from peace_tool_pool.knowledge import Bounds, KnowledgeRequest
from peace_tool_pool.knowledge.errors import MissingAssetError
from peace_tool_pool.knowledge.providers.earthquakes import EarthquakeHistoryProvider
from peace_tool_pool.knowledge.providers.faults import ActiveFaultProvider
from peace_tool_pool.knowledge.providers.rock import RockLookupProvider


FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


def test_rock_provider_prefers_exact_then_longest_substring_and_handles_cjk():
    provider = RockLookupProvider(
        asset_path=FIXTURES / "k2_rock_type.json",
        provider_id="rock_type",
        output_key="rock_type",
        name="Rock type",
    )

    items = provider.query(
        KnowledgeRequest(
            legend_labels=[
                "sandstone",
                "coarse red sandstone beds",
                "灰色灰岩夹白云岩",
                "mystery unit",
            ]
        )
    )

    values = [item.value["value"] for item in items]
    assert values == ["sedimentary", "red sedimentary", "carbonate", "unknown"]
    assert items[0].provenance["match_type"] == "exact"
    assert items[1].provenance["matched_name"] == "red sandstone"
    assert items[2].provenance["matched_name"] == "灰岩"
    assert items[3].record_count == 0


def test_earthquake_provider_filters_sorts_and_truncates():
    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes.csv", default_max_records=50)
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    items = provider.query(KnowledgeRequest(bounds=bounds, max_records_by_provider={"earthquake_history": 1}))

    assert len(items) == 1
    item = items[0]
    assert item.record_count == 2
    assert item.truncated is True
    assert item.value == [
        {
            "time": "2022-01-02T00:00:00Z",
            "latitude": 37.5,
            "longitude": -121.5,
            "place": "Newer in bounds",
            "mag": 5.2,
            "magType": "mb",
            "depth": 8.0,
            "type": "earthquake",
            "updated": "2022-01-03T00:00:00Z",
            "gap": 30,
        }
    ]


def test_earthquake_provider_tolerates_missing_optional_columns():
    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes_minimal.csv", default_max_records=50)
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    item = provider.query(KnowledgeRequest(bounds=bounds))[0]

    assert item.record_count == 1
    assert item.value == [
        {"time": "2021-05-01T00:00:00Z", "latitude": 37.4, "longitude": -121.6}
    ]


def test_fault_provider_filters_geojson_and_truncates():
    provider = ActiveFaultProvider(FIXTURES / "active_faults.geojson", default_max_records=1)
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    item = provider.query(KnowledgeRequest(bounds=bounds))[0]

    assert item.record_count == 2
    assert item.truncated is True
    assert item.value == [
        {
            "slip_type": "strike-slip",
            "name": "Alpha Fault",
            "catalog_name": "Fixture",
            "dip_dir": "NE",
            "average_dip": 70,
            "average_rake": 10,
            "lower_seis_depth": 12,
            "upper_seis_depth": 1,
            "geometry_bbox": [-122.0, 37.0, -121.0, 38.0],
        }
    ]


def test_file_backed_providers_raise_for_missing_assets(tmp_path):
    provider = EarthquakeHistoryProvider(tmp_path / "missing.csv")
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    with pytest.raises(MissingAssetError):
        provider.query(KnowledgeRequest(bounds=bounds))

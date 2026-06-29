from pathlib import Path

import pytest

from peace_tool_pool.knowledge import Bounds, KnowledgeRequest
from peace_tool_pool.knowledge.errors import MissingAssetError, ProviderOptionError
from peace_tool_pool.knowledge.providers import earthquakes, faults
from peace_tool_pool.knowledge.providers.earthquakes import (
    EarthquakeHistoryProvider,
    EarthquakeSourceBinding,
)
from peace_tool_pool.knowledge.providers.faults import ActiveFaultProvider, FaultSourceBinding
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


def test_earthquake_provider_validates_and_applies_provider_options():
    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes.csv", default_max_records=50)
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    item = provider.query(
        KnowledgeRequest(
            bounds=bounds,
            provider_options={"earthquake_history": {"starttime": "2021-01-01", "minmagnitude": "5.0"}},
        )
    )[0]

    assert item.record_count == 1
    assert item.value[0]["place"] == "Newer in bounds"
    assert item.provenance["provider_options"]["minmagnitude"] == 5.0

    with pytest.raises(ProviderOptionError):
        provider.validate_options({"bogus": True})


def test_earthquake_provider_federates_selected_sources_with_exact_id_dedupe(tmp_path):
    usgs_csv = tmp_path / "usgs.csv"
    emsc_csv = tmp_path / "emsc.csv"
    header = "time,latitude,longitude,place,mag,magType,depth,type,updated,gap,ids,sources\n"
    usgs_csv.write_text(
        header
        + "2026-01-01T00:00:00Z,41.5,12.5,USGS event,5.1,mw,10,earthquake,2026-01-02T00:00:00Z,20,shared-a,us\n",
        encoding="utf-8",
    )
    emsc_csv.write_text(
        header
        + "2026-01-01T00:00:01Z,41.5,12.5,EMSC event,5.1,mw,10,earthquake,2026-01-02T00:00:00Z,20,shared-a,emsc\n",
        encoding="utf-8",
    )
    provider = EarthquakeHistoryProvider(
        usgs_csv,
        source_bindings=[
            EarthquakeSourceBinding(
                source_id="usgs_fdsn_events",
                source_mode="legacy_asset",
                asset_path=usgs_csv,
            ),
            EarthquakeSourceBinding(
                source_id="emsc_fdsn_events",
                source_mode="legacy_asset",
                asset_path=emsc_csv,
            ),
        ],
    )

    item = provider.query(
        KnowledgeRequest(
            bounds=Bounds(min_lon=12, min_lat=41, max_lon=13, max_lat=42),
            provider_options={
                "earthquake_history": {"sources": ["usgs_fdsn_events", "emsc_fdsn_events"]}
            },
        )
    )[0]

    assert item.record_count == 1
    assert item.value[0]["place"] == "EMSC event"
    assert item.provenance["source_ids"] == ["usgs_fdsn_events", "emsc_fdsn_events"]
    assert len(item.provenance["sources"]) == 2
    assert item.provenance["dedupe_key"] == "association_set_overlap_exact"


def test_earthquake_provider_does_not_fuzzy_merge_cross_catalog_events(tmp_path):
    usgs_csv = tmp_path / "usgs.csv"
    emsc_csv = tmp_path / "emsc.csv"
    header = "time,latitude,longitude,place,mag,magType,depth,type,updated,gap,ids,sources\n"
    usgs_csv.write_text(
        header
        + "2026-01-01T00:00:00Z,41.500,12.500,USGS event,5.1,mw,10,earthquake,2026-01-02T00:00:00Z,20,us-a,us\n",
        encoding="utf-8",
    )
    emsc_csv.write_text(
        header
        + "2026-01-01T00:00:05Z,41.501,12.501,EMSC event,5.1,mw,10,earthquake,2026-01-02T00:00:00Z,20,emsc-a,emsc\n",
        encoding="utf-8",
    )
    provider = EarthquakeHistoryProvider(
        usgs_csv,
        source_bindings=[
            EarthquakeSourceBinding(
                source_id="usgs_fdsn_events",
                source_mode="legacy_asset",
                asset_path=usgs_csv,
            ),
            EarthquakeSourceBinding(
                source_id="emsc_fdsn_events",
                source_mode="legacy_asset",
                asset_path=emsc_csv,
            ),
        ],
    )

    item = provider.query(
        KnowledgeRequest(
            bounds=Bounds(min_lon=12, min_lat=41, max_lon=13, max_lat=42),
            provider_options={"earthquake_history": {"source": "all"}},
        )
    )[0]

    assert item.record_count == 2
    assert item.provenance["dedupe_key"] == "association_set_overlap_exact"


def test_earthquake_provider_live_source_version_is_request_specific(tmp_path):
    provider = EarthquakeHistoryProvider(tmp_path / "missing.csv")

    assert provider.source_version_for_options({"source_mode": "live"}).startswith("1@live:")
    assert provider.source_version().startswith("1@missing:")


def test_earthquake_provider_cache_config_records_resolved_auto_engine(monkeypatch):
    monkeypatch.setattr(earthquakes, "_dependency_available", lambda name: name == "pandas")
    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes.csv", engine="auto")

    assert provider.cache_config()["resolved_engine"] == "pandas"

    monkeypatch.setattr(earthquakes, "_dependency_available", lambda name: False)
    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes.csv", engine="auto")

    assert provider.cache_config()["resolved_engine"] == "csv"


def test_earthquake_provider_coerces_numpy_like_scalars_to_native_float():
    class FakeScalar:
        def item(self):
            return 8.0

    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes.csv")

    assert provider._coerce_value(FakeScalar()) == 8.0
    assert provider._coerce_value(8.0) == 8.0


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


def test_fault_provider_rejects_live_mode_and_warns_on_zero_result_gap():
    provider = ActiveFaultProvider(FIXTURES / "active_faults.geojson", default_max_records=1)

    with pytest.raises(ProviderOptionError):
        provider.validate_options({"source_mode": "live"})

    item = provider.query(
        KnowledgeRequest(
            bounds=Bounds(min_lon=43, min_lat=-26, max_lon=51, max_lat=-11),
            provider_options={"active_faults": {"source_mode": "local_mirror"}},
        )
    )[0]

    assert item.record_count == 0
    assert any("madagascar" in warning.lower() for warning in provider.last_warnings)
    assert any("not evidence" in warning for warning in provider.last_warnings)
    assert item.provenance["coverage_caveats"]


def test_fault_provider_supports_explicit_live_diss_binding():
    class FakeDissAdapter:
        endpoint = "fixture://diss"
        normalizer_version = "1"

        def __init__(self):
            self.calls = []

        def query_bbox(self, bounds):
            self.calls.append(bounds)
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": "diss-1", "name": "DISS Source"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[12.0, 41.0], [13.0, 42.0]],
                        },
                    }
                ],
            }

        def normalize_geojson(self, data):
            return data

    adapter = FakeDissAdapter()
    provider = ActiveFaultProvider(
        FIXTURES / "active_faults.geojson",
        source_bindings=[
            FaultSourceBinding(
                source_id="diss_seismogenic_sources",
                source_mode="live",
                adapter=adapter,
                supports_live=True,
                coverage_bounds=Bounds(min_lon=5, min_lat=35, max_lon=20, max_lat=48),
                region_name="Italy",
            )
        ],
    )

    item = provider.query(
        KnowledgeRequest(
            bounds=Bounds(min_lon=12, min_lat=41, max_lon=13, max_lat=42),
            provider_options={"active_faults": {"source": "diss_seismogenic_sources"}},
        )
    )[0]

    assert item.record_count == 1
    assert adapter.calls
    assert item.provenance["source_ids"] == ["diss_seismogenic_sources"]
    assert item.provenance["sources"][0]["source_mode"] == "live"


def test_fault_provider_cache_config_records_resolved_auto_engine(monkeypatch):
    monkeypatch.setattr(faults, "_dependency_available", lambda name: name == "shapely")
    provider = ActiveFaultProvider(FIXTURES / "active_faults.geojson", geometry_engine="auto")

    assert provider.cache_config()["resolved_geometry_engine"] == "shapely"

    monkeypatch.setattr(faults, "_dependency_available", lambda name: False)
    provider = ActiveFaultProvider(FIXTURES / "active_faults.geojson", geometry_engine="auto")

    assert provider.cache_config()["resolved_geometry_engine"] == "bbox"


def test_file_backed_providers_raise_for_missing_assets(tmp_path):
    provider = EarthquakeHistoryProvider(tmp_path / "missing.csv")
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    with pytest.raises(MissingAssetError):
        provider.query(KnowledgeRequest(bounds=bounds))


def test_fault_provider_legacy_provenance_carries_gem_attribution():
    """Legacy bundled GEM data is still GEM-derived; its CC BY-SA attribution must survive."""
    provider = ActiveFaultProvider(FIXTURES / "active_faults.geojson")
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    provenance = provider.query(KnowledgeRequest(bounds=bounds))[0].provenance

    assert provenance["source_mode"] == "legacy_asset"
    assert provenance["license"] == (
        "CC BY-SA 4.0 (Creative Commons Attribution Share Alike 4.0 International)"
    )
    assert provenance["citation"] == "GEM Global Active Faults Database"
    assert "GEM" in (provenance["attribution"] or "")


def test_earthquake_provider_legacy_provenance_carries_usgs_attribution():
    provider = EarthquakeHistoryProvider(FIXTURES / "earthquakes.csv")
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    provenance = provider.query(KnowledgeRequest(bounds=bounds))[0].provenance

    assert provenance["source_mode"] == "legacy_asset"
    assert provenance["license"] == "See USGS source policy"
    assert provenance["citation"] == "USGS FDSN Event API"
    assert "USGS" in (provenance["attribution"] or "")

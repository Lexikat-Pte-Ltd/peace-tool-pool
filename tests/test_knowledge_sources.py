import json
from pathlib import Path

import pytest

from peace_tool_pool.knowledge import Bounds, KnowledgeConfig
from peace_tool_pool.knowledge.errors import (
    OptionalDependencyError,
    SourceRegistryError,
    SourceManifestError,
    SourceQueryError,
)
from peace_tool_pool.knowledge.sources.diss_faults import DissSeismogenicSourceAdapter
from peace_tool_pool.knowledge.sources.sigeom_minerals import (
    SigeomMineralOccurrenceAdapter,
    normalize_sigeom_features,
)
from peace_tool_pool.knowledge.sources.gem_faults import (
    GEM_GAP_BBOXES,
    GemActiveFaultSourceAdapter,
    coverage_caveats_for_bounds,
)
from peace_tool_pool.knowledge.sources.manifest import SourceManifest, find_latest_manifest
from peace_tool_pool.knowledge.sources.registry import SourceRegistry, default_source_registry
from peace_tool_pool.knowledge.sources.usgs_events import (
    EMSC_DEFAULT_PROFILE,
    EMSC_EVENT_BASE_URL,
    FdsnEventSourceAdapter,
    UsgsFdsnEventAdapter,
    associated_id_set,
    normalize_geojson_events,
)


FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


def test_source_manifest_round_trips_and_hash_ignores_retrieval_time():
    manifest = SourceManifest(
        source_id="usgs_fdsn_events",
        family="earthquake_events",
        retrieved_at="2026-06-25T00:00:00Z",
        source_version="usgs-fdsn-event-service",
        normalizer_version="1",
        source_url="https://earthquake.usgs.gov/fdsnws/event/1/query",
        request={"format": "geojson", "eventtype": "earthquake"},
        record_count=2,
        normalized_sha256="abc123",
        license="See USGS source policy",
        citation="USGS FDSN Event API",
        attribution="USGS Earthquake Hazards Program FDSN Event API",
        coverage={"status": "global-service", "notes": []},
        artifacts={"normalized": "normalized/earthquakes.csv"},
    )

    as_dict = manifest.to_dict()
    assert as_dict["schema_version"] == "knowledge-source/v1"
    assert SourceManifest.from_dict(as_dict) == manifest

    changed_retrieval = SourceManifest.from_dict({**as_dict, "retrieved_at": "2026-06-26T00:00:00Z"})
    assert changed_retrieval.stable_hash() == manifest.stable_hash()

    changed_request = SourceManifest.from_dict(
        {**as_dict, "request": {"format": "geojson", "eventtype": "earthquake", "minmagnitude": 4.5}}
    )
    assert changed_request.stable_hash() != manifest.stable_hash()


def test_source_manifest_rejects_unknown_schema_version():
    with pytest.raises(SourceManifestError):
        SourceManifest.from_dict({"schema_version": "bad"})


def test_default_source_registry_definitions_and_profile_validation():
    registry = default_source_registry()

    usgs = registry.get("usgs_fdsn_events")
    assert usgs.family == "earthquake_events"
    assert usgs.validate_profile({})["eventtype"] == "earthquake"
    assert usgs.validate_profile({"minmagnitude": "4.5"})["minmagnitude"] == 4.5

    gem = registry.get("gem_global_active_faults")
    assert gem.family == "active_faults"
    assert "CC BY-SA 4.0" in (gem.license or "")
    assert registry.resolve(family="active_faults")[0].id == "gem_global_active_faults"

    emsc = registry.get("emsc_fdsn_events")
    assert emsc.family == "earthquake_events"
    assert emsc.validate_profile({})["format"] == "json"

    diss = registry.get("diss_seismogenic_sources")
    assert diss.family == "active_faults"
    assert diss.coverage_bounds is not None

    sigeom = registry.get("sigeom_mineral_occurrences")
    assert sigeom.family == "mineral_occurrences"
    assert sigeom.coverage_bounds is not None

    with pytest.raises(Exception):
        SourceRegistry([]).get("missing")


def test_source_registry_resolves_ordered_coverage_aware_source_sets():
    registry = default_source_registry()
    quebec = Bounds(min_lon=-77, min_lat=52, max_lon=-75, max_lat=53)
    california = Bounds(min_lon=-122.5, min_lat=37.0, max_lon=-121.5, max_lat=38.0)

    mineral_ids = [
        definition.id
        for definition in registry.resolve(family="mineral_occurrences", bounds=quebec)
    ]
    assert mineral_ids == ["ontario_mineral_deposit_inventory", "sigeom_mineral_occurrences"]
    assert registry.resolve(family="mineral_occurrences", bounds=california) == []

    selected = registry.resolve(
        family="earthquake_events",
        options={"sources": ["emsc_fdsn_events", "usgs_fdsn_events"]},
    )
    assert [definition.id for definition in selected] == ["usgs_fdsn_events", "emsc_fdsn_events"]

    with pytest.raises(SourceRegistryError):
        registry.resolve(family="active_faults", source_id="emsc_fdsn_events")


def test_config_from_env_parses_source_root_and_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOMAP_KNOWLEDGE_SOURCES_ROOT", "source-root")
    monkeypatch.setenv("GEOMAP_EARTHQUAKE_SOURCE_ID", "usgs_fdsn_events")
    monkeypatch.setenv("GEOMAP_ACTIVE_FAULT_SOURCE_ID", "gem_global_active_faults")
    monkeypatch.setenv("GEOMAP_GEM_ACTIVE_FAULT_VERSION", "v1")

    config = KnowledgeConfig.from_env(base_dir=tmp_path)

    assert config.knowledge_sources_root == (tmp_path / "source-root").resolve()
    assert config.earthquake_source_id == "usgs_fdsn_events"
    assert config.active_fault_source_id == "gem_global_active_faults"
    assert config.gem_active_fault_version == "v1"
    assert config.cache_namespace_root == config.cache_root / "knowledge" / "v2"


def test_find_latest_manifest_prefers_default_then_sorted_version(tmp_path):
    root = tmp_path / "sources"
    source_root = root / "usgs_fdsn_events"
    (source_root / "2024").mkdir(parents=True)
    (source_root / "2025").mkdir()
    (source_root / "2024" / "manifest.json").write_text("{}", encoding="utf-8")
    (source_root / "2025" / "manifest.json").write_text("{}", encoding="utf-8")

    assert find_latest_manifest(root, "usgs_fdsn_events") == source_root / "2025" / "manifest.json"

    (source_root / "default").mkdir()
    (source_root / "default" / "manifest.json").write_text("{}", encoding="utf-8")
    assert find_latest_manifest(root, "usgs_fdsn_events") == source_root / "default" / "manifest.json"
    assert find_latest_manifest(root, "usgs_fdsn_events", preferred_version="2024") == source_root / "2024" / "manifest.json"


def test_usgs_url_builder_maps_bounds_and_defaults():
    adapter = UsgsFdsnEventAdapter(client=object())
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    params = adapter.query_params({}, bounds=bounds)

    assert params["format"] == "geojson"
    assert params["eventtype"] == "earthquake"
    assert params["minlatitude"] == 37.0
    assert params["maxlatitude"] == 38.0
    assert params["minlongitude"] == -122.0
    assert params["maxlongitude"] == -121.0


def test_emsc_fdsn_adapter_builds_json_query_and_normalizes_geojson():
    adapter = FdsnEventSourceAdapter(
        source_id="emsc_fdsn_events",
        base_url=EMSC_EVENT_BASE_URL,
        default_profile=EMSC_DEFAULT_PROFILE,
        client=object(),
    )
    bounds = Bounds(min_lon=12, min_lat=41, max_lon=13, max_lat=42)

    params = adapter.query_params({"minmagnitude": "4.0"}, bounds=bounds)

    assert params["format"] == "json"
    assert params["eventtype"] == "earthquake"
    assert params["minlongitude"] == 12.0
    assert params["maxlongitude"] == 13.0
    assert params["minmagnitude"] == 4.0

    records = normalize_geojson_events(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "emsc2026abcd",
                    "properties": {
                        "time": "2026-01-01T00:00:00Z",
                        "mag": 5.1,
                        "magType": "mw",
                        "source_id": "EMSC",
                        "source_catalog": "EMSC-RTS",
                    },
                    "geometry": {"type": "Point", "coordinates": [12.5, 41.5, 10]},
                }
            ],
        }
    )

    assert records[0]["event_id"] == "emsc2026abcd"
    assert records[0]["longitude"] == 12.5


def test_diss_adapter_builds_wfs_query_and_normalizes_faults():
    adapter = DissSeismogenicSourceAdapter(client=object())
    bounds = Bounds(min_lon=12, min_lat=41, max_lon=13, max_lat=42)

    params = adapter.query_params(bounds, type_name="DISS331:iss331")

    assert params["service"] == "WFS"
    assert params["request"] == "GetFeature"
    assert params["typeNames"] == "DISS331:iss331"
    assert params["outputFormat"] == "application/json"
    assert params["srsName"] == "CRS:84"
    assert params["bbox"] == "12.0,41.0,13.0,42.0,CRS:84"

    normalized = adapter.normalize_geojson(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "iss-1",
                    "properties": {"name": "IT Source", "slip_type": "reverse"},
                    "geometry": {"type": "LineString", "coordinates": [[12, 41], [13, 42]]},
                }
            ],
        }
    )

    props = normalized["features"][0]["properties"]
    assert props["source_id"] == "diss_seismogenic_sources"
    assert props["raw_properties"]["name"] == "IT Source"


def test_sigeom_adapter_builds_wfs_query_and_normalizes_occurrences():
    adapter = SigeomMineralOccurrenceAdapter(client=object())
    bounds = Bounds(min_lon=-77, min_lat=52, max_lon=-75, max_lat=53)

    params = adapter.query_params(bounds, type_name="SGM:Substances_metalliques")

    assert params["service"] == "WFS"
    assert params["request"] == "GetFeature"
    assert params["typeNames"] == "SGM:Substances_metalliques"
    assert params["outputFormat"] == "application/json"
    assert params["srsName"] == "CRS:84"
    assert params["bbox"] == "-77.0,52.0,-75.0,53.0,CRS:84"

    # Property names captured from the live SGM:Substances_metalliques WFS layer
    # (the production schema is French-coded, not the generic NOM/SUBSTANCE fields).
    records = normalize_sigeom_features(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "NOM_CORPS_MINR": "Junior",
                        "SUBST_PRINC": "Zinc; Plomb",
                        "ETAT_CORPS_MINR": "Indice, aucun travail",
                    },
                    "geometry": {"type": "Point", "coordinates": [-76.0, 52.5]},
                }
            ],
        }
    )

    assert records[0]["name"] == "Junior"
    assert records[0]["primary_commodity"] == "Zinc; Plomb"
    assert records[0]["status"] == "Indice, aucun travail"
    assert records[0]["longitude"] == -76.0


def test_usgs_associated_id_set_handles_comcat_delimited_ids():
    assert associated_id_set(",us7000abcd,ci1234,") == {"us7000abcd", "ci1234"}
    assert associated_id_set("") == set()


def test_usgs_chunking_raises_when_subday_window_still_overflows():
    adapter = UsgsFdsnEventAdapter(client=object())

    with pytest.raises(SourceQueryError):
        adapter.split_time_window(
            {"starttime": "2026-01-01T00:00:00Z", "endtime": "2026-01-01T12:00:00Z"},
            lambda _profile: 20001,
        )


def test_usgs_fetch_requires_knowledge_network_extra(monkeypatch):
    adapter = UsgsFdsnEventAdapter(client=None)
    monkeypatch.setattr("peace_tool_pool.knowledge.sources.usgs_events._httpx_module", lambda: None)

    with pytest.raises(OptionalDependencyError):
        adapter.count({})


def test_gem_coverage_caveats_cover_known_gap_boxes():
    canada = Bounds(*GEM_GAP_BBOXES["canada"])

    caveats = coverage_caveats_for_bounds([canada])

    assert any("canada" in caveat.lower() for caveat in caveats)


def test_gem_normalizer_preserves_raw_properties_and_parses_tuples(tmp_path):
    source = tmp_path / "faults.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Tuple Fault", "average_dip": "(45,30,60)"},
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = GemActiveFaultSourceAdapter(client=object())

    normalized = adapter.normalize_geojson(json.loads(source.read_text(encoding="utf-8")))

    properties = normalized["features"][0]["properties"]
    assert properties["raw_properties"]["average_dip"] == "(45,30,60)"
    assert properties["average_dip_uncertainty"] == {"most_likely": 45, "min": 30, "max": 60}

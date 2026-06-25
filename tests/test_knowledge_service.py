import sys
from pathlib import Path

import pytest

from peace_tool_pool.knowledge import Bounds, KnowledgeConfig, KnowledgeRequest, KnowledgeService
from peace_tool_pool.knowledge.cache import write_json_atomic
from peace_tool_pool.knowledge.errors import MissingAssetError, ProviderError
from peace_tool_pool.knowledge.types import SCHEMA_VERSION


FIXTURES = Path(__file__).parent / "fixtures" / "knowledge"


def fixture_config(tmp_path):
    return KnowledgeConfig(
        data_root=tmp_path / "data",
        knowledge_root=FIXTURES,
        cache_root=tmp_path / "cache",
        earthquake_csv_path=FIXTURES / "earthquakes.csv",
        active_fault_geojson_path=FIXTURES / "active_faults.geojson",
        k2_rock_type_path=FIXTURES / "k2_rock_type.json",
        k2_rock_age_path=FIXTURES / "k2_rock_age.json",
        max_records_per_provider=1,
    )


def test_config_from_env_resolves_knowledge_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOMAP_DATA_ROOT", "data-root")
    monkeypatch.setenv("GEOMAP_CACHE_ROOT", "cache-root")
    monkeypatch.setenv("GEOMAP_KNOWLEDGE_ROOT", "knowledge-root")
    monkeypatch.setenv("GEOMAP_EARTHENGINE_PROJECT", "earth-project")
    monkeypatch.setenv("GEOMAP_KNOWLEDGE_EARTHQUAKE_ENGINE", "pandas")
    monkeypatch.setenv("GEOMAP_KNOWLEDGE_FAULT_GEOMETRY_ENGINE", "shapely")
    monkeypatch.setenv("GEOMAP_SEMANTIC_MODEL", "fixture/model")
    monkeypatch.setenv("GEOMAP_SEMANTIC_DEVICE", "cuda:0")
    monkeypatch.setenv("GEOMAP_SEMANTIC_TOP_K", "7")
    monkeypatch.setenv("GEOMAP_SEMANTIC_MIN_SCORE", "0.25")
    monkeypatch.setenv("GEOMAP_SEMANTIC_LOCAL_FILES_ONLY", "true")

    config = KnowledgeConfig.from_env(base_dir=tmp_path)

    assert config.data_root == (tmp_path / "data-root").resolve()
    assert config.cache_root == (tmp_path / "cache-root").resolve()
    assert config.knowledge_root == (tmp_path / "knowledge-root").resolve()
    assert config.earthengine_project == "earth-project"
    assert config.earthquake_engine == "pandas"
    assert config.fault_geometry_engine == "shapely"
    assert config.semantic_model_name == "fixture/model"
    assert config.semantic_device == "cuda:0"
    assert config.semantic_top_k == 7
    assert config.semantic_min_score == 0.25
    assert config.semantic_local_files_only is True
    assert config.cache_namespace_root == config.cache_root / "knowledge" / "v1"


def test_service_queries_fixture_providers_enriches_legend_and_writes_cache(tmp_path):
    service = KnowledgeService(config=fixture_config(tmp_path))
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    bundle = service.query(
        KnowledgeRequest(
            bounds=bounds,
            legend_labels=["sandstone"],
            include=("rock_type", "rock_age", "earthquake_history", "active_faults"),
        )
    )

    by_key = bundle.items_by_key()
    assert by_key["rock_type"][0].value["value"] == "sedimentary"
    assert by_key["rock_age"][0].value["value"] == "mesozoic"
    assert by_key["earthquake_history"][0].record_count == 2
    assert by_key["active_faults"][0].record_count == 2
    assert bundle.provider_versions["rock_type"].startswith("1@sha256:")
    assert bundle.warnings == []

    enrichment = service.enrich_legend_label("coarse red sandstone beds")
    assert enrichment.lithology == "red sedimentary"
    assert enrichment.stratigraphic_age == "jurassic"

    provider_cache_root = service.config.cache_namespace_root / "providers"
    assert any(provider_cache_root.glob("rock_type/*.json"))
    assert any(provider_cache_root.glob("earthquake_history/*.json"))


def test_service_include_exclude_aliases_and_unknown_warning(tmp_path):
    service = KnowledgeService(config=fixture_config(tmp_path))
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    bundle = service.query_bounds(bounds, include=("Earthquake History", "unknown_provider"))
    assert [item.key for item in bundle.items] == ["earthquake_history"]
    assert any("unknown_provider" in warning for warning in bundle.warnings)

    excluded = service.query_bounds(bounds, exclude=("earthquake-history",))
    assert "earthquake_history" not in excluded.items_by_key()

    with pytest.raises(ProviderError):
        service.query_bounds(bounds, include=("unknown_provider",))


def test_explicit_missing_asset_raises_but_implicit_missing_assets_warn(tmp_path):
    config = KnowledgeConfig(
        data_root=tmp_path / "data",
        knowledge_root=tmp_path / "missing-knowledge",
        cache_root=tmp_path / "cache",
        write_cache=False,
    )
    service = KnowledgeService(config=config)
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    with pytest.raises(MissingAssetError):
        service.query_bounds(bounds, include=("earthquake_history",))

    bundle = service.query_bounds(bounds)
    assert bundle.items == []
    assert any("earthquake_history" in warning for warning in bundle.warnings)


def test_explicit_partial_missing_asset_warns_and_returns_successful_provider(tmp_path):
    config = fixture_config(tmp_path)
    config.earthquake_csv_path = tmp_path / "missing.csv"
    service = KnowledgeService(config=config)
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)

    bundle = service.query(
        KnowledgeRequest(
            bounds=bounds,
            legend_labels=["sandstone"],
            include=("earthquake_history", "rock_type"),
        )
    )

    assert bundle.items_by_key()["rock_type"][0].value["value"] == "sedimentary"
    assert any("earthquake_history" in warning for warning in bundle.warnings)


def test_from_env_is_cheap_and_baseline_imports_stay_light(tmp_path, monkeypatch):
    for module_name in (
        "pandas",
        "geopandas",
        "ee",
        "shapely",
        "torch",
        "transformers",
        "sentence_transformers",
        "deep_translator",
    ):
        sys.modules.pop(module_name, None)

    monkeypatch.setenv("GEOMAP_KNOWLEDGE_ROOT", str(tmp_path / "does-not-exist"))
    service = KnowledgeService.from_env(base_dir=tmp_path)

    assert service.config.knowledge_root == tmp_path / "does-not-exist"
    for module_name in (
        "pandas",
        "geopandas",
        "ee",
        "shapely",
        "torch",
        "transformers",
        "sentence_transformers",
        "deep_translator",
    ):
        assert module_name not in sys.modules


def test_optional_heavy_providers_are_explicit_only(tmp_path):
    service = KnowledgeService(config=fixture_config(tmp_path))

    bundle = service.query(KnowledgeRequest(query_text="legend usage"))

    assert bundle.items == []
    assert bundle.warnings == []


def test_corrupt_provider_cache_is_treated_as_cache_miss(tmp_path):
    service = KnowledgeService(config=fixture_config(tmp_path))
    cache_path = service.cache.provider_path("rock_type", "bad-cache")
    write_json_atomic(
        cache_path,
        {
            "schema_version": SCHEMA_VERSION,
            "provider": "rock_type",
            "provider_version": "1@sha256:bad",
            "items": [{"id": "missing-required-fields"}],
        },
    )

    assert service.cache.read_provider_items("rock_type", "bad-cache", "1@sha256:bad") is None


def test_atomic_json_write_cleans_temp_file_on_serialization_failure(tmp_path):
    cache_path = tmp_path / "cache" / "bad.json"

    with pytest.raises(TypeError):
        write_json_atomic(cache_path, {"not_serializable": object()})

    assert cache_path.parent.exists()
    assert list(cache_path.parent.iterdir()) == []

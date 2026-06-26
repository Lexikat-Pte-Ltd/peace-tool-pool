"""Configuration for local knowledge services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None or value.strip() == "":
        return None
    return _resolve_path(value, base_dir)


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _int_value(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _bool_value(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class KnowledgeConfig:
    data_root: Path
    knowledge_root: Path
    cache_root: Path
    knowledge_sources_root: Path | None = None
    cache_namespace: str = "knowledge"
    earthquake_csv_path: Path | None = None
    active_fault_geojson_path: Path | None = None
    k2_rock_type_path: Path | None = None
    k2_rock_age_path: Path | None = None
    k2_rock_detail_path: Path | None = None
    k2_usage_path: Path | None = None
    k2_expertise_path: Path | None = None
    earthengine_project: str | None = None
    earthquake_source_id: str = "usgs_fdsn_events"
    active_fault_source_id: str = "gem_global_active_faults"
    mineral_occurrence_source_id: str = "ontario_mineral_deposit_inventory"
    gem_active_fault_version: str | None = None
    earthquake_engine: str = "auto"
    fault_geometry_engine: str = "auto"
    earthengine_landcover_dataset_id: str = "ESA/WorldCover/v200"
    earthengine_population_dataset_id: str = "WorldPop/GP/100m/pop"
    earthengine_scale: int = 100
    earthengine_max_pixels: int = 100_000_000
    semantic_model_name: str = DEFAULT_SEMANTIC_MODEL
    semantic_model_revision: str | None = None
    semantic_device: str = "auto"
    semantic_top_k: int = 5
    semantic_min_score: float | None = None
    semantic_batch_size: int = 32
    semantic_local_files_only: bool = False
    write_cache: bool = True
    max_records_per_provider: int = 50
    bounds_cache_precision: int = 4

    @classmethod
    def from_env(cls, base_dir: str | Path | None = None) -> "KnowledgeConfig":
        root = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        return cls(
            data_root=_resolve_path(os.getenv("GEOMAP_DATA_ROOT", "./data"), root),
            knowledge_root=_resolve_path(
                os.getenv("GEOMAP_KNOWLEDGE_ROOT", "./dependencies/knowledge"), root
            ),
            knowledge_sources_root=_resolve_path(
                os.getenv("GEOMAP_KNOWLEDGE_SOURCES_ROOT", "./data/knowledge/sources"), root
            ),
            cache_root=_resolve_path(os.getenv("GEOMAP_CACHE_ROOT", ".cache"), root),
            earthquake_csv_path=_optional_path(os.getenv("GEOMAP_EARTHQUAKE_CSV"), root),
            active_fault_geojson_path=_optional_path(os.getenv("GEOMAP_ACTIVE_FAULT_GEOJSON"), root),
            k2_rock_type_path=_optional_path(os.getenv("GEOMAP_K2_ROCK_TYPE_JSON"), root),
            k2_rock_age_path=_optional_path(os.getenv("GEOMAP_K2_ROCK_AGE_JSON"), root),
            k2_rock_detail_path=_optional_path(os.getenv("GEOMAP_K2_ROCK_DETAIL_JSON"), root),
            k2_usage_path=_optional_path(os.getenv("GEOMAP_K2_USAGE_JSON"), root),
            k2_expertise_path=_optional_path(os.getenv("GEOMAP_K2_EXPERTISE_JSON"), root),
            earthengine_project=os.getenv("GEOMAP_EARTHENGINE_PROJECT") or None,
            earthquake_source_id=os.getenv("GEOMAP_EARTHQUAKE_SOURCE_ID", "usgs_fdsn_events"),
            active_fault_source_id=os.getenv(
                "GEOMAP_ACTIVE_FAULT_SOURCE_ID", "gem_global_active_faults"
            ),
            mineral_occurrence_source_id=os.getenv(
                "GEOMAP_MINERAL_OCCURRENCE_SOURCE_ID", "ontario_mineral_deposit_inventory"
            ),
            gem_active_fault_version=os.getenv("GEOMAP_GEM_ACTIVE_FAULT_VERSION") or None,
            earthquake_engine=os.getenv("GEOMAP_KNOWLEDGE_EARTHQUAKE_ENGINE", "auto"),
            fault_geometry_engine=os.getenv("GEOMAP_KNOWLEDGE_FAULT_GEOMETRY_ENGINE", "auto"),
            earthengine_landcover_dataset_id=os.getenv(
                "GEOMAP_EARTHENGINE_LANDCOVER_DATASET", "ESA/WorldCover/v200"
            ),
            earthengine_population_dataset_id=os.getenv(
                "GEOMAP_EARTHENGINE_POPULATION_DATASET", "WorldPop/GP/100m/pop"
            ),
            earthengine_scale=_int_value(os.getenv("GEOMAP_EARTHENGINE_SCALE"), 100),
            earthengine_max_pixels=_int_value(
                os.getenv("GEOMAP_EARTHENGINE_MAX_PIXELS"), 100_000_000
            ),
            semantic_model_name=os.getenv("GEOMAP_SEMANTIC_MODEL", DEFAULT_SEMANTIC_MODEL),
            semantic_model_revision=os.getenv("GEOMAP_SEMANTIC_MODEL_REVISION") or None,
            semantic_device=os.getenv("GEOMAP_SEMANTIC_DEVICE", "auto"),
            semantic_top_k=_int_value(os.getenv("GEOMAP_SEMANTIC_TOP_K"), 5),
            semantic_min_score=_optional_float(os.getenv("GEOMAP_SEMANTIC_MIN_SCORE")),
            semantic_batch_size=_int_value(os.getenv("GEOMAP_SEMANTIC_BATCH_SIZE"), 32),
            semantic_local_files_only=_bool_value(
                os.getenv("GEOMAP_SEMANTIC_LOCAL_FILES_ONLY"), False
            ),
        )

    @property
    def resolved_earthquake_csv_path(self) -> Path:
        return self.earthquake_csv_path or self.knowledge_root / "earthquake_1970_4.5mag.csv"

    @property
    def resolved_active_fault_geojson_path(self) -> Path:
        return self.active_fault_geojson_path or self.knowledge_root / "gem_active_faults_harmonized.geojson"

    @property
    def resolved_k2_rock_type_path(self) -> Path:
        return self.k2_rock_type_path or self.knowledge_root / "k2_rock_type.json"

    @property
    def resolved_k2_rock_age_path(self) -> Path:
        return self.k2_rock_age_path or self.knowledge_root / "k2_rock_age.json"

    @property
    def resolved_k2_rock_detail_path(self) -> Path:
        return self.k2_rock_detail_path or self.knowledge_root / "k2_rock_detail.json"

    @property
    def resolved_k2_usage_path(self) -> Path:
        return self.k2_usage_path or self.knowledge_root / "k2_usage.json"

    @property
    def resolved_k2_expertise_path(self) -> Path:
        return self.k2_expertise_path or self.knowledge_root / "k2_expertise.json"

    @property
    def cache_namespace_root(self) -> Path:
        return self.cache_root / self.cache_namespace / "v2"

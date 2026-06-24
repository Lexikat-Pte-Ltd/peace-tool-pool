"""Configuration for local knowledge services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None or value.strip() == "":
        return None
    return _resolve_path(value, base_dir)


@dataclass
class KnowledgeConfig:
    data_root: Path
    knowledge_root: Path
    cache_root: Path
    cache_namespace: str = "knowledge"
    earthquake_csv_path: Path | None = None
    active_fault_geojson_path: Path | None = None
    k2_rock_type_path: Path | None = None
    k2_rock_age_path: Path | None = None
    earthengine_project: str | None = None
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
            cache_root=_resolve_path(os.getenv("GEOMAP_CACHE_ROOT", ".cache"), root),
            earthquake_csv_path=_optional_path(os.getenv("GEOMAP_EARTHQUAKE_CSV"), root),
            active_fault_geojson_path=_optional_path(os.getenv("GEOMAP_ACTIVE_FAULT_GEOJSON"), root),
            k2_rock_type_path=_optional_path(os.getenv("GEOMAP_K2_ROCK_TYPE_JSON"), root),
            k2_rock_age_path=_optional_path(os.getenv("GEOMAP_K2_ROCK_AGE_JSON"), root),
            earthengine_project=os.getenv("GEOMAP_EARTHENGINE_PROJECT") or None,
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
    def cache_namespace_root(self) -> Path:
        return self.cache_root / self.cache_namespace / "v1"

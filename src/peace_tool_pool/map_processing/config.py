"""Configuration for local map processing."""

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
class MapProcessingConfig:
    data_root: Path
    model_root: Path
    cache_root: Path
    dataset_source: str = "usgs"
    cache_namespace: str = "map_processing"
    component_model_path: Path | None = None
    legend_model_path: Path | None = None
    peace_source_root: Path | None = None
    ultralytics_root: Path | None = None
    write_cache: bool = True

    @classmethod
    def from_env(cls, base_dir: str | Path | None = None) -> "MapProcessingConfig":
        root = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        data_root = _resolve_path(os.getenv("GEOMAP_DATA_ROOT", "./data"), root)
        model_root = _resolve_path(os.getenv("GEOMAP_MODEL_ROOT", "./dependencies/models"), root)
        cache_root = _resolve_path(os.getenv("GEOMAP_CACHE_ROOT", ".cache"), root)
        peace_source_root = _optional_path(os.getenv("PEACE_SOURCE_ROOT"), root)
        ultralytics_root = _optional_path(os.getenv("GEOMAP_ULTRALYTICS_ROOT"), root)
        dataset_source = os.getenv("GEOMAP_DATASET_SOURCE", "usgs")
        return cls(
            data_root=data_root,
            model_root=model_root,
            cache_root=cache_root,
            dataset_source=dataset_source,
            peace_source_root=peace_source_root,
            ultralytics_root=ultralytics_root,
        )

    @property
    def resolved_component_model_path(self) -> Path:
        if self.component_model_path is not None:
            return Path(self.component_model_path)
        return self.model_root / "det_component" / "weights" / "best.pt"

    @property
    def resolved_legend_model_path(self) -> Path:
        if self.legend_model_path is not None:
            return Path(self.legend_model_path)
        return self.model_root / "det_legend" / "weights" / "best.pt"

    @property
    def resolved_ultralytics_root(self) -> Path:
        if self.ultralytics_root is not None:
            return Path(self.ultralytics_root)

        local_root = self.model_root.parent / "ultralytics"
        if local_root.exists():
            return local_root

        if self.peace_source_root is not None:
            return self.peace_source_root / "dependencies" / "ultralytics"

        return local_root

    @property
    def cache_namespace_root(self) -> Path:
        return self.cache_root / self.dataset_source / self.cache_namespace

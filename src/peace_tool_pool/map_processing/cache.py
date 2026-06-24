"""Filesystem cache helpers for map processing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .config import MapProcessingConfig
from .types import MapProcessingResult


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as temp_file:
        json.dump(data, temp_file, indent=2, ensure_ascii=False)
        temp_name = temp_file.name
    Path(temp_name).replace(target)


class MapProcessingCache:
    def __init__(self, config: MapProcessingConfig):
        self.config = config

    def det_dir(self, map_name: str) -> Path:
        return self.config.cache_namespace_root / "det" / map_name

    def component_path(self, map_name: str, label: str, index: int) -> Path:
        return self.det_dir(map_name) / f"{label}_{index}.png"

    def visualization_path(self, map_name: str) -> Path:
        return self.config.cache_namespace_root / "vis" / f"{map_name}_detections.png"

    def metadata_path(self, map_name: str) -> Path:
        return self.config.cache_namespace_root / "meta" / f"{map_name}.json"

    def save_result(self, result: MapProcessingResult) -> None:
        write_json_atomic(self.metadata_path(result.name), result.to_peace_metadata())

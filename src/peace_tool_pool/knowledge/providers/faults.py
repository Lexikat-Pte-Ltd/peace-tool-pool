"""Local active fault GeoJSON provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..bounds import Bounds
from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


class ActiveFaultProvider:
    id = "active_faults"
    name = "Active faults"
    version = "1"
    output_keys = ("active_faults",)

    _selected_columns = (
        "slip_type",
        "name",
        "catalog_name",
        "length_in_kilometers",
        "dip_dir",
        "average_dip",
        "average_rake",
        "lower_seis_depth",
        "upper_seis_depth",
    )

    def __init__(self, asset_path: str | Path, default_max_records: int = 50):
        self.asset_path = Path(asset_path)
        self.default_max_records = default_max_records
        self._features: list[dict[str, Any]] | None = None
        self._digest: str | None = None

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        return source_version(self.version, self._digest)

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        self.source_version()
        if request.bounds is None:
            return []
        features = self._load_features()
        matching = [feature for feature in features if self._feature_intersects(feature, request.bounds)]
        matching.sort(key=self._sort_key)
        limit = max_records_for_request(self.id, request, self.default_max_records)
        limited = matching[:limit]
        records = [self._feature_record(feature) for feature in limited]
        total = len(matching)
        truncated = total > len(records)
        if total:
            summary = f"Found {total} active fault features intersecting bounds; returning {len(records)} records."
        else:
            summary = "No active fault features intersect the given bounds."
        return [
            KnowledgeItem(
                id=f"{self.id}:{self.id}",
                key=self.id,
                provider=self.id,
                value=records,
                summary=summary,
                source=str(self.asset_path),
                record_count=total,
                truncated=truncated,
                provenance={"asset_path": str(self.asset_path), "geometry_mode": "bbox"},
            )
        ]

    def _load_features(self) -> list[dict[str, Any]]:
        if self._features is not None:
            return self._features
        data = json.loads(self.asset_path.read_text(encoding="utf-8"))
        self._features = list(data.get("features") or [])
        return self._features

    def _feature_intersects(self, feature: dict[str, Any], bounds: Bounds) -> bool:
        feature_bbox = self._feature_bbox(feature)
        if feature_bbox is None:
            return False
        min_lon, min_lat, max_lon, max_lat = feature_bbox
        return not (
            max_lon < bounds.min_lon
            or min_lon > bounds.max_lon
            or max_lat < bounds.min_lat
            or min_lat > bounds.max_lat
        )

    def _feature_record(self, feature: dict[str, Any]) -> dict[str, Any]:
        properties = dict(feature.get("properties") or {})
        columns = [column for column in self._selected_columns if column in properties]
        if not columns:
            columns = sorted(properties)
        record = {column: properties.get(column) for column in columns}
        bbox = self._feature_bbox(feature)
        if bbox is not None:
            record["geometry_bbox"] = [round(value, 6) for value in bbox]
        return record

    def _sort_key(self, feature: dict[str, Any]) -> tuple[str, str, str]:
        properties = feature.get("properties") or {}
        return (
            str(properties.get("name") or ""),
            str(properties.get("catalog_name") or ""),
            str(properties.get("slip_type") or ""),
        )

    def _feature_bbox(self, feature: dict[str, Any]) -> tuple[float, float, float, float] | None:
        bbox = feature.get("bbox")
        if bbox and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        geometry = feature.get("geometry") or {}
        geometry_bbox = geometry.get("bbox")
        if geometry_bbox and len(geometry_bbox) >= 4:
            return (
                float(geometry_bbox[0]),
                float(geometry_bbox[1]),
                float(geometry_bbox[2]),
                float(geometry_bbox[3]),
            )
        positions = list(self._iter_positions(geometry))
        if not positions:
            return None
        longitudes = [position[0] for position in positions]
        latitudes = [position[1] for position in positions]
        return (min(longitudes), min(latitudes), max(longitudes), max(latitudes))

    def _iter_positions(self, geometry: Any) -> Iterable[tuple[float, float]]:
        if not isinstance(geometry, dict):
            return
        if geometry.get("type") == "GeometryCollection":
            for child in geometry.get("geometries") or []:
                yield from self._iter_positions(child)
            return
        yield from self._iter_coordinates(geometry.get("coordinates"))

    def _iter_coordinates(self, coordinates: Any) -> Iterable[tuple[float, float]]:
        if not isinstance(coordinates, list) or not coordinates:
            return
        if len(coordinates) >= 2 and all(isinstance(value, (int, float)) for value in coordinates[:2]):
            yield (float(coordinates[0]), float(coordinates[1]))
            return
        for child in coordinates:
            yield from self._iter_coordinates(child)

"""Local active fault GeoJSON provider."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..bounds import Bounds
from ..errors import OptionalDependencyError, ProviderOptionError
from ..sources.gem_faults import coverage_caveats_for_bounds
from ..sources.manifest import SourceManifest
from ..sources.registry import source_attribution
from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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

    def __init__(
        self,
        asset_path: str | Path,
        default_max_records: int = 50,
        geometry_engine: str = "auto",
        source_id: str = "gem_global_active_faults",
        source_mode: str = "legacy_asset",
        source_manifest_path: str | Path | None = None,
        source_manifest: SourceManifest | None = None,
        fallback_warning: str | None = None,
    ):
        self.asset_path = Path(asset_path)
        self.default_max_records = default_max_records
        self.geometry_engine = geometry_engine
        self.source_id = source_id
        self.source_mode = source_mode
        self.source_manifest_path = Path(source_manifest_path) if source_manifest_path else None
        self.source_manifest = source_manifest
        self.fallback_warning = fallback_warning
        self.last_warnings: list[str] = []
        self._features: list[dict[str, Any]] | None = None
        self._digest: str | None = None
        self._shapely_index: tuple[Any, list[tuple[dict[str, Any], Any]], dict[int, int], Any] | None = None
        self._active_geometry_mode = "bbox"

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        if self.source_manifest is not None:
            return f"{self.version}@manifest:{self.source_manifest.stable_hash()}"
        if not self.asset_path.exists():
            return f"{self.version}@missing:{self.asset_path}"
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        return source_version(self.version, self._digest)

    def source_version_for_options(self, options: Mapping[str, Any]) -> str:
        del options
        return self.source_version()

    def cache_config(self) -> dict[str, Any]:
        return {
            "geometry_engine": self.geometry_engine,
            "resolved_geometry_engine": self._resolved_geometry_engine_for_cache(),
            "source_id": self.source_id,
            "source_mode": self.source_mode,
            "source_manifest_hash": self.source_manifest.stable_hash()
            if self.source_manifest is not None
            else None,
            "normalizer_version": self.source_manifest.normalizer_version
            if self.source_manifest is not None
            else None,
        }

    def _resolved_geometry_engine_for_cache(self) -> str:
        if self.geometry_engine == "auto":
            return "shapely" if _dependency_available("shapely") else "bbox"
        return self.geometry_engine

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        if request.bounds is None:
            return []
        return self.query_bounds_parts(request, [request.bounds])

    def query_bounds_parts(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
    ) -> list[KnowledgeItem]:
        self.last_warnings = []
        if self.fallback_warning:
            self.last_warnings.append(self.fallback_warning)
        options = self.validate_options(request.provider_options.get(self.id, {}))
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        matching = self._matching_features_for_parts(bounds_parts)
        matching.sort(key=self._sort_key)
        limit = max_records_for_request(self.id, request, self.default_max_records)
        limited = matching[:limit]
        records = [self._feature_record(feature) for feature in limited]
        total = len(matching)
        truncated = total > len(records)
        coverage_caveats = coverage_caveats_for_bounds(bounds_parts)
        self.last_warnings.extend(coverage_caveats)
        if total == 0:
            self.last_warnings.append(
                "No GEM active-fault features intersected the bounds; this is not evidence "
                "that no active faults exist."
            )
        if total:
            summary = f"Found {total} active fault features intersecting bounds; returning {len(records)} records."
        else:
            summary = (
                "No active fault features intersected the bounds; this is not evidence "
                "that no active faults exist."
            )
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
                provenance=self._provenance(options, bounds_parts, coverage_caveats),
            )
        ]

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"source", "source_mode"}
        unknown = set(options) - allowed
        if unknown:
            raise ProviderOptionError(f"Unknown active_faults provider option keys: {sorted(unknown)}")
        validated = dict(options)
        source = str(validated.get("source") or self.source_id)
        if source != self.source_id:
            raise ProviderOptionError(
                f"active_faults source {source!r} does not match configured source {self.source_id!r}."
            )
        validated["source"] = source
        source_mode = str(validated.get("source_mode") or self.source_mode)
        if source_mode == "live":
            raise ProviderOptionError("active_faults does not support source_mode='live'.")
        if source_mode not in {"local_mirror", "legacy_asset"}:
            raise ProviderOptionError(f"Unsupported active_faults source_mode: {source_mode!r}")
        validated["source_mode"] = source_mode
        return validated

    def warnings_for_cached_result(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        cached_items: list[KnowledgeItem],
    ) -> list[str]:
        del request
        warnings: list[str] = []
        if self.fallback_warning:
            warnings.append(self.fallback_warning)
        warnings.extend(coverage_caveats_for_bounds(bounds_parts))
        if cached_items and (cached_items[0].record_count or 0) == 0:
            warnings.append(
                "No GEM active-fault features intersected the bounds; this is not evidence "
                "that no active faults exist."
            )
        return warnings

    def _matching_features_for_parts(self, bounds_parts: list[Bounds]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for bounds in bounds_parts:
            for feature in self._matching_features(bounds):
                deduped[self._dedupe_key(feature)] = feature
        return list(deduped.values())

    def _matching_features(self, bounds: Bounds) -> list[dict[str, Any]]:
        if self.geometry_engine in {"auto", "shapely"}:
            try:
                return self._matching_features_shapely(bounds)
            except OptionalDependencyError:
                if self.geometry_engine == "shapely":
                    raise
        if self.geometry_engine not in {"auto", "bbox", "shapely"}:
            raise ValueError(f"Unsupported active fault geometry engine: {self.geometry_engine!r}")
        self._active_geometry_mode = "bbox"
        features = self._load_features()
        return [feature for feature in features if self._feature_intersects(feature, bounds)]

    def _matching_features_shapely(self, bounds: Bounds) -> list[dict[str, Any]]:
        tree, entries, index_by_geometry_id, box = self._load_shapely_index()
        query_geometry = box(bounds.min_lon, bounds.min_lat, bounds.max_lon, bounds.max_lat)
        matching: list[dict[str, Any]] = []
        for candidate in tree.query(query_geometry):
            index = self._shapely_candidate_index(candidate, index_by_geometry_id)
            if index is None:
                continue
            feature, geometry = entries[index]
            if geometry.intersects(query_geometry):
                matching.append(feature)
        self._active_geometry_mode = "shapely"
        return matching

    def _load_shapely_index(
        self,
    ) -> tuple[Any, list[tuple[dict[str, Any], Any]], dict[int, int], Any]:
        if self._shapely_index is not None:
            return self._shapely_index
        try:
            from shapely.geometry import box, shape
            from shapely.strtree import STRtree
        except ImportError as exc:
            raise OptionalDependencyError(
                "ActiveFaultProvider shapely engine requires `uv sync --extra knowledge-local`."
            ) from exc

        entries: list[tuple[dict[str, Any], Any]] = []
        for feature in self._load_features():
            geometry_data = feature.get("geometry")
            if geometry_data is None:
                continue
            try:
                geometry = shape(geometry_data)
            except Exception:  # noqa: BLE001 - malformed feature geometries are skipped.
                continue
            if geometry.is_empty:
                continue
            entries.append((feature, geometry))
        tree = STRtree([geometry for _, geometry in entries])
        index_by_geometry_id = {id(geometry): index for index, (_, geometry) in enumerate(entries)}
        self._shapely_index = (tree, entries, index_by_geometry_id, box)
        return self._shapely_index

    def _shapely_candidate_index(self, candidate: Any, index_by_geometry_id: dict[int, int]) -> int | None:
        if isinstance(candidate, int):
            return candidate
        index_method = getattr(candidate, "__index__", None)
        if index_method is not None:
            return int(index_method())
        return index_by_geometry_id.get(id(candidate))

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

    def _dedupe_key(self, feature: dict[str, Any]) -> str:
        properties = feature.get("properties") or {}
        for key in ("id", "fid", "source_id", "name"):
            value = properties.get(key) or feature.get(key)
            if value not in (None, ""):
                return f"{key}:{value}"
        return f"bbox:{self._feature_bbox(feature)}"

    def _provenance(
        self,
        options: Mapping[str, Any],
        bounds_parts: list[Bounds],
        coverage_caveats: list[str],
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {
            "asset_path": str(self.asset_path),
            "geometry_mode": self._active_geometry_mode,
            "source_id": self.source_id,
            "source_family": "active_faults",
            "source_mode": options.get("source_mode", self.source_mode),
            "provider_options": dict(options),
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "dedupe_key": "feature_id_or_name_or_bbox",
            "coverage_caveats": list(coverage_caveats),
        }
        if self.source_manifest is not None:
            manifest_notes = list(self.source_manifest.coverage.get("notes") or [])
            provenance.update(
                {
                    "source_url": self.source_manifest.source_url,
                    "source_version": self.source_manifest.source_version,
                    "source_manifest_path": str(self.source_manifest_path),
                    "source_manifest_hash": self.source_manifest.stable_hash(),
                    "retrieved_at": self.source_manifest.retrieved_at,
                    "license": self.source_manifest.license,
                    "citation": self.source_manifest.citation,
                    "attribution": self.source_manifest.attribution,
                    "request": dict(self.source_manifest.request),
                    "normalizer_version": self.source_manifest.normalizer_version,
                    "coverage_status": self.source_manifest.coverage.get("status"),
                    "coverage_caveats": manifest_notes + list(coverage_caveats),
                }
            )
        else:
            attribution = source_attribution(self.source_id)
            provenance.update(
                {
                    "source_url": None,
                    "source_version": self.source_version(),
                    "source_manifest_path": None,
                    "source_manifest_hash": None,
                    "retrieved_at": None,
                    "license": attribution["license"],
                    "citation": attribution["citation"],
                    "attribution": attribution["attribution"],
                    "normalizer_version": None,
                    "coverage_status": "legacy-local-asset",
                }
            )
        return provenance

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

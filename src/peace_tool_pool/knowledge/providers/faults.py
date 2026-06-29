"""Local active fault GeoJSON provider."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..bounds import Bounds
from ..cache import stable_hash
from ..errors import OptionalDependencyError, ProviderOptionError
from ..sources.gem_faults import coverage_caveats_for_bounds
from ..sources.manifest import SourceManifest
from ..sources.registry import source_attribution
from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class FaultSourceBinding:
    source_id: str
    source_mode: str
    asset_path: str | Path | None = None
    source_manifest_path: str | Path | None = None
    source_manifest: SourceManifest | None = None
    adapter: Any | None = None
    supports_live: bool = False
    coverage_bounds: Bounds | None = None
    region_name: str | None = None
    fallback_warning: str | None = None


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
        source_bindings: list[FaultSourceBinding] | None = None,
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
        self._features_by_path: dict[Path, list[dict[str, Any]]] = {}
        self._digest: str | None = None
        self._digests_by_path: dict[Path, str] = {}
        self._shapely_index: tuple[Any, list[tuple[dict[str, Any], Any]], dict[int, int], Any] | None = None
        self._active_geometry_mode = "bbox"
        self.source_bindings = source_bindings or [
            FaultSourceBinding(
                source_id=source_id,
                source_mode=source_mode,
                asset_path=self.asset_path,
                source_manifest_path=source_manifest_path,
                source_manifest=source_manifest,
                fallback_warning=fallback_warning,
            )
        ]

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        return self._composite_source_version(self.source_bindings[:1])

    def source_version_for_options(self, options: Mapping[str, Any]) -> str:
        return self._composite_source_version(self._selected_bindings(options), options)

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
            "source_bindings": [self._binding_cache_config(binding) for binding in self.source_bindings],
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
        options = self.validate_options(request.provider_options.get(self.id, {}))
        selected_bindings = self._selected_bindings(options)
        matching: list[dict[str, Any]] = []
        source_infos: list[dict[str, Any]] = []
        for binding in selected_bindings:
            if binding.fallback_warning:
                self.last_warnings.append(binding.fallback_warning)
            if not self._binding_intersects_parts(binding, bounds_parts):
                self.last_warnings.append(
                    f"active_faults source {binding.source_id!r} does not cover the requested bounds."
                )
                source_infos.append(self._source_info(binding, options, 0))
                continue
            features, geometry_mode = self._features_for_binding(binding, bounds_parts, options)
            for feature in features:
                feature.setdefault("properties", {})["_source_id"] = binding.source_id
            matching.extend(features)
            source_infos.append(self._source_info(binding, options, len(features)))
            self._active_geometry_mode = (
                "federated" if len(selected_bindings) > 1 else geometry_mode
            )
        matching = self._dedupe_features(matching)
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
                provenance=self._provenance(
                    options,
                    bounds_parts,
                    coverage_caveats,
                    selected_bindings,
                    source_infos,
                ),
            )
        ]

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"source", "sources", "source_mode"}
        unknown = set(options) - allowed
        if unknown:
            raise ProviderOptionError(f"Unknown active_faults provider option keys: {sorted(unknown)}")
        validated = dict(options)
        if "source" in validated and "sources" in validated:
            raise ProviderOptionError("Use either active_faults source or sources, not both.")
        available = {binding.source_id for binding in self.source_bindings}
        if "source" in validated and validated["source"] not in (None, "", "all"):
            source = str(validated["source"])
            if source not in available:
                raise ProviderOptionError(f"Unknown configured active_faults source: {source!r}")
            validated["source"] = source
        if "sources" in validated:
            sources = self._coerce_source_ids(validated["sources"])
            unknown_sources = [source for source in sources if source not in available]
            if unknown_sources:
                raise ProviderOptionError(f"Unknown configured active_faults sources: {unknown_sources}")
            validated["sources"] = sources
        if "source_mode" in validated:
            source_mode = str(validated["source_mode"])
            if source_mode not in {"local_mirror", "legacy_asset", "live"}:
                raise ProviderOptionError(f"Unsupported active_faults source_mode: {source_mode!r}")
            if source_mode == "live" and not all(
                binding.supports_live for binding in self._selected_bindings(validated)
            ):
                raise ProviderOptionError(
                    "active_faults source_mode='live' is only valid for live-capable sources."
                )
            validated["source_mode"] = source_mode
        return validated

    def warnings_for_cached_result(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        cached_items: list[KnowledgeItem],
    ) -> list[str]:
        warnings: list[str] = []
        try:
            options = self.validate_options(request.provider_options.get(self.id, {}))
            selected_bindings = self._selected_bindings(options)
        except ProviderOptionError:
            selected_bindings = self.source_bindings[:1]
        for binding in selected_bindings:
            if binding.fallback_warning:
                warnings.append(binding.fallback_warning)
        warnings.extend(coverage_caveats_for_bounds(bounds_parts))
        if cached_items and (cached_items[0].record_count or 0) == 0:
            warnings.append(
                "No GEM active-fault features intersected the bounds; this is not evidence "
                "that no active faults exist."
            )
        return warnings

    def _matching_features_for_parts(self, bounds_parts: list[Bounds]) -> list[dict[str, Any]]:
        return self._dedupe_features(
            [feature for bounds in bounds_parts for feature in self._matching_features(bounds)]
        )

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

    def _load_features(self, asset_path: str | Path | None = None) -> list[dict[str, Any]]:
        path = Path(asset_path) if asset_path is not None else self.asset_path
        if path == self.asset_path and self._features is not None:
            return self._features
        if path in self._features_by_path:
            return self._features_by_path[path]
        data = json.loads(path.read_text(encoding="utf-8"))
        features = list(data.get("features") or [])
        self._features_by_path[path] = features
        if path == self.asset_path:
            self._features = features
        return features

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
        selected_bindings: list[FaultSourceBinding] | None = None,
        source_infos: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected_bindings = selected_bindings or self.source_bindings[:1]
        primary = selected_bindings[0]
        primary_manifest = primary.source_manifest
        provenance: dict[str, Any] = {
            "asset_path": str(primary.asset_path or self.asset_path),
            "geometry_mode": self._active_geometry_mode,
            "source_id": primary.source_id,
            "source_ids": [binding.source_id for binding in selected_bindings],
            "source_family": "active_faults",
            "source_mode": self._binding_mode(primary, options),
            "provider_options": dict(options),
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "dedupe_key": "feature_id_or_name_or_bbox",
            "coverage_caveats": list(coverage_caveats),
            "sources": list(
                source_infos
                or [self._source_info(binding, options, None) for binding in selected_bindings]
            ),
        }
        if primary_manifest is not None:
            manifest_notes = list(primary_manifest.coverage.get("notes") or [])
            provenance.update(
                {
                    "source_url": primary_manifest.source_url,
                    "source_version": primary_manifest.source_version,
                    "source_manifest_path": str(primary.source_manifest_path),
                    "source_manifest_hash": primary_manifest.stable_hash(),
                    "retrieved_at": primary_manifest.retrieved_at,
                    "license": primary_manifest.license,
                    "citation": primary_manifest.citation,
                    "attribution": primary_manifest.attribution,
                    "request": dict(primary_manifest.request),
                    "normalizer_version": primary_manifest.normalizer_version,
                    "coverage_status": primary_manifest.coverage.get("status"),
                    "coverage_caveats": manifest_notes + list(coverage_caveats),
                }
            )
        else:
            attribution = source_attribution(primary.source_id)
            provenance.update(
                {
                    "source_url": None,
                    "source_version": self._binding_version(primary, options),
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

    def _features_for_binding(
        self,
        binding: FaultSourceBinding,
        bounds_parts: list[Bounds],
        options: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        mode = self._binding_mode(binding, options)
        if mode == "live":
            if not binding.supports_live or binding.adapter is None:
                raise ProviderOptionError(
                    f"active_faults source {binding.source_id!r} does not support live mode."
                )
            features: list[dict[str, Any]] = []
            for bounds in bounds_parts:
                data = binding.adapter.query_bbox(bounds)
                normalize = getattr(binding.adapter, "normalize_geojson", None)
                normalized = normalize(data) if callable(normalize) else data
                features.extend(list((normalized or {}).get("features") or []))
            return self._dedupe_features(features), "live"
        if binding.asset_path is None:
            raise ProviderOptionError(f"active_faults source {binding.source_id!r} has no local artifact.")
        path = Path(binding.asset_path)
        self._digest_for_path(path)
        features = [
            feature
            for bounds in bounds_parts
            for feature in self._matching_features_for_path(path, bounds)
        ]
        return self._dedupe_features(features), self._active_geometry_mode

    def _matching_features_for_path(self, path: Path, bounds: Bounds) -> list[dict[str, Any]]:
        if path == self.asset_path:
            return self._matching_features(bounds)
        self._active_geometry_mode = "bbox"
        return [feature for feature in self._load_features(path) if self._feature_intersects(feature, bounds)]

    def _dedupe_features(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for feature in features:
            deduped[self._dedupe_key(feature)] = feature
        return list(deduped.values())

    def _selected_bindings(self, options: Mapping[str, Any]) -> list[FaultSourceBinding]:
        if "sources" in options:
            requested = list(options["sources"])
        elif options.get("source") == "all":
            requested = [binding.source_id for binding in self.source_bindings]
        elif options.get("source") not in (None, ""):
            requested = [str(options["source"])]
        else:
            requested = [self.source_bindings[0].source_id]
        bindings = [binding for binding in self.source_bindings if binding.source_id in requested]
        if not bindings:
            raise ProviderOptionError(f"No configured active_faults sources matched {requested!r}.")
        return bindings

    def _binding_mode(self, binding: FaultSourceBinding, options: Mapping[str, Any]) -> str:
        return str(options.get("source_mode") or binding.source_mode)

    def _coerce_source_ids(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        try:
            return [str(part).strip() for part in value if str(part).strip()]
        except TypeError as exc:
            raise ProviderOptionError("active_faults sources must be a string or iterable.") from exc

    def _binding_intersects_parts(
        self,
        binding: FaultSourceBinding,
        bounds_parts: list[Bounds],
    ) -> bool:
        if binding.coverage_bounds is None:
            return True
        return any(self._bounds_intersect(part, binding.coverage_bounds) for part in bounds_parts)

    def _bounds_intersect(self, left: Bounds, right: Bounds) -> bool:
        return not (
            left.max_lon < right.min_lon
            or left.min_lon > right.max_lon
            or left.max_lat < right.min_lat
            or left.min_lat > right.max_lat
        )

    def _digest_for_path(self, path: Path) -> str:
        if path == self.asset_path and self._digest is not None:
            return self._digest
        if path not in self._digests_by_path:
            self._digests_by_path[path] = file_sha256_digest(path)
        digest = self._digests_by_path[path]
        if path == self.asset_path:
            self._digest = digest
        return digest

    def _binding_version(
        self,
        binding: FaultSourceBinding,
        options: Mapping[str, Any] | None = None,
    ) -> str:
        if binding.source_manifest is not None:
            return f"{self.version}@manifest:{binding.source_manifest.stable_hash()}"
        mode = self._binding_mode(binding, options or {})
        if mode == "live":
            return f"{self.version}@live:{binding.source_id}:normalizer:1"
        if binding.asset_path is None:
            return f"{self.version}@missing:{binding.source_id}"
        path = Path(binding.asset_path)
        if not path.exists():
            return f"{self.version}@missing:{path}"
        return source_version(self.version, self._digest_for_path(path))

    def _composite_source_version(
        self,
        bindings: list[FaultSourceBinding],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        if len(bindings) == 1:
            return self._binding_version(bindings[0], options)
        material = [self._binding_version(binding, options) for binding in bindings]
        return f"{self.version}@federated:{stable_hash(material)}"

    def _binding_cache_config(self, binding: FaultSourceBinding) -> dict[str, Any]:
        return {
            "source_id": binding.source_id,
            "source_mode": binding.source_mode,
            "asset_path": str(binding.asset_path) if binding.asset_path is not None else None,
            "source_manifest_hash": binding.source_manifest.stable_hash()
            if binding.source_manifest is not None
            else None,
            "normalizer_version": binding.source_manifest.normalizer_version
            if binding.source_manifest is not None
            else None,
            "supports_live": binding.supports_live,
            "coverage_bounds": binding.coverage_bounds.to_dict()
            if binding.coverage_bounds is not None
            else None,
        }

    def _source_info(
        self,
        binding: FaultSourceBinding,
        options: Mapping[str, Any],
        record_count: int | None,
    ) -> dict[str, Any]:
        manifest = binding.source_manifest
        attribution = source_attribution(binding.source_id)
        info = {
            "source_id": binding.source_id,
            "source_mode": self._binding_mode(binding, options),
            "source_version": self._binding_version(binding, options),
            "record_count": record_count,
            "asset_path": str(binding.asset_path) if binding.asset_path is not None else None,
            "source_manifest_path": str(binding.source_manifest_path)
            if binding.source_manifest_path is not None
            else None,
            "region": binding.region_name,
            "license": attribution["license"],
            "citation": attribution["citation"],
            "attribution": attribution["attribution"],
        }
        if manifest is not None:
            info.update(
                {
                    "source_url": manifest.source_url,
                    "source_manifest_hash": manifest.stable_hash(),
                    "retrieved_at": manifest.retrieved_at,
                    "license": manifest.license,
                    "citation": manifest.citation,
                    "attribution": manifest.attribution,
                    "normalizer_version": manifest.normalizer_version,
                    "coverage_status": manifest.coverage.get("status"),
                }
            )
        return info

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

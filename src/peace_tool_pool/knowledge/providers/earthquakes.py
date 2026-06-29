"""Local earthquake history provider."""

from __future__ import annotations

import csv
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..bounds import Bounds
from ..cache import stable_hash
from ..errors import OptionalDependencyError, ProviderOptionError
from ..sources.manifest import SourceManifest
from ..sources.registry import source_attribution
from ..sources.usgs_events import (
    UsgsFdsnEventAdapter,
    associated_id_set,
    event_identity_key,
)
from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class EarthquakeSourceBinding:
    source_id: str
    source_mode: str
    asset_path: str | Path | None = None
    source_manifest_path: str | Path | None = None
    source_manifest: SourceManifest | None = None
    adapter: Any | None = None
    supports_live: bool = False
    fallback_warning: str | None = None


class EarthquakeHistoryProvider:
    id = "earthquake_history"
    name = "Earthquake history"
    version = "1"
    output_keys = ("earthquake_history",)

    _selected_columns = (
        "time",
        "latitude",
        "longitude",
        "place",
        "mag",
        "magType",
        "depth",
        "type",
        "updated",
        "gap",
    )

    def __init__(
        self,
        asset_path: str | Path,
        default_max_records: int = 50,
        margin_degrees: float = 0.05,
        engine: str = "auto",
        source_id: str = "usgs_fdsn_events",
        source_mode: str = "legacy_asset",
        source_manifest_path: str | Path | None = None,
        source_manifest: SourceManifest | None = None,
        fallback_warning: str | None = None,
        source_bindings: list[EarthquakeSourceBinding] | None = None,
    ):
        self.asset_path = Path(asset_path)
        self.default_max_records = default_max_records
        self.margin_degrees = float(margin_degrees)
        self.engine = engine
        self.source_id = source_id
        self.source_mode = source_mode
        self.source_manifest_path = Path(source_manifest_path) if source_manifest_path else None
        self.source_manifest = source_manifest
        self.fallback_warning = fallback_warning
        self.last_warnings: list[str] = []
        self._rows: list[dict[str, str]] | None = None
        self._rows_by_path: dict[Path, list[dict[str, str]]] = {}
        self._frame: Any | None = None
        self._digest: str | None = None
        self._digests_by_path: dict[Path, str] = {}
        self._active_engine = "csv"
        self.source_bindings = source_bindings or [
            EarthquakeSourceBinding(
                source_id=source_id,
                source_mode=source_mode,
                asset_path=self.asset_path,
                source_manifest_path=source_manifest_path,
                source_manifest=source_manifest,
                supports_live=True,
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
            "engine": self.engine,
            "resolved_engine": self._resolved_engine_for_cache(),
            "margin_degrees": self.margin_degrees,
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

    def _resolved_engine_for_cache(self) -> str:
        if self.engine == "auto":
            return "pandas" if _dependency_available("pandas") else "csv"
        return self.engine

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
        failures: list[BaseException] = []
        for binding in selected_bindings:
            if binding.fallback_warning:
                self.last_warnings.append(binding.fallback_warning)
            try:
                records, engine = self._records_for_binding(binding, request, bounds_parts, options)
            except Exception as exc:  # noqa: BLE001 - per-source federation isolates failures.
                failures.append(exc)
                if len(selected_bindings) == 1:
                    raise
                self.last_warnings.append(
                    f"{self.id}: source {binding.source_id!r} failed: {type(exc).__name__}: {exc}"
                )
                source_infos.append(self._source_info(binding, options, 0, error=exc))
                continue
            for record in records:
                record["_source_id"] = binding.source_id
            matching.extend(records)
            source_infos.append(self._source_info(binding, options, len(records)))
            self._active_engine = "federated" if len(selected_bindings) > 1 else engine
        if not matching and failures:
            raise failures[0]
        matching = self._dedupe_rows(matching)
        matching.sort(
            key=lambda row: (str(row.get("time") or ""), self._float_or_default(row.get("mag"))),
            reverse=True,
        )
        limit = max_records_for_request(self.id, request, self.default_max_records)
        limited = matching[:limit]
        records = [self._shape_row(row) for row in limited]
        total = len(matching)
        truncated = total > len(records)
        if total:
            summary = f"Found {total} earthquakes within bounds; returning {len(records)} records."
        else:
            summary = "No earthquakes with configured filters were found within bounds."
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
                provenance=self._provenance(options, bounds_parts, selected_bindings, source_infos),
            )
        ]

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "source",
            "sources",
            "source_mode",
            "starttime",
            "endtime",
            "minmagnitude",
            "maxmagnitude",
            "reviewstatus",
            "catalog",
            "contributor",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ProviderOptionError(
                f"Unknown earthquake_history provider option keys: {sorted(unknown)}"
            )
        validated = dict(options)
        if "source" in validated and "sources" in validated:
            raise ProviderOptionError("Use either earthquake_history source or sources, not both.")
        available = {binding.source_id for binding in self.source_bindings}
        if "source" in validated and validated["source"] not in (None, "", "all"):
            source = str(validated["source"])
            if source not in available:
                raise ProviderOptionError(f"Unknown configured earthquake_history source: {source!r}")
            validated["source"] = source
        if "sources" in validated:
            sources = self._coerce_source_ids(validated["sources"])
            unknown_sources = [source for source in sources if source not in available]
            if unknown_sources:
                raise ProviderOptionError(
                    f"Unknown configured earthquake_history sources: {unknown_sources}"
                )
            validated["sources"] = sources
        if "source_mode" in validated:
            source_mode = str(validated["source_mode"])
            if source_mode not in {"local_mirror", "legacy_asset", "live"}:
                raise ProviderOptionError(f"Unsupported earthquake_history source_mode: {source_mode!r}")
            validated["source_mode"] = source_mode
        for key in ("minmagnitude", "maxmagnitude"):
            if key in validated and validated[key] not in (None, ""):
                try:
                    validated[key] = float(validated[key])
                except (TypeError, ValueError) as exc:
                    raise ProviderOptionError(f"{key} must be numeric.") from exc
        for key in ("starttime", "endtime", "reviewstatus", "catalog", "contributor"):
            if key in validated and validated[key] not in (None, ""):
                validated[key] = str(validated[key])
        return validated

    def warnings_for_cached_result(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        cached_items: list[KnowledgeItem],
    ) -> list[str]:
        del bounds_parts, cached_items
        warnings: list[str] = []
        try:
            options = self.validate_options(request.provider_options.get(self.id, {}))
            selected_bindings = self._selected_bindings(options)
        except ProviderOptionError:
            selected_bindings = self.source_bindings[:1]
        for binding in selected_bindings:
            if binding.fallback_warning:
                warnings.append(binding.fallback_warning)
        return warnings

    def _matching_rows_for_parts(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        options: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        binding = self.source_bindings[0]
        rows, _engine = self._records_for_binding(binding, request, bounds_parts, options)
        return rows

    def _matching_rows(self, request: KnowledgeRequest) -> list[dict[str, Any]]:
        if self.engine in {"auto", "pandas"}:
            try:
                return self._matching_rows_pandas(request)
            except OptionalDependencyError:
                if self.engine == "pandas":
                    raise
        if self.engine not in {"auto", "csv", "pandas"}:
            raise ValueError(f"Unsupported earthquake provider engine: {self.engine!r}")
        self._active_engine = "csv"
        rows = self._load_rows()
        return [row for row in rows if self._row_in_bounds(row, request)]

    def _matching_rows_pandas(self, request: KnowledgeRequest) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise OptionalDependencyError(
                "EarthquakeHistoryProvider pandas engine requires `uv sync --extra knowledge-local`."
            ) from exc

        frame = self._load_frame(pd)
        if request.bounds is None or "latitude" not in frame.columns or "longitude" not in frame.columns:
            self._active_engine = "pandas"
            return []
        latitude = pd.to_numeric(frame["latitude"], errors="coerce")
        longitude = pd.to_numeric(frame["longitude"], errors="coerce")
        bounds = request.bounds
        mask = (
            (latitude >= bounds.min_lat - self.margin_degrees)
            & (latitude <= bounds.max_lat + self.margin_degrees)
            & (longitude >= bounds.min_lon - self.margin_degrees)
            & (longitude <= bounds.max_lon + self.margin_degrees)
        )
        matching = frame.loc[mask].copy()
        self._active_engine = "pandas"
        return matching.to_dict(orient="records")

    def _load_frame(self, pandas_module: Any) -> Any:
        if self._frame is None:
            self._frame = pandas_module.read_csv(self.asset_path)
        return self._frame

    def _load_rows(self, asset_path: str | Path | None = None) -> list[dict[str, str]]:
        path = Path(asset_path) if asset_path is not None else self.asset_path
        if path == self.asset_path and self._rows is not None:
            return self._rows
        if path in self._rows_by_path:
            return self._rows_by_path[path]
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            rows = [dict(row) for row in csv.DictReader(file_obj)]
        self._rows_by_path[path] = rows
        if path == self.asset_path:
            self._rows = rows
        return rows

    def _row_in_bounds(self, row: dict[str, str], request: KnowledgeRequest) -> bool:
        bounds = request.bounds
        if bounds is None:
            return False
        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            bounds.min_lat - self.margin_degrees <= latitude <= bounds.max_lat + self.margin_degrees
            and bounds.min_lon - self.margin_degrees
            <= longitude
            <= bounds.max_lon + self.margin_degrees
        )

    def _row_matches_options(self, row: Mapping[str, Any], options: Mapping[str, Any]) -> bool:
        magnitude = self._float_or_default(row.get("mag"))
        if "minmagnitude" in options and magnitude < float(options["minmagnitude"]):
            return False
        if "maxmagnitude" in options and magnitude > float(options["maxmagnitude"]):
            return False
        time_value = str(row.get("time") or "")
        if "starttime" in options and time_value and time_value < str(options["starttime"]):
            return False
        if "endtime" in options and time_value and time_value > str(options["endtime"]):
            return False
        if "reviewstatus" in options:
            status = str(row.get("reviewstatus") or row.get("status") or "")
            if status != str(options["reviewstatus"]):
                return False
        if "catalog" in options and str(row.get("net") or "") != str(options["catalog"]):
            return False
        if "contributor" in options:
            sources = str(row.get("sources") or "")
            if str(options["contributor"]) not in sources.split(","):
                return False
        return True

    def _shape_row(self, row: dict[str, Any]) -> dict[str, Any]:
        columns = [column for column in self._selected_columns if column in row]
        if not columns:
            columns = list(row)
        return {column: self._coerce_value(row.get(column)) for column in columns}

    def _provenance(
        self,
        options: Mapping[str, Any],
        bounds_parts: list[Bounds],
        selected_bindings: list[EarthquakeSourceBinding] | None = None,
        source_infos: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected_bindings = selected_bindings or self.source_bindings[:1]
        primary = selected_bindings[0]
        primary_manifest = primary.source_manifest
        provenance: dict[str, Any] = {
            "asset_path": str(primary.asset_path or self.asset_path),
            "engine": self._active_engine,
            "margin_degrees": self.margin_degrees,
            "source_id": primary.source_id,
            "source_ids": [binding.source_id for binding in selected_bindings],
            "source_family": "earthquake_events",
            "source_mode": self._binding_mode(primary, options),
            "provider_options": dict(options),
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "dedupe_key": "association_set_overlap_exact",
            "sources": list(
                source_infos
                or [self._source_info(binding, options, None) for binding in selected_bindings]
            ),
        }
        if primary_manifest is not None:
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
                    "coverage_caveats": list(primary_manifest.coverage.get("notes") or []),
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
                    "coverage_caveats": [],
                }
            )
        return provenance

    def _records_for_binding(
        self,
        binding: EarthquakeSourceBinding,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        options: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], str]:
        mode = self._binding_mode(binding, options)
        if mode == "live":
            adapter = binding.adapter or UsgsFdsnEventAdapter()
            return adapter.live_records(self._source_filter_options(options), bounds_parts), "live"
        if binding.asset_path is None:
            raise ProviderOptionError(
                f"earthquake_history source {binding.source_id!r} has no local artifact."
            )
        path = Path(binding.asset_path)
        self._digest_for_path(path)
        records: list[dict[str, Any]] = []
        for bounds in bounds_parts:
            part_request = KnowledgeRequest(
                bounds=bounds,
                max_records=request.max_records,
                max_records_by_provider=dict(request.max_records_by_provider),
                provider_options={self.id: dict(options)},
            )
            for row in self._matching_rows_for_path(path, part_request):
                if self._row_matches_options(row, options):
                    records.append(dict(row))
        return self._dedupe_rows(records), self._active_engine

    def _matching_rows_for_path(
        self,
        path: Path,
        request: KnowledgeRequest,
    ) -> list[dict[str, Any]]:
        if path == self.asset_path:
            return self._matching_rows(request)
        self._active_engine = "csv"
        rows = self._load_rows(path)
        return [row for row in rows if self._row_in_bounds(row, request)]

    def _dedupe_rows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        identity_sets: list[set[str]] = []
        for record in records:
            ids = associated_id_set(record.get("ids")) or {event_identity_key(record)}
            replacement_index: int | None = None
            for index, existing_ids in enumerate(identity_sets):
                if ids & existing_ids:
                    replacement_index = index
                    break
            if replacement_index is None:
                deduped.append(record)
                identity_sets.append(ids)
            else:
                deduped[replacement_index] = record
                identity_sets[replacement_index] |= ids
        return deduped

    def _selected_bindings(self, options: Mapping[str, Any]) -> list[EarthquakeSourceBinding]:
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
            raise ProviderOptionError(f"No configured earthquake_history sources matched {requested!r}.")
        return bindings

    def _binding_mode(self, binding: EarthquakeSourceBinding, options: Mapping[str, Any]) -> str:
        return str(options.get("source_mode") or binding.source_mode)

    def _source_filter_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in options.items()
            if key not in {"source", "sources", "source_mode"}
        }

    def _coerce_source_ids(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        try:
            return [str(part).strip() for part in value if str(part).strip()]
        except TypeError as exc:
            raise ProviderOptionError("earthquake_history sources must be a string or iterable.") from exc

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
        binding: EarthquakeSourceBinding,
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
        bindings: list[EarthquakeSourceBinding],
        options: Mapping[str, Any] | None = None,
    ) -> str:
        if len(bindings) == 1:
            return self._binding_version(bindings[0], options)
        material = [self._binding_version(binding, options) for binding in bindings]
        return f"{self.version}@federated:{stable_hash(material)}"

    def _binding_cache_config(self, binding: EarthquakeSourceBinding) -> dict[str, Any]:
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
        }

    def _source_info(
        self,
        binding: EarthquakeSourceBinding,
        options: Mapping[str, Any],
        record_count: int | None,
        error: BaseException | None = None,
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
        if error is not None:
            info["error"] = f"{type(error).__name__}: {error}"
        return info

    def _coerce_value(self, value: Any) -> Any:
        if value is None:
            return None
        item_method = getattr(value, "item", None)
        if callable(item_method) and not isinstance(value, (str, bytes)):
            value = item_method()
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        value = str(value)
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        if number.is_integer() and "." not in stripped and "e" not in stripped.lower():
            return int(number)
        return number

    def _float_or_default(self, value: str | None) -> float:
        try:
            return float(value or 0)
        except ValueError:
            return 0.0

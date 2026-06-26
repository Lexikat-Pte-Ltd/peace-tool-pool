"""Local earthquake history provider."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping

from ..bounds import Bounds
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
        self._frame: Any | None = None
        self._digest: str | None = None
        self._active_engine = "csv"

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        if self.source_manifest is not None:
            return f"{self.version}@manifest:{self.source_manifest.stable_hash()}"
        if self.source_mode == "live":
            return f"{self.version}@live:{self.source_id}:normalizer:1"
        if not self.asset_path.exists():
            return f"{self.version}@missing:{self.asset_path}"
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        return source_version(self.version, self._digest)

    def source_version_for_options(self, options: Mapping[str, Any]) -> str:
        source_mode = str(options.get("source_mode") or self.source_mode)
        if source_mode == "live":
            return f"{self.version}@live:{self.source_id}:normalizer:1"
        return self.source_version()

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
        if self.fallback_warning:
            self.last_warnings.append(self.fallback_warning)
        options = self.validate_options(request.provider_options.get(self.id, {}))
        if options["source_mode"] == "live":
            matching = UsgsFdsnEventAdapter().live_records(options, bounds_parts)
            self._active_engine = "live"
        else:
            if self._digest is None:
                self._digest = file_sha256_digest(self.asset_path)
            matching = self._matching_rows_for_parts(request, bounds_parts, options)
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
                provenance=self._provenance(options, bounds_parts),
            )
        ]

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "source",
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
        source = str(validated.get("source") or self.source_id)
        if source != self.source_id:
            raise ProviderOptionError(
                f"earthquake_history source {source!r} does not match configured source {self.source_id!r}."
            )
        validated["source"] = source
        source_mode = str(validated.get("source_mode") or self.source_mode)
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
        del request, bounds_parts, cached_items
        warnings: list[str] = []
        if self.fallback_warning:
            warnings.append(self.fallback_warning)
        return warnings

    def _matching_rows_for_parts(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        options: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        identity_sets: list[set[str]] = []
        for bounds in bounds_parts:
            part_request = KnowledgeRequest(
                bounds=bounds,
                max_records=request.max_records,
                max_records_by_provider=dict(request.max_records_by_provider),
                provider_options={self.id: dict(options)},
            )
            for row in self._matching_rows(part_request):
                if not self._row_matches_options(row, options):
                    continue
                ids = associated_id_set(row.get("ids")) or {event_identity_key(row)}
                replacement_index: int | None = None
                for index, existing_ids in enumerate(identity_sets):
                    if ids & existing_ids:
                        replacement_index = index
                        break
                if replacement_index is None:
                    deduped.append(row)
                    identity_sets.append(ids)
                else:
                    deduped[replacement_index] = row
                    identity_sets[replacement_index] |= ids
        return deduped

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

    def _load_rows(self) -> list[dict[str, str]]:
        if self._rows is not None:
            return self._rows
        with self.asset_path.open("r", encoding="utf-8", newline="") as file_obj:
            self._rows = [dict(row) for row in csv.DictReader(file_obj)]
        return self._rows

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
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {
            "asset_path": str(self.asset_path),
            "engine": self._active_engine,
            "margin_degrees": self.margin_degrees,
            "source_id": self.source_id,
            "source_family": "earthquake_events",
            "source_mode": options.get("source_mode", self.source_mode),
            "provider_options": dict(options),
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "dedupe_key": "usgs_association_set_overlap",
        }
        if self.source_manifest is not None:
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
                    "coverage_caveats": list(self.source_manifest.coverage.get("notes") or []),
                }
            )
        else:
            attribution = source_attribution(self.source_id)
            provenance.update(
                {
                    "source_url": None,
                    "source_version": self.source_version_for_options(options),
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

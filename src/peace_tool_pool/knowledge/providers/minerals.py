"""Mineral-occurrence provider (Ontario Geological Survey MDI prototype).

A bbox-queryable knowledge provider over a regional mineral-occurrence service.
This is the first source in the ``mineral_occurrences`` family and the first
exercise of the deferred Approach-C "region-aware source selection" seam: the
provider holds a coverage region and only queries the source when the request
bbox intersects it. A second regional source (e.g. Quebec SIGEOM) lands as an
appended source + coverage region, not a rewrite.

Live-only and registered non-default, so it never touches the network unless a
caller explicitly includes ``mineral_occurrences`` (mirroring how the heavy
Earth Engine providers are explicit-only).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..bounds import Bounds
from ..errors import ProviderOptionError
from ..sources.ogs_minerals import OgsMineralOccurrenceAdapter, SELECTED_COLUMNS, normalize_features
from ..sources.registry import source_attribution
from ..types import KnowledgeItem, KnowledgeRequest
from .base import max_records_for_request


class MineralOccurrenceProvider:
    id = "mineral_occurrences"
    name = "Mineral occurrences"
    version = "1"
    output_keys = ("mineral_occurrences",)

    def __init__(
        self,
        adapter: OgsMineralOccurrenceAdapter | None = None,
        source_id: str = "ontario_mineral_deposit_inventory",
        coverage_bounds: Bounds | None = None,
        region_name: str = "Ontario",
        default_max_records: int = 50,
        source_mode: str = "live",
    ):
        self.adapter = adapter or OgsMineralOccurrenceAdapter()
        self.source_id = source_id
        self.coverage_bounds = coverage_bounds
        self.region_name = region_name
        self.default_max_records = default_max_records
        self.source_mode = source_mode
        self.last_warnings: list[str] = []

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        return f"{self.version}@live:{self.source_id}"

    def source_version_for_options(self, options: Mapping[str, Any]) -> str:
        del options
        return self.source_version()

    def cache_config(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_mode": self.source_mode,
            "endpoint": self.adapter.endpoint,
            "normalizer_version": self.adapter.normalizer_version,
            "region": self.region_name,
        }

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"source", "source_mode"}
        unknown = set(options) - allowed
        if unknown:
            raise ProviderOptionError(
                f"Unknown mineral_occurrences provider option keys: {sorted(unknown)}"
            )
        validated = dict(options)
        source = str(validated.get("source") or self.source_id)
        if source != self.source_id:
            raise ProviderOptionError(
                f"mineral_occurrences source {source!r} does not match configured "
                f"source {self.source_id!r}."
            )
        validated["source"] = source
        source_mode = str(validated.get("source_mode") or self.source_mode)
        if source_mode != "live":
            raise ProviderOptionError(
                "mineral_occurrences currently supports only source_mode='live' "
                "(no offline mirror yet)."
            )
        validated["source_mode"] = source_mode
        return validated

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

        in_region = [part for part in bounds_parts if self._in_coverage(part)]
        out_of_region = len(in_region) < len(bounds_parts)
        coverage_caveats: list[str] = []
        records: list[dict[str, Any]] = []

        if out_of_region:
            caveat = (
                f"Query falls outside the {self.region_name} coverage of "
                f"{self.source_id!r}; no mineral-occurrence source is wired for that "
                "region yet, so this is not evidence that no occurrences exist."
            )
            coverage_caveats.append(caveat)
            self.last_warnings.append(caveat)

        for part in in_region:
            records.extend(normalize_features(self.adapter.query_bbox(part)))
        records = self._dedupe(records)
        records.sort(key=lambda record: str(record.get("name") or ""))

        snapshot_caveat = (
            f"{self.source_id!r} is queried live; the public layer may be a dated "
            "snapshot, so absence/counts reflect source vintage and coverage."
        )
        coverage_caveats.append(snapshot_caveat)

        limit = max_records_for_request(self.id, request, self.default_max_records)
        total = len(records)
        limited = [self._shape_record(record) for record in records[:limit]]
        truncated = total > len(limited)

        if total == 0 and not out_of_region:
            absence = (
                f"No mineral occurrences intersected the bounds within {self.region_name}; "
                "this is not evidence that no occurrences exist (source vintage/coverage)."
            )
            self.last_warnings.append(absence)
            summary = absence
        elif total == 0:
            summary = (
                f"No mineral occurrences returned; query was outside {self.region_name} coverage."
            )
        else:
            summary = (
                f"Found {total} mineral occurrences intersecting bounds in "
                f"{self.region_name}; returning {len(limited)} records."
            )

        return [
            KnowledgeItem(
                id=f"{self.id}:{self.id}",
                key=self.id,
                provider=self.id,
                value=limited,
                summary=summary,
                source=self.adapter.endpoint,
                record_count=total,
                truncated=truncated,
                provenance=self._provenance(options, bounds_parts, coverage_caveats),
            )
        ]

    def warnings_for_cached_result(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        cached_items: list[KnowledgeItem],
    ) -> list[str]:
        del request
        warnings: list[str] = []
        in_region = [part for part in bounds_parts if self._in_coverage(part)]
        if len(in_region) < len(bounds_parts):
            warnings.append(
                f"Query falls outside the {self.region_name} coverage of "
                f"{self.source_id!r}; no mineral-occurrence source is wired for that "
                "region yet, so this is not evidence that no occurrences exist."
            )
        if cached_items and (cached_items[0].record_count or 0) == 0 and in_region:
            warnings.append(
                f"No mineral occurrences intersected the bounds within {self.region_name}; "
                "this is not evidence that no occurrences exist (source vintage/coverage)."
            )
        return warnings

    def _in_coverage(self, bounds: Bounds) -> bool:
        if self.coverage_bounds is None:
            return True
        return not (
            bounds.max_lon < self.coverage_bounds.min_lon
            or bounds.min_lon > self.coverage_bounds.max_lon
            or bounds.max_lat < self.coverage_bounds.min_lat
            or bounds.min_lat > self.coverage_bounds.max_lat
        )

    def _dedupe(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for record in records:
            key = (record.get("name"), record.get("longitude"), record.get("latitude"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _shape_record(self, record: dict[str, Any]) -> dict[str, Any]:
        shaped = {column: record.get(column) for column in SELECTED_COLUMNS if column in record}
        for key in ("longitude", "latitude"):
            if key in record:
                shaped[key] = record[key]
        return shaped

    def _provenance(
        self,
        options: Mapping[str, Any],
        bounds_parts: list[Bounds],
        coverage_caveats: list[str],
    ) -> dict[str, Any]:
        attribution = source_attribution(self.source_id)
        return {
            "source_id": self.source_id,
            "source_family": "mineral_occurrences",
            "source_mode": options.get("source_mode", self.source_mode),
            "source_url": self.adapter.endpoint,
            "source_version": self.source_version(),
            "retrieved_at": None,
            "region": self.region_name,
            "provider_options": dict(options),
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "dedupe_key": "name_longitude_latitude",
            "normalizer_version": self.adapter.normalizer_version,
            "coverage_status": "live-regional-service",
            "coverage_caveats": list(coverage_caveats),
            "license": attribution["license"],
            "citation": attribution["citation"],
            "attribution": attribution["attribution"],
        }

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

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..bounds import Bounds
from ..cache import stable_hash
from ..errors import ProviderOptionError
from ..sources.ogs_minerals import OgsMineralOccurrenceAdapter, SELECTED_COLUMNS, normalize_features
from ..sources.registry import source_attribution
from ..types import KnowledgeItem, KnowledgeRequest
from .base import max_records_for_request


@dataclass
class MineralSourceBinding:
    source_id: str
    adapter: Any
    normalize: Callable[[Mapping[str, Any]], list[dict[str, Any]]]
    coverage_bounds: Bounds | None = None
    region_name: str = "regional source"
    source_mode: str = "live"


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
        source_bindings: list[MineralSourceBinding] | None = None,
    ):
        self.adapter = adapter or OgsMineralOccurrenceAdapter()
        self.source_id = source_id
        self.coverage_bounds = coverage_bounds
        self.region_name = region_name
        self.default_max_records = default_max_records
        self.source_mode = source_mode
        self.last_warnings: list[str] = []
        self.source_bindings = source_bindings or [
            MineralSourceBinding(
                source_id=source_id,
                adapter=self.adapter,
                normalize=normalize_features,
                coverage_bounds=coverage_bounds,
                region_name=region_name,
                source_mode=source_mode,
            )
        ]

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        return self._composite_source_version(self.source_bindings)

    def source_version_for_options(self, options: Mapping[str, Any]) -> str:
        return self._composite_source_version(self._selected_bindings(options))

    def cache_config(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_mode": self.source_mode,
            "endpoint": self.adapter.endpoint,
            "normalizer_version": self.adapter.normalizer_version,
            "region": self.region_name,
            "source_bindings": [self._binding_cache_config(binding) for binding in self.source_bindings],
        }

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"source", "sources", "source_mode"}
        unknown = set(options) - allowed
        if unknown:
            raise ProviderOptionError(
                f"Unknown mineral_occurrences provider option keys: {sorted(unknown)}"
            )
        validated = dict(options)
        if "source" in validated and "sources" in validated:
            raise ProviderOptionError("Use either mineral_occurrences source or sources, not both.")
        available = {binding.source_id for binding in self.source_bindings}
        if "source" in validated and validated["source"] not in (None, "", "all"):
            source = str(validated["source"])
            if source not in available:
                raise ProviderOptionError(f"Unknown configured mineral_occurrences source: {source!r}")
            validated["source"] = source
        if "sources" in validated:
            sources = self._coerce_source_ids(validated["sources"])
            unknown_sources = [source for source in sources if source not in available]
            if unknown_sources:
                raise ProviderOptionError(
                    f"Unknown configured mineral_occurrences sources: {unknown_sources}"
                )
            validated["sources"] = sources
        if "source_mode" in validated:
            source_mode = str(validated["source_mode"])
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
        selected_bindings = self._selected_bindings(options)
        coverage_caveats: list[str] = []
        records: list[dict[str, Any]] = []
        source_infos: list[dict[str, Any]] = []

        for binding in selected_bindings:
            in_region = [part for part in bounds_parts if self._binding_covers_part(binding, part)]
            if not in_region:
                caveat = (
                    f"Query falls outside the {binding.region_name} coverage of "
                    f"{binding.source_id!r}; this is not evidence that no occurrences exist."
                )
                coverage_caveats.append(caveat)
                self.last_warnings.append(caveat)
                source_infos.append(self._source_info(binding, 0))
                continue
            binding_records: list[dict[str, Any]] = []
            for part in in_region:
                binding_records.extend(binding.normalize(binding.adapter.query_bbox(part)))
            for record in binding_records:
                record["_source_id"] = binding.source_id
            records.extend(binding_records)
            source_infos.append(self._source_info(binding, len(binding_records)))
        records = self._dedupe(records)
        records.sort(key=lambda record: str(record.get("name") or ""))

        for binding in selected_bindings:
            coverage_caveats.append(
                f"{binding.source_id!r} is queried live; absence/counts reflect source vintage and coverage."
            )

        limit = max_records_for_request(self.id, request, self.default_max_records)
        total = len(records)
        limited = [self._shape_record(record) for record in records[:limit]]
        truncated = total > len(limited)

        covered_any_part = any(
            self._binding_covers_part(binding, part)
            for binding in selected_bindings
            for part in bounds_parts
        )
        if total == 0 and covered_any_part:
            absence = (
                "No mineral occurrences intersected the bounds within configured source coverage; "
                "this is not evidence that no occurrences exist (source vintage/coverage)."
            )
            self.last_warnings.append(absence)
            summary = absence
        elif total == 0:
            summary = (
                "No mineral occurrences returned; query was outside configured source coverage."
            )
        else:
            summary = (
                f"Found {total} mineral occurrences intersecting bounds in "
                "configured source coverage; returning "
                f"{len(limited)} records."
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
                provenance=self._provenance(
                    options,
                    bounds_parts,
                    coverage_caveats,
                    selected_bindings,
                    source_infos,
                ),
            )
        ]

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
            selected_bindings = self.source_bindings
        covered = [
            (binding, part)
            for binding in selected_bindings
            for part in bounds_parts
            if self._binding_covers_part(binding, part)
        ]
        for binding in selected_bindings:
            if not any(item_binding is binding for item_binding, _part in covered):
                warnings.append(
                    f"Query falls outside the {binding.region_name} coverage of "
                    f"{binding.source_id!r}; this is not evidence that no occurrences exist."
                )
        if cached_items and (cached_items[0].record_count or 0) == 0 and covered:
            warnings.append(
                "No mineral occurrences intersected the bounds within configured source coverage; "
                "this is not evidence that no occurrences exist (source vintage/coverage)."
            )
        return warnings

    def _in_coverage(self, bounds: Bounds) -> bool:
        return self._binding_covers_part(self.source_bindings[0], bounds)

    def _binding_covers_part(self, binding: MineralSourceBinding, bounds: Bounds) -> bool:
        coverage = binding.coverage_bounds
        if coverage is None:
            return True
        return not (
            bounds.max_lon < coverage.min_lon
            or bounds.min_lon > coverage.max_lon
            or bounds.max_lat < coverage.min_lat
            or bounds.min_lat > coverage.max_lat
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
        selected_bindings: list[MineralSourceBinding] | None = None,
        source_infos: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected_bindings = selected_bindings or self.source_bindings
        primary = selected_bindings[0]
        attribution = source_attribution(primary.source_id)
        return {
            "source_id": primary.source_id,
            "source_ids": [binding.source_id for binding in selected_bindings],
            "source_family": "mineral_occurrences",
            "source_mode": options.get("source_mode", primary.source_mode),
            "source_url": primary.adapter.endpoint,
            "source_version": self._binding_version(primary),
            "retrieved_at": None,
            "region": primary.region_name,
            "provider_options": dict(options),
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "dedupe_key": "name_longitude_latitude",
            "normalizer_version": primary.adapter.normalizer_version,
            "coverage_status": "live-regional-service",
            "coverage_caveats": list(coverage_caveats),
            "sources": list(
                source_infos
                or [self._source_info(binding, None) for binding in selected_bindings]
            ),
            "license": attribution["license"],
            "citation": attribution["citation"],
            "attribution": attribution["attribution"],
        }

    def _selected_bindings(self, options: Mapping[str, Any]) -> list[MineralSourceBinding]:
        if "sources" in options:
            requested = list(options["sources"])
        elif options.get("source") == "all":
            requested = [binding.source_id for binding in self.source_bindings]
        elif options.get("source") not in (None, ""):
            requested = [str(options["source"])]
        else:
            requested = [binding.source_id for binding in self.source_bindings]
        bindings = [binding for binding in self.source_bindings if binding.source_id in requested]
        if not bindings:
            raise ProviderOptionError(f"No configured mineral_occurrences sources matched {requested!r}.")
        return bindings

    def _coerce_source_ids(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        try:
            return [str(part).strip() for part in value if str(part).strip()]
        except TypeError as exc:
            raise ProviderOptionError("mineral_occurrences sources must be a string or iterable.") from exc

    def _binding_version(self, binding: MineralSourceBinding) -> str:
        return f"{self.version}@live:{binding.source_id}:normalizer:{binding.adapter.normalizer_version}"

    def _composite_source_version(self, bindings: list[MineralSourceBinding]) -> str:
        if len(bindings) == 1:
            return self._binding_version(bindings[0])
        material = [self._binding_version(binding) for binding in bindings]
        return f"{self.version}@federated:{stable_hash(material)}"

    def _binding_cache_config(self, binding: MineralSourceBinding) -> dict[str, Any]:
        return {
            "source_id": binding.source_id,
            "source_mode": binding.source_mode,
            "endpoint": binding.adapter.endpoint,
            "normalizer_version": binding.adapter.normalizer_version,
            "region": binding.region_name,
            "coverage_bounds": binding.coverage_bounds.to_dict()
            if binding.coverage_bounds is not None
            else None,
        }

    def _source_info(
        self,
        binding: MineralSourceBinding,
        record_count: int | None,
    ) -> dict[str, Any]:
        attribution = source_attribution(binding.source_id)
        return {
            "source_id": binding.source_id,
            "source_mode": binding.source_mode,
            "source_url": binding.adapter.endpoint,
            "source_version": self._binding_version(binding),
            "record_count": record_count,
            "region": binding.region_name,
            "normalizer_version": binding.adapter.normalizer_version,
            "coverage_status": "live-regional-service",
            "license": attribution["license"],
            "citation": attribution["citation"],
            "attribution": attribution["attribution"],
        }

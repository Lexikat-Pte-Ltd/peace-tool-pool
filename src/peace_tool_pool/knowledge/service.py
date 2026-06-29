"""Local knowledge service facade."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .bounds import Bounds, split_antimeridian
from .cache import KnowledgeCache, stable_hash
from .config import KnowledgeConfig
from .errors import MissingAssetError, OptionalDependencyError, ProviderError, ProviderOptionError
from .providers.earthengine import (
    EarthEngineLandcoverProvider,
    EarthEnginePopulationDensityProvider,
)
from .providers.earthquakes import EarthquakeHistoryProvider, EarthquakeSourceBinding
from .providers.faults import ActiveFaultProvider, FaultSourceBinding
from .providers.rock import RockLookupProvider
from .providers.semantic_k2 import SemanticK2Provider, SentenceTransformerSemanticBackend
from .sources.manifest import SourceManifest, find_latest_manifest
from .types import KnowledgeBundle, KnowledgeItem, KnowledgeRequest, LegendEnrichment


ProviderFactory = Callable[[], Any]
SupportPredicate = Callable[[KnowledgeRequest], bool]


@dataclass
class ProviderRegistration:
    id: str
    name: str
    output_keys: tuple[str, ...]
    factory: ProviderFactory
    supports: SupportPredicate
    default_enabled: bool = True


class KnowledgeService:
    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        providers: Iterable[Any] | None = None,
        provider_registrations: Iterable[ProviderRegistration] | None = None,
    ):
        self.config = config or KnowledgeConfig.from_env()
        self.cache = KnowledgeCache(self.config)
        self._provider_instances: dict[str, Any] = {}
        if provider_registrations is not None:
            self._registrations = list(provider_registrations)
        elif providers is not None:
            self._registrations = [self._registration_for_provider(provider) for provider in providers]
        else:
            self._registrations = self._default_registrations()

    @classmethod
    def from_env(cls, base_dir: str | Path | None = None) -> "KnowledgeService":
        return cls(config=KnowledgeConfig.from_env(base_dir=base_dir))

    def query(self, request: KnowledgeRequest) -> KnowledgeBundle:
        bounds_parts = [request.bounds] if request.bounds is not None else []
        return self._query(request, bounds_parts=bounds_parts, raw_extent=None)

    def _query(
        self,
        request: KnowledgeRequest,
        *,
        bounds_parts: list[Bounds],
        raw_extent: dict[str, Any] | None,
    ) -> KnowledgeBundle:
        warnings: list[str] = []
        registrations, explicit_ids = self._select_registrations(request, warnings)
        request = self._request_with_validated_provider_options(request, registrations, warnings)
        items: list[KnowledgeItem] = []
        provider_versions: dict[str, str] = {}
        trace_events: list[dict[str, Any]] = []
        explicit_success_ids: set[str] = set()
        explicit_failures: list[tuple[str, BaseException]] = []

        for registration in registrations:
            explicit = registration.id in explicit_ids
            try:
                provider = self._provider_for_registration(registration)
                provider_items, provider_version, cache_hit = self._query_provider(
                    provider,
                    request,
                    bounds_parts=bounds_parts,
                )
                items.extend(provider_items)
                warnings.extend(getattr(provider, "last_warnings", []) or [])
                if explicit:
                    explicit_success_ids.add(registration.id)
                provider_versions[provider.id] = provider_version
                item_provenance = provider_items[0].provenance if provider_items else {}
                trace_events.append(
                    {
                        "provider": provider.id,
                        "source_id": item_provenance.get(
                            "source_id", getattr(provider, "source_id", None)
                        ),
                        "source_ids": item_provenance.get("source_ids"),
                        "source_mode": item_provenance.get(
                            "source_mode", getattr(provider, "source_mode", None)
                        ),
                        "source_version": provider_version,
                        "cache_hit": cache_hit,
                        "record_count": sum(item.record_count or 0 for item in provider_items),
                        "truncated": any(item.truncated for item in provider_items),
                        "parts": len(bounds_parts),
                        "warning_count": len(getattr(provider, "last_warnings", []) or []),
                    }
                )
            except ProviderOptionError:
                raise
            except (MissingAssetError, OptionalDependencyError) as exc:
                if explicit:
                    explicit_failures.append((registration.id, exc))
                    warnings.append(f"{registration.id}: {exc}")
                    continue
                warnings.append(f"{registration.id}: configured provider asset or dependency is unavailable.")
            except Exception as exc:  # noqa: BLE001 - provider boundaries should isolate failures.
                provider_error = ProviderError(f"Provider {registration.id!r} failed: {exc}")
                if explicit:
                    explicit_failures.append((registration.id, provider_error))
                    warnings.append(
                        f"{registration.id}: provider failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                warnings.append(f"{registration.id}: provider failed: {type(exc).__name__}: {exc}")

        if explicit_ids and not explicit_success_ids and explicit_failures:
            raise explicit_failures[0][1]

        trace = {
            "trace_id": request.trace_id,
            "providers": trace_events,
            "bounds_parts": [part.to_dict() for part in bounds_parts],
            "raw_extent": raw_extent,
        }
        return KnowledgeBundle(
            bounds=request.bounds if len(bounds_parts) <= 1 else None,
            items=items,
            selected_item_ids=None,
            warnings=warnings,
            provider_versions=provider_versions,
            trace=trace,
        )

    def query_bounds(
        self,
        bounds: Bounds,
        include: tuple[str, ...] = (),
        exclude: tuple[str, ...] = (),
        max_records: int | None = None,
        provider_options: dict[str, dict[str, Any]] | None = None,
    ) -> KnowledgeBundle:
        return self.query(
            KnowledgeRequest(
                bounds=bounds,
                include=include,
                exclude=exclude,
                max_records=max_records,
                provider_options=provider_options or {},
            )
        )

    def query_extent(
        self,
        *,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        include: tuple[str, ...] = (),
        exclude: tuple[str, ...] = (),
        max_records: int | None = None,
        provider_options: dict[str, dict[str, Any]] | None = None,
    ) -> KnowledgeBundle:
        parts = split_antimeridian(min_lon, min_lat, max_lon, max_lat)
        request = KnowledgeRequest(
            bounds=parts[0],
            include=include,
            exclude=exclude,
            max_records=max_records,
            provider_options=provider_options or {},
        )
        raw_extent = {
            "min_lon": float(min_lon),
            "min_lat": float(min_lat),
            "max_lon": float(max_lon),
            "max_lat": float(max_lat),
            "crs": parts[0].crs,
        }
        return self._query(request, bounds_parts=parts, raw_extent=raw_extent)

    def enrich_legend_label(self, label: str) -> LegendEnrichment:
        bundle = self.query(
            KnowledgeRequest(legend_labels=[label], include=("rock_type", "rock_age"))
        )
        by_key = bundle.items_by_key()
        lithology = self._legend_value(by_key.get("rock_type", []))
        stratigraphic_age = self._legend_value(by_key.get("rock_age", []))
        return LegendEnrichment(
            label=label,
            lithology=lithology,
            stratigraphic_age=stratigraphic_age,
            items=bundle.items,
            warnings=bundle.warnings,
        )

    def query_map(
        self,
        metadata: Mapping[str, Any],
        question: str | None = None,
        *,
        include: tuple[str, ...] = (),
        exclude: tuple[str, ...] = (),
        max_records: int | None = None,
    ) -> KnowledgeBundle:
        """Query knowledge for a processed map.

        ``metadata`` supplies the spatial extent via either ``"bounds"`` (a
        :class:`Bounds` or ``{min_lon, min_lat, max_lon, max_lat[, crs]}`` dict)
        or ``"georef"`` (``{crs, gcps, pixel_extent}``, reprojected via the
        ``georef`` package — requires the ``geo`` extra). Legend labels are taken
        from ``"legend_labels"`` or extracted from a ``"legend"`` block (native
        dict or PEACE ``[id, {...}]`` pair form).
        """
        bounds = self._bounds_from_metadata(metadata)
        legend_labels = self._legend_labels_from_metadata(metadata)
        if bounds is None and not legend_labels:
            raise ValueError(
                "query_map requires 'bounds', 'georef', or 'legend'/'legend_labels' "
                "in metadata."
            )
        return self.query(
            KnowledgeRequest(
                bounds=bounds,
                legend_labels=legend_labels,
                query_text=question,
                include=include,
                exclude=exclude,
                max_records=max_records,
            )
        )

    @staticmethod
    def _bounds_from_metadata(metadata: Mapping[str, Any]) -> Bounds | None:
        raw = metadata.get("bounds")
        if isinstance(raw, Bounds):
            return raw
        if isinstance(raw, Mapping):
            return Bounds(**raw)
        georef = metadata.get("georef")
        if isinstance(georef, Mapping):
            # Lazy import: reprojection needs the optional 'geo' extra (pyproj).
            from ..georef import GroundControlPoint, georeference_bounds

            gcps = []
            for gcp in georef["gcps"]:
                if isinstance(gcp, Mapping):
                    gcps.append(GroundControlPoint(**gcp))
                else:
                    px, py, wx, wy = gcp
                    gcps.append(
                        GroundControlPoint(pixel_x=px, pixel_y=py, world_x=wx, world_y=wy)
                    )
            ref = georeference_bounds(
                crs=georef["crs"],
                gcps=gcps,
                pixel_extent=tuple(georef["pixel_extent"]),
            )
            return ref.bounds
        return None

    @staticmethod
    def _legend_labels_from_metadata(metadata: Mapping[str, Any]) -> list[str]:
        explicit = metadata.get("legend_labels")
        if explicit:
            return [str(x) for x in explicit if str(x).strip()]
        labels: list[str] = []
        for entry in metadata.get("legend", []) or []:
            text: Any = None
            if isinstance(entry, Mapping):
                text = entry.get("label") or entry.get("text")
            elif isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(
                entry[1], Mapping
            ):
                text = entry[1].get("text") or entry[1].get("label")
            if text and str(text).strip():
                labels.append(str(text))
        return labels

    def _default_registrations(self) -> list[ProviderRegistration]:
        semantic_backend_factory = self._semantic_backend_factory()
        return [
            ProviderRegistration(
                id="rock_type",
                name="Rock type",
                output_keys=("rock_type",),
                factory=lambda: RockLookupProvider(
                    asset_path=self.config.resolved_k2_rock_type_path,
                    provider_id="rock_type",
                    output_key="rock_type",
                    name="Rock type",
                ),
                supports=lambda request: bool(request.legend_labels),
            ),
            ProviderRegistration(
                id="rock_age",
                name="Rock age",
                output_keys=("rock_age",),
                factory=lambda: RockLookupProvider(
                    asset_path=self.config.resolved_k2_rock_age_path,
                    provider_id="rock_age",
                    output_key="rock_age",
                    name="Rock age",
                ),
                supports=lambda request: bool(request.legend_labels),
            ),
            ProviderRegistration(
                id="earthquake_history",
                name="Earthquake history",
                output_keys=("earthquake_history",),
                factory=self._earthquake_provider_factory,
                supports=lambda request: request.bounds is not None,
            ),
            ProviderRegistration(
                id="active_faults",
                name="Active faults",
                output_keys=("active_faults",),
                factory=self._active_fault_provider_factory,
                supports=lambda request: request.bounds is not None,
            ),
            ProviderRegistration(
                id="mineral_occurrences",
                name="Mineral occurrences",
                output_keys=("mineral_occurrences",),
                factory=self._mineral_occurrence_provider_factory,
                supports=lambda request: request.bounds is not None,
                default_enabled=False,
            ),
            ProviderRegistration(
                id="landcover_distribution",
                name="Landcover distribution",
                output_keys=("landcover_distribution",),
                factory=lambda: EarthEngineLandcoverProvider(
                    dataset_id=self.config.earthengine_landcover_dataset_id,
                    project=self.config.earthengine_project,
                    scale=self.config.earthengine_scale,
                    max_pixels=self.config.earthengine_max_pixels,
                ),
                supports=lambda request: request.bounds is not None,
                default_enabled=False,
            ),
            ProviderRegistration(
                id="population_density",
                name="Population density",
                output_keys=("population_density",),
                factory=lambda: EarthEnginePopulationDensityProvider(
                    dataset_id=self.config.earthengine_population_dataset_id,
                    project=self.config.earthengine_project,
                    scale=self.config.earthengine_scale,
                    max_pixels=self.config.earthengine_max_pixels,
                ),
                supports=lambda request: request.bounds is not None,
                default_enabled=False,
            ),
            ProviderRegistration(
                id="rock_knowledge",
                name="K2 semantic rock knowledge",
                output_keys=("rock_knowledge",),
                factory=lambda: SemanticK2Provider(
                    provider_id="rock_knowledge",
                    name="K2 semantic rock knowledge",
                    output_key="rock_knowledge",
                    asset_path=self.config.resolved_k2_rock_detail_path,
                    query_field="id",
                    answer_field="answer",
                    query_template="{query}",
                    backend_factory=semantic_backend_factory,
                    default_top_k=self.config.semantic_top_k,
                    min_score=self.config.semantic_min_score,
                    batch_size=self.config.semantic_batch_size,
                    model_name=self.config.semantic_model_name,
                    model_revision=self.config.semantic_model_revision,
                    device=self.config.semantic_device,
                    local_files_only=self.config.semantic_local_files_only,
                ),
                supports=lambda request: bool(request.query_text),
                default_enabled=False,
            ),
            ProviderRegistration(
                id="component_usage_knowledge",
                name="K2 component usage knowledge",
                output_keys=("component_usage_knowledge",),
                factory=lambda: SemanticK2Provider(
                    provider_id="component_usage_knowledge",
                    name="K2 component usage knowledge",
                    output_key="component_usage_knowledge",
                    asset_path=self.config.resolved_k2_usage_path,
                    query_field="question",
                    answer_field="answer",
                    query_template="What is the function of {query} in geologic maps?",
                    backend_factory=semantic_backend_factory,
                    default_top_k=self.config.semantic_top_k,
                    min_score=self.config.semantic_min_score,
                    batch_size=self.config.semantic_batch_size,
                    model_name=self.config.semantic_model_name,
                    model_revision=self.config.semantic_model_revision,
                    device=self.config.semantic_device,
                    local_files_only=self.config.semantic_local_files_only,
                ),
                supports=lambda request: bool(request.query_text),
                default_enabled=False,
            ),
            ProviderRegistration(
                id="downstream_task_knowledge",
                name="K2 downstream task knowledge",
                output_keys=("downstream_task_knowledge",),
                factory=lambda: SemanticK2Provider(
                    provider_id="downstream_task_knowledge",
                    name="K2 downstream task knowledge",
                    output_key="downstream_task_knowledge",
                    asset_path=self.config.resolved_k2_expertise_path,
                    query_field="question",
                    answer_field="answer",
                    query_template="How do geologists conduct the task of {query}?",
                    backend_factory=semantic_backend_factory,
                    default_top_k=self.config.semantic_top_k,
                    min_score=self.config.semantic_min_score,
                    batch_size=self.config.semantic_batch_size,
                    model_name=self.config.semantic_model_name,
                    model_revision=self.config.semantic_model_revision,
                    device=self.config.semantic_device,
                    local_files_only=self.config.semantic_local_files_only,
                ),
                supports=lambda request: bool(request.query_text),
                default_enabled=False,
            ),
        ]

    def _earthquake_provider_factory(self) -> EarthquakeHistoryProvider:
        bindings = [
            binding
            for source_id in self.config.earthquake_source_ids
            if (binding := self._earthquake_source_binding(source_id)) is not None
        ]
        if not bindings:
            bindings = [self._legacy_earthquake_binding(self.config.earthquake_source_id)]
        primary = bindings[0]
        return EarthquakeHistoryProvider(
            primary.asset_path or self.config.resolved_earthquake_csv_path,
            default_max_records=self.config.max_records_per_provider,
            engine=self.config.earthquake_engine,
            source_id=primary.source_id,
            source_mode=primary.source_mode,
            source_manifest_path=primary.source_manifest_path,
            source_manifest=primary.source_manifest,
            fallback_warning=primary.fallback_warning,
            source_bindings=bindings,
        )

    def _active_fault_provider_factory(self) -> ActiveFaultProvider:
        bindings = [
            binding
            for source_id in self.config.active_fault_source_ids
            if (binding := self._active_fault_source_binding(source_id)) is not None
        ]
        if not bindings:
            bindings = [self._legacy_active_fault_binding(self.config.active_fault_source_id)]
        primary = bindings[0]
        return ActiveFaultProvider(
            primary.asset_path or self.config.resolved_active_fault_geojson_path,
            default_max_records=self.config.max_records_per_provider,
            geometry_engine=self.config.fault_geometry_engine,
            source_id=primary.source_id,
            source_mode=primary.source_mode,
            source_manifest_path=primary.source_manifest_path,
            source_manifest=primary.source_manifest,
            fallback_warning=primary.fallback_warning,
            source_bindings=bindings,
        )

    def _earthquake_source_binding(self, source_id: str) -> EarthquakeSourceBinding | None:
        from .sources.usgs_events import (
            EMSC_DEFAULT_PROFILE,
            EMSC_EVENT_BASE_URL,
            FdsnEventSourceAdapter,
            UsgsFdsnEventAdapter,
        )

        manifest_path = self._latest_manifest_path(source_id)
        if manifest_path is not None:
            manifest = SourceManifest.from_path(manifest_path)
            artifact_path = manifest.normalized_artifact_path(manifest_path)
            if artifact_path.exists():
                return EarthquakeSourceBinding(
                    source_id=source_id,
                    source_mode="local_mirror",
                    asset_path=artifact_path,
                    source_manifest_path=manifest_path,
                    source_manifest=manifest,
                    supports_live=True,
                    adapter=UsgsFdsnEventAdapter()
                    if source_id == "usgs_fdsn_events"
                    else FdsnEventSourceAdapter(
                        source_id=source_id,
                        base_url=EMSC_EVENT_BASE_URL,
                        default_profile=EMSC_DEFAULT_PROFILE,
                    ),
                )
        if source_id == "usgs_fdsn_events":
            return self._legacy_earthquake_binding(source_id)
        if source_id == "emsc_fdsn_events":
            return EarthquakeSourceBinding(
                source_id=source_id,
                source_mode="live",
                adapter=FdsnEventSourceAdapter(
                    source_id=source_id,
                    base_url=EMSC_EVENT_BASE_URL,
                    default_profile=EMSC_DEFAULT_PROFILE,
                ),
                supports_live=True,
            )
        return None

    def _legacy_earthquake_binding(self, source_id: str) -> EarthquakeSourceBinding:
        return EarthquakeSourceBinding(
            source_id=source_id,
            source_mode="legacy_asset",
            asset_path=self.config.resolved_earthquake_csv_path,
            supports_live=source_id == "usgs_fdsn_events",
            fallback_warning=(
                "earthquake_history: using legacy local asset because no source mirror was found."
            ),
        )

    def _active_fault_source_binding(self, source_id: str) -> FaultSourceBinding | None:
        from .sources.diss_faults import DissSeismogenicSourceAdapter
        from .sources.registry import default_source_registry

        preferred_version = (
            self.config.gem_active_fault_version if source_id == "gem_global_active_faults" else None
        )
        manifest_path = self._latest_manifest_path(source_id, preferred_version=preferred_version)
        if manifest_path is not None:
            manifest = SourceManifest.from_path(manifest_path)
            artifact_path = manifest.normalized_artifact_path(manifest_path)
            if artifact_path.exists():
                return FaultSourceBinding(
                    source_id=source_id,
                    source_mode="local_mirror",
                    asset_path=artifact_path,
                    source_manifest_path=manifest_path,
                    source_manifest=manifest,
                )
        if source_id == "gem_global_active_faults":
            return self._legacy_active_fault_binding(source_id)
        if source_id == "diss_seismogenic_sources":
            definition = default_source_registry().get(source_id)
            profile = definition.validate_profile(None)
            coverage_bounds = self._bounds_from_tuple(definition.coverage_bounds)
            return FaultSourceBinding(
                source_id=source_id,
                source_mode="live",
                adapter=DissSeismogenicSourceAdapter(endpoint=profile["endpoint"]),
                supports_live=True,
                coverage_bounds=coverage_bounds,
                region_name="DISS Italy region",
            )
        return None

    def _legacy_active_fault_binding(self, source_id: str) -> FaultSourceBinding:
        return FaultSourceBinding(
            source_id=source_id,
            source_mode="legacy_asset",
            asset_path=self.config.resolved_active_fault_geojson_path,
            fallback_warning=(
                "active_faults: using legacy local asset because no source mirror was found."
            ),
        )

    def _mineral_occurrence_provider_factory(self) -> Any:
        # Imported here (not at module load) to keep the import graph for the
        # default service light; the modules are stdlib-only but this matches the
        # explicit-only nature of the live, network-backed mineral provider.
        from .providers.minerals import MineralOccurrenceProvider, MineralSourceBinding
        from .sources.ogs_minerals import OgsMineralOccurrenceAdapter
        from .sources.ogs_minerals import normalize_features as normalize_ogs_features
        from .sources.registry import default_source_registry
        from .sources.sigeom_minerals import (
            SigeomMineralOccurrenceAdapter,
            normalize_sigeom_features,
        )

        registry = default_source_registry()
        bindings: list[MineralSourceBinding] = []
        for source_id in self.config.mineral_occurrence_source_ids:
            definition = registry.get(source_id)
            profile = definition.validate_profile(None)
            coverage_bounds = self._bounds_from_tuple(definition.coverage_bounds)
            if source_id == "ontario_mineral_deposit_inventory":
                bindings.append(
                    MineralSourceBinding(
                        source_id=source_id,
                        adapter=OgsMineralOccurrenceAdapter(endpoint=profile["endpoint"]),
                        normalize=normalize_ogs_features,
                        coverage_bounds=coverage_bounds,
                        region_name=str(profile.get("region", "Ontario")),
                    )
                )
            elif source_id == "sigeom_mineral_occurrences":
                bindings.append(
                    MineralSourceBinding(
                        source_id=source_id,
                        adapter=SigeomMineralOccurrenceAdapter(endpoint=profile["endpoint"]),
                        normalize=normalize_sigeom_features,
                        coverage_bounds=coverage_bounds,
                        region_name=str(profile.get("region", "Quebec")),
                    )
                )
        if not bindings:
            definition = registry.get(self.config.mineral_occurrence_source_id)
            profile = definition.validate_profile(None)
            bindings = [
                MineralSourceBinding(
                    source_id=definition.id,
                    adapter=OgsMineralOccurrenceAdapter(endpoint=profile["endpoint"]),
                    normalize=normalize_ogs_features,
                    coverage_bounds=self._bounds_from_tuple(definition.coverage_bounds),
                    region_name=str(profile.get("region", "Ontario")),
                )
            ]
        primary = bindings[0]
        return MineralOccurrenceProvider(
            adapter=primary.adapter,
            source_id=primary.source_id,
            coverage_bounds=primary.coverage_bounds,
            region_name=primary.region_name,
            default_max_records=self.config.max_records_per_provider,
            source_bindings=bindings,
        )

    def _bounds_from_tuple(
        self,
        raw_bounds: tuple[float, float, float, float] | None,
    ) -> Bounds | None:
        if raw_bounds is None:
            return None
        min_lon, min_lat, max_lon, max_lat = raw_bounds
        return Bounds(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)

    def _latest_manifest_path(
        self,
        source_id: str,
        preferred_version: str | None = None,
    ) -> Path | None:
        if self.config.knowledge_sources_root is None:
            return None
        return find_latest_manifest(
            self.config.knowledge_sources_root,
            source_id,
            preferred_version=preferred_version,
        )

    def refresh_providers(self) -> None:
        self._provider_instances.clear()

    def _semantic_backend_factory(self) -> Callable[[], SentenceTransformerSemanticBackend]:
        backend: SentenceTransformerSemanticBackend | None = None

        def factory() -> SentenceTransformerSemanticBackend:
            nonlocal backend
            if backend is None:
                backend = SentenceTransformerSemanticBackend(
                    self.config.semantic_model_name,
                    model_revision=self.config.semantic_model_revision,
                    device=self.config.semantic_device,
                    local_files_only=self.config.semantic_local_files_only,
                )
            return backend

        return factory

    def _registration_for_provider(self, provider: Any) -> ProviderRegistration:
        output_keys = tuple(getattr(provider, "output_keys", (provider.id,)))
        return ProviderRegistration(
            id=provider.id,
            name=getattr(provider, "name", provider.id),
            output_keys=output_keys,
            factory=lambda provider=provider: provider,
            supports=provider.supports,
            default_enabled=True,
        )

    def _select_registrations(
        self,
        request: KnowledgeRequest,
        warnings: list[str],
    ) -> tuple[list[ProviderRegistration], set[str]]:
        explicit_ids: set[str] = set()
        if request.include:
            selected: list[ProviderRegistration] = []
            for include in request.include:
                matches = self._resolve_filter(include, warnings)
                if not matches:
                    continue
                for registration in matches:
                    explicit_ids.add(registration.id)
                    if not registration.supports(request):
                        warnings.append(
                            f"Requested provider {registration.id!r} does not support this request."
                        )
                        continue
                    selected.append(registration)
        else:
            selected = [
                registration
                for registration in self._registrations
                if registration.default_enabled and registration.supports(request)
            ]

        exclude_ids: set[str] = set()
        for exclude in request.exclude:
            for registration in self._resolve_filter(exclude, warnings):
                exclude_ids.add(registration.id)

        selected = [registration for registration in selected if registration.id not in exclude_ids]
        selected = self._dedupe_registrations(selected)
        if request.include and not selected:
            raise ProviderError(
                "No available knowledge providers matched the include filters for this request."
            )
        return selected, explicit_ids

    def _resolve_filter(
        self,
        token: str,
        warnings: list[str],
    ) -> list[ProviderRegistration]:
        normalized = self._normalize_alias(token)
        matches = [
            registration
            for registration in self._registrations
            if normalized in self._registration_aliases(registration)
        ]
        if not matches:
            warnings.append(f"No knowledge provider matched filter {token!r}.")
            return []
        if len(matches) > 1:
            warnings.append(f"Knowledge provider filter {token!r} is ambiguous and was ignored.")
            return []
        return matches

    def _registration_aliases(self, registration: ProviderRegistration) -> set[str]:
        aliases = {registration.id, registration.name, *registration.output_keys}
        return {self._normalize_alias(alias) for alias in aliases}

    def _normalize_alias(self, value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")

    def _dedupe_registrations(
        self,
        registrations: Iterable[ProviderRegistration],
    ) -> list[ProviderRegistration]:
        seen: set[str] = set()
        deduped: list[ProviderRegistration] = []
        for registration in registrations:
            if registration.id in seen:
                continue
            seen.add(registration.id)
            deduped.append(registration)
        return deduped

    def _request_with_validated_provider_options(
        self,
        request: KnowledgeRequest,
        registrations: list[ProviderRegistration],
        warnings: list[str],
    ) -> KnowledgeRequest:
        if not request.provider_options:
            return request
        selected_ids = {registration.id for registration in registrations}
        validated_options: dict[str, dict[str, Any]] = {}
        for raw_key, raw_options in request.provider_options.items():
            provider_id = self._provider_id_for_option_key(raw_key)
            if provider_id is None:
                message = f"No knowledge provider matched provider_options key {raw_key!r}."
                if request.include:
                    raise ProviderOptionError(message)
                warnings.append(message)
                continue
            if provider_id not in selected_ids:
                warnings.append(
                    f"provider_options for {provider_id!r} were ignored because the provider "
                    "was not selected."
                )
                continue
            registration = next(item for item in registrations if item.id == provider_id)
            provider = self._provider_for_registration(registration)
            validate_options = getattr(provider, "validate_options", None)
            if callable(validate_options):
                validated_options[provider_id] = dict(validate_options(raw_options))
            else:
                if raw_options:
                    raise ProviderOptionError(
                        f"Provider {provider_id!r} does not accept provider_options."
                    )
                validated_options[provider_id] = {}
        return KnowledgeRequest(
            bounds=request.bounds,
            legend_labels=list(request.legend_labels),
            query_text=request.query_text,
            include=request.include,
            exclude=request.exclude,
            max_records=request.max_records,
            max_records_by_provider=dict(request.max_records_by_provider),
            provider_options=validated_options,
            trace_id=request.trace_id,
        )

    def _provider_id_for_option_key(self, token: str) -> str | None:
        normalized = self._normalize_alias(token)
        matches = [
            registration.id
            for registration in self._registrations
            if normalized in self._registration_aliases(registration)
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    def _provider_for_registration(self, registration: ProviderRegistration) -> Any:
        if registration.id not in self._provider_instances:
            self._provider_instances[registration.id] = registration.factory()
        return self._provider_instances[registration.id]

    def _query_provider(
        self,
        provider: Any,
        request: KnowledgeRequest,
        *,
        bounds_parts: list[Bounds],
    ) -> tuple[list[KnowledgeItem], str, bool]:
        provider_version = self._provider_version(provider, request)
        if not self.config.write_cache:
            return self._query_provider_uncached(provider, request, bounds_parts), provider_version, False
        cache_key = self._provider_cache_key(provider, provider_version, request, bounds_parts)
        cached = self.cache.read_provider_items(provider.id, cache_key, provider_version)
        if cached is not None:
            self._prepare_cached_provider_warnings(provider, request, bounds_parts, cached)
            return cached, provider_version, True
        items = self._query_provider_uncached(provider, request, bounds_parts)
        self.cache.write_provider_items(provider.id, cache_key, provider_version, items)
        return items, provider_version, False

    def _prepare_cached_provider_warnings(
        self,
        provider: Any,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
        cached_items: list[KnowledgeItem],
    ) -> None:
        warnings_for_cached_result = getattr(provider, "warnings_for_cached_result", None)
        if callable(warnings_for_cached_result):
            provider.last_warnings = list(
                warnings_for_cached_result(request, bounds_parts, cached_items)
            )
        elif hasattr(provider, "last_warnings"):
            provider.last_warnings = []

    def _query_provider_uncached(
        self,
        provider: Any,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
    ) -> list[KnowledgeItem]:
        if len(bounds_parts) > 1:
            query_bounds_parts = getattr(provider, "query_bounds_parts", None)
            if callable(query_bounds_parts):
                return list(query_bounds_parts(request, bounds_parts))
            raise ProviderError(
                f"Provider {provider.id!r} does not support antimeridian-split extent queries."
            )
        if bounds_parts:
            query_bounds_parts = getattr(provider, "query_bounds_parts", None)
            if callable(query_bounds_parts):
                return list(query_bounds_parts(request, bounds_parts))
        return list(provider.query(request))

    def _provider_cache_key(
        self,
        provider: Any,
        provider_version: str,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
    ) -> str:
        query_hash = stable_hash(request.query_text) if request.query_text is not None else None
        return stable_hash(
            {
                "provider": provider.id,
                "provider_version": provider_version,
                "provider_config": self._provider_cache_config(provider),
                "source_asset_path": str(getattr(provider, "asset_path", "")) or None,
                "bounds_parts": [
                    part.to_cache_dict(self.config.bounds_cache_precision) for part in bounds_parts
                ],
                "legend_labels": list(request.legend_labels),
                "query_text_hash": query_hash,
                "max_records": request.max_records,
                "max_records_by_provider": dict(request.max_records_by_provider),
                "provider_options": {
                    provider_id: dict(options)
                    for provider_id, options in request.provider_options.items()
                },
                "default_max_records": self.config.max_records_per_provider,
            }
        )

    def _provider_cache_config(self, provider: Any) -> dict[str, Any] | None:
        cache_config_method = getattr(provider, "cache_config", None)
        if cache_config_method is None:
            return None
        return dict(cache_config_method())

    def _provider_version(self, provider: Any, request: KnowledgeRequest | None = None) -> str:
        source_version_for_options = getattr(provider, "source_version_for_options", None)
        if callable(source_version_for_options) and request is not None:
            options = request.provider_options.get(getattr(provider, "id", ""), {})
            return str(source_version_for_options(options))
        source_version_method = getattr(provider, "source_version", None)
        if source_version_method is not None:
            return str(source_version_method())
        return str(getattr(provider, "version", "unknown"))

    def _legend_value(self, items: list[KnowledgeItem]) -> str | None:
        if not items:
            return None
        value = items[0].value
        if not isinstance(value, dict):
            return None
        result = value.get("value")
        if result in (None, "unknown"):
            return None
        return str(result)

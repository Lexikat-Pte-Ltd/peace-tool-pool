"""Local knowledge service facade."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .bounds import Bounds
from .cache import KnowledgeCache, stable_hash
from .config import KnowledgeConfig
from .errors import MissingAssetError, OptionalDependencyError, ProviderError
from .providers.earthengine import (
    EarthEngineLandcoverProvider,
    EarthEnginePopulationDensityProvider,
)
from .providers.earthquakes import EarthquakeHistoryProvider
from .providers.faults import ActiveFaultProvider
from .providers.rock import RockLookupProvider
from .providers.semantic_k2 import SemanticK2Provider, SentenceTransformerSemanticBackend
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
        warnings: list[str] = []
        registrations, explicit_ids = self._select_registrations(request, warnings)
        items: list[KnowledgeItem] = []
        provider_versions: dict[str, str] = {}
        trace_events: list[dict[str, Any]] = []
        explicit_success_ids: set[str] = set()
        explicit_failures: list[tuple[str, BaseException]] = []

        for registration in registrations:
            explicit = registration.id in explicit_ids
            try:
                provider = self._provider_for_registration(registration)
                provider_items = self._query_provider(provider, request)
                items.extend(provider_items)
                if explicit:
                    explicit_success_ids.add(registration.id)
                provider_versions[provider.id] = self._provider_version(provider)
                trace_events.append(
                    {
                        "provider": provider.id,
                        "record_count": sum(item.record_count or 0 for item in provider_items),
                        "truncated": any(item.truncated for item in provider_items),
                    }
                )
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

        trace = {"trace_id": request.trace_id, "providers": trace_events}
        return KnowledgeBundle(
            bounds=request.bounds,
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
    ) -> KnowledgeBundle:
        return self.query(
            KnowledgeRequest(
                bounds=bounds,
                include=include,
                exclude=exclude,
                max_records=max_records,
            )
        )

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
    ) -> KnowledgeBundle:
        raise NotImplementedError(
            "KnowledgeService.query_map requires tested decimal bounds extraction from map "
            "metadata; pass explicit Bounds to query_bounds in this phase."
        )

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
                factory=lambda: EarthquakeHistoryProvider(
                    self.config.resolved_earthquake_csv_path,
                    default_max_records=self.config.max_records_per_provider,
                    engine=self.config.earthquake_engine,
                ),
                supports=lambda request: request.bounds is not None,
            ),
            ProviderRegistration(
                id="active_faults",
                name="Active faults",
                output_keys=("active_faults",),
                factory=lambda: ActiveFaultProvider(
                    self.config.resolved_active_fault_geojson_path,
                    default_max_records=self.config.max_records_per_provider,
                    geometry_engine=self.config.fault_geometry_engine,
                ),
                supports=lambda request: request.bounds is not None,
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

    def _provider_for_registration(self, registration: ProviderRegistration) -> Any:
        if registration.id not in self._provider_instances:
            self._provider_instances[registration.id] = registration.factory()
        return self._provider_instances[registration.id]

    def _query_provider(self, provider: Any, request: KnowledgeRequest) -> list[KnowledgeItem]:
        if not self.config.write_cache:
            return provider.query(request)
        provider_version = self._provider_version(provider)
        cache_key = self._provider_cache_key(provider, provider_version, request)
        cached = self.cache.read_provider_items(provider.id, cache_key, provider_version)
        if cached is not None:
            return cached
        items = provider.query(request)
        self.cache.write_provider_items(provider.id, cache_key, provider_version, items)
        return items

    def _provider_cache_key(
        self,
        provider: Any,
        provider_version: str,
        request: KnowledgeRequest,
    ) -> str:
        query_hash = stable_hash(request.query_text) if request.query_text is not None else None
        return stable_hash(
            {
                "provider": provider.id,
                "provider_version": provider_version,
                "provider_config": self._provider_cache_config(provider),
                "source_asset_path": str(getattr(provider, "asset_path", "")) or None,
                "bounds": request.bounds.to_cache_dict(self.config.bounds_cache_precision)
                if request.bounds is not None
                else None,
                "legend_labels": list(request.legend_labels),
                "query_text_hash": query_hash,
                "max_records": request.max_records,
                "max_records_by_provider": dict(request.max_records_by_provider),
                "default_max_records": self.config.max_records_per_provider,
            }
        )

    def _provider_cache_config(self, provider: Any) -> dict[str, Any] | None:
        cache_config_method = getattr(provider, "cache_config", None)
        if cache_config_method is None:
            return None
        return dict(cache_config_method())

    def _provider_version(self, provider: Any) -> str:
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

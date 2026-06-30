"""SDK-backed implementation of the geomap MCP tool surface."""

from __future__ import annotations

import functools
import importlib.util
import json
from contextvars import ContextVar
from typing import Any, Callable, Mapping

from ..knowledge import Bounds, KnowledgeBundle, KnowledgeItem, KnowledgeRequest
from ..knowledge import KnowledgeService
from ..knowledge.visualization import extract_knowledge_overlay, render_knowledge_overlay_svg
from .errors import McpToolError
from .images import make_inline_preview
from .resources import ResourceRegistry
from .schemas import (
    knowledge_bundle_to_mcp,
    legend_enrichment_to_mcp,
    map_processing_result_to_mcp,
    new_trace_id,
    redact_paths,
    serialize_georef,
    success_result,
)


KnowledgeServiceFactory = Callable[[], KnowledgeService]
MapServiceFactory = Callable[[], Any]

# One trace id per agent-facing tool call. Nested adapter calls (e.g. query_map
# delegating to query_knowledge) reuse the outer id so a single call has a single
# trace, and any McpToolError raised mid-call -- including trace-agnostic registry
# errors -- is stamped with it before it leaves the adapter.
_ACTIVE_TRACE: ContextVar[str | None] = ContextVar("geomap_active_trace_id", default=None)


def _active_trace_id() -> str:
    current = _ACTIVE_TRACE.get()
    return current if current is not None else new_trace_id()


def _traced(method: Callable[..., Any]) -> Callable[..., Any]:
    """Stamp the call's trace id onto any McpToolError that lacks one."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        existing = _ACTIVE_TRACE.get()
        trace_id = existing if existing is not None else new_trace_id()
        token = _ACTIVE_TRACE.set(trace_id)
        try:
            return method(self, *args, **kwargs)
        except McpToolError as exc:
            if exc.trace_id is None:
                exc.trace_id = trace_id
            raise
        finally:
            _ACTIVE_TRACE.reset(token)

    return wrapper


class GeomapMcpAdapter:
    """Thin protocol-independent adapter over existing local SDK services."""

    def __init__(
        self,
        *,
        registry: ResourceRegistry | None = None,
        knowledge_service_factory: KnowledgeServiceFactory | None = None,
        map_service_factory: MapServiceFactory | None = None,
    ):
        self.registry = registry or ResourceRegistry.from_env()
        self._knowledge_service_factory = knowledge_service_factory or KnowledgeService.from_env
        self._map_service_factory = map_service_factory or self._default_map_service
        self._knowledge_service: KnowledgeService | None = None
        self._map_service: Any | None = None

    @_traced
    def list_capabilities(self) -> dict[str, Any]:
        trace_id = _active_trace_id()
        service = self._knowledge()
        providers = [
            {
                "id": registration.id,
                "name": registration.name,
                "output_keys": list(registration.output_keys),
                "default_enabled": bool(registration.default_enabled),
            }
            for registration in getattr(service, "_registrations", [])
        ]
        config = self._map_config()
        structured = {
            "schema_versions": {
                "map_processing": "map-processing/v1",
                "knowledge": "knowledge/v2",
                "georef": "georef/v1",
                "mcp_registry": "mcp-registry/v1",
            },
            "installed": {
                "mcp": _module_available("mcp"),
                "pillow": _module_available("PIL"),
                "cv2": _module_available("cv2"),
                "numpy": _module_available("numpy"),
                "pyproj": _module_available("pyproj"),
                "httpx": _module_available("httpx"),
                "earthengine": _module_available("ee"),
                "sentence_transformers": _module_available("sentence_transformers"),
                "torch": _module_available("torch"),
            },
            "detectors": {
                "component_model_present": config.resolved_component_model_path.exists(),
                "legend_model_present": config.resolved_legend_model_path.exists(),
                "runtime_available": _module_available("cv2") and _module_available("numpy"),
            },
            "providers": providers,
            "allowed_roots": self.registry.allowed_root_labels(),
            "limits": self.registry.limits,
        }
        return success_result(
            structured=structured,
            text_summary=f"geomap MCP exposes {len(providers)} knowledge providers.",
            trace_id=trace_id,
        )

    @_traced
    def register_map(self, path: str) -> dict[str, Any]:
        trace_id = _active_trace_id()
        path_string = str(path)
        if path_string.startswith("geomap://maps/"):
            structured = self.registry.map_public(self.registry.map_id_from_uri(path_string))
        else:
            structured = self.registry.register_map(path_string)
        return success_result(
            structured=structured,
            text_summary=f"Registered map {structured['map_id']}.",
            trace_id=trace_id,
            resource_links=[
                {
                    "uri": structured["source_uri"],
                    "name": "source map image",
                    "mimeType": structured["mime_type"],
                }
            ],
        )

    @_traced
    def process_image(self, *, map_id: str | None = None, map_uri: str | None = None) -> dict[str, Any]:
        trace_id = _active_trace_id()
        resolved_map_id = self._resolve_map_id(map_id=map_id, map_uri=map_uri)
        image_path = self.registry.source_path(resolved_map_id)
        try:
            result = self._map().process_image(image_path)
        except Exception as exc:  # noqa: BLE001 - optional detector failures need typed MCP errors.
            if exc.__class__.__name__ in {"OptionalDependencyError", "DetectorLoadError"}:
                raise McpToolError("missing_extra", str(exc), trace_id=trace_id) from exc
            raise
        # One map carries many artifacts; coalesce their registrations into a
        # single locked merge-write instead of one per artifact.
        with self.registry.deferred_save():
            structured = map_processing_result_to_mcp(
                result,
                registry=self.registry,
                map_id=resolved_map_id,
            )
            content: list[dict[str, Any]] = []
            preview = self._preview_for_role(structured.get("artifacts", []), "detection_overlay")
            if preview is not None:
                structured["preview"] = preview["metadata"]
                content.append(preview["content"])
            self.registry.set_map_processing(resolved_map_id, structured)
        resource_links = [
            {"uri": artifact["uri"], "name": artifact.get("role") or "artifact", "mimeType": artifact.get("mime_type")}
            for artifact in structured.get("artifacts", [])
        ]
        return success_result(
            structured=structured,
            text_summary=(
                f"Processed map {resolved_map_id}: "
                f"{sum(len(items) for items in structured['regions'].values())} regions, "
                f"{len(structured['legend'])} legend entries."
            ),
            trace_id=trace_id,
            content=content,
            resource_links=resource_links,
        )

    @_traced
    def georeference(
        self,
        *,
        crs: str | int,
        gcps: list[Mapping[str, Any] | list[float] | tuple[float, float, float, float]],
        pixel_extent: list[float] | tuple[float, float, float, float] | None = None,
        map_id: str | None = None,
        map_uri: str | None = None,
        main_map_artifact_uri: str | None = None,
    ) -> dict[str, Any]:
        trace_id = _active_trace_id()
        resolved_map_id = self._resolve_map_id(map_id=map_id, map_uri=map_uri, required=False)
        normalized_gcps = [_gcp_dict(gcp) for gcp in gcps]
        if pixel_extent is None and main_map_artifact_uri:
            entry = self.registry.artifact_entry(main_map_artifact_uri)
            bbox = entry.get("bbox")
            if bbox:
                pixel_extent = [float(value) for value in bbox]
        if pixel_extent is None and resolved_map_id:
            pixel_extent = self._main_map_pixel_extent(resolved_map_id)
        if pixel_extent is None:
            raise McpToolError("invalid_bounds", "pixel_extent or main map artifact bbox is required.")
        try:
            from ..georef import GroundControlPoint, georeference_bounds
        except Exception as exc:  # noqa: BLE001 - missing geo extra surfaces as typed error.
            raise McpToolError("missing_extra", "Install the 'geo' extra for georeferencing.") from exc
        try:
            ref = georeference_bounds(
                crs=crs,
                gcps=[GroundControlPoint(**gcp) for gcp in normalized_gcps],
                pixel_extent=tuple(float(value) for value in pixel_extent),
            )
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "GeoReferenceError" and "install" in str(exc).lower():
                raise McpToolError("missing_extra", str(exc), trace_id=trace_id) from exc
            raise McpToolError("invalid_bounds", str(exc), trace_id=trace_id) from exc
        structured = serialize_georef(
            ref,
            pixel_extent=list(pixel_extent),
            gcps=normalized_gcps,
            trace_id=trace_id,
        )
        resource_links: list[dict[str, Any]] = []
        if resolved_map_id:
            resource = self.registry.set_map_georef(resolved_map_id, structured)
            structured["georef_uri"] = resource["uri"]
            resource_links.append(
                {"uri": resource["uri"], "name": "georef JSON", "mimeType": "application/json"}
            )
        return success_result(
            structured=structured,
            text_summary=(
                f"Georeferenced map bounds: lon "
                f"[{structured['bounds']['min_lon']:.4f}, {structured['bounds']['max_lon']:.4f}], "
                f"lat [{structured['bounds']['min_lat']:.4f}, {structured['bounds']['max_lat']:.4f}]."
            ),
            trace_id=trace_id,
            resource_links=resource_links,
        )

    @_traced
    def query_knowledge(
        self,
        *,
        bounds: Mapping[str, Any] | Bounds | None = None,
        legend_labels: list[str] | tuple[str, ...] | None = None,
        query_text: str | None = None,
        include: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        max_records: int | None = None,
        max_records_by_provider: Mapping[str, int] | None = None,
        provider_options: Mapping[str, Mapping[str, Any]] | None = None,
        map_id: str | None = None,
    ) -> dict[str, Any]:
        trace_id = _active_trace_id()
        try:
            request = KnowledgeRequest(
                bounds=_bounds_from_any(bounds),
                legend_labels=list(legend_labels or []),
                query_text=query_text,
                include=tuple(include or ()),
                exclude=tuple(exclude or ()),
                max_records=max_records,
                max_records_by_provider=dict(max_records_by_provider or {}),
                provider_options={key: dict(value) for key, value in (provider_options or {}).items()},
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "InvalidBoundsError":
                raise McpToolError("invalid_bounds", str(exc), trace_id=trace_id) from exc
            raise
        try:
            bundle = self._knowledge().query(request)
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ in {"ProviderError", "ProviderOptionError"}:
                raise McpToolError("unknown_provider", str(exc), trace_id=trace_id) from exc
            raise
        return self._bundle_result(bundle, trace_id=trace_id, map_id=map_id)

    @_traced
    def query_map(
        self,
        *,
        map_id: str | None = None,
        map_uri: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        question: str | None = None,
        include: list[str] | tuple[str, ...] | None = None,
        exclude: list[str] | tuple[str, ...] | None = None,
        max_records: int | None = None,
        max_records_by_provider: Mapping[str, int] | None = None,
        provider_options: Mapping[str, Mapping[str, Any]] | None = None,
        bounds: Mapping[str, Any] | Bounds | None = None,
        legend_labels: list[str] | tuple[str, ...] | None = None,
        query_text: str | None = None,
    ) -> dict[str, Any]:
        resolved_map_id = self._resolve_map_id(map_id=map_id, map_uri=map_uri, required=False)
        metadata = dict(metadata or {})
        if bounds is None:
            bounds = metadata.get("bounds")
        labels = list(legend_labels or []) or _legend_labels_from_metadata(metadata)
        if resolved_map_id:
            processing = self.registry.get_map_processing(resolved_map_id) or {}
            labels = labels or _legend_labels_from_metadata(processing)
            if bounds is None:
                georef = self.registry.get_map_georef(resolved_map_id)
                if georef is not None:
                    bounds = georef.get("bounds")
        if bounds is None and not labels:
            raise McpToolError(
                "georef_required",
                "query_map needs stored georef bounds, explicit bounds, or legend labels.",
            )
        return self.query_knowledge(
            bounds=bounds,
            legend_labels=labels,
            query_text=query_text or question,
            include=include,
            exclude=exclude,
            max_records=max_records,
            max_records_by_provider=max_records_by_provider,
            provider_options=provider_options,
            map_id=resolved_map_id,
        )

    @_traced
    def enrich_legend(self, label: str) -> dict[str, Any]:
        trace_id = _active_trace_id()
        enrichment = self._knowledge().enrich_legend_label(label)
        structured = legend_enrichment_to_mcp(enrichment)
        return success_result(
            structured=structured,
            text_summary=f"Enriched legend label {label!r}.",
            trace_id=trace_id,
        )

    @_traced
    def render_knowledge_overlay(
        self,
        *,
        map_id: str | None = None,
        map_uri: str | None = None,
        bundle_uri: str | None = None,
        bundle: Mapping[str, Any] | None = None,
        georef: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace_id = _active_trace_id()
        resolved_map_id = self._resolve_map_id(map_id=map_id, map_uri=map_uri, required=False)
        bundle_data = self._bundle_data(bundle_uri=bundle_uri, bundle=bundle)
        georef_data = dict(georef or {})
        if not georef_data and resolved_map_id:
            georef_data = self.registry.get_map_georef(resolved_map_id) or {}
        if not georef_data:
            raise McpToolError(
                "georef_required",
                "A stored or inline georef is required to render a map-backed overlay.",
                trace_id=trace_id,
            )
        knowledge_bundle = _bundle_from_dict(bundle_data)
        metadata: dict[str, Any] = {"georef": georef_data}
        if resolved_map_id:
            metadata["image_path"] = str(self.registry.source_path(resolved_map_id))
        overlay = extract_knowledge_overlay(knowledge_bundle, metadata=metadata)
        output_dir = self.registry.cache_root / "mcp" / "v1" / "overlays"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = new_trace_id()
        svg_path = output_dir / f"{stem}.svg"
        render_knowledge_overlay_svg(overlay, svg_path)
        resources = [self.registry.register_overlay(svg_path, map_id=resolved_map_id)]
        warnings = list(overlay.warnings)
        png_preview = None
        try:
            from ..knowledge.visualization import render_knowledge_overlay_on_image

            georef_object = _georef_from_dict(georef_data)
            if resolved_map_id:
                png_path = output_dir / f"{stem}.png"
                render_knowledge_overlay_on_image(
                    overlay,
                    georef_object,
                    self.registry.source_path(resolved_map_id),
                    png_path,
                )
                png_resource = self.registry.register_overlay(png_path, map_id=resolved_map_id)
                resources.append(png_resource)
                png_preview = make_inline_preview(png_path, artifact_uri=png_resource["uri"])
        except Exception as exc:  # noqa: BLE001 - SVG remains useful when raster deps are absent.
            warnings.append(f"annotated PNG unavailable: {type(exc).__name__}: {exc}")
        structured = {
            "overlay": redact_paths(overlay.to_dict()),
            "resources": resources,
            "warnings": warnings,
        }
        content = [png_preview["content"]] if png_preview else None
        if png_preview:
            structured["preview"] = png_preview["metadata"]
        return success_result(
            structured=structured,
            text_summary=f"Rendered knowledge overlay with {len(overlay.items)} annotation item(s).",
            trace_id=trace_id,
            content=content,
            resource_links=[
                {"uri": item["uri"], "name": "knowledge overlay", "mimeType": item["mime_type"]}
                for item in resources
            ],
        )

    def read_resource(self, uri: str) -> dict[str, Any]:
        return self.registry.read_resource(uri)

    def _bundle_result(
        self,
        bundle: Any,
        *,
        trace_id: str,
        map_id: str | None = None,
    ) -> dict[str, Any]:
        structured = knowledge_bundle_to_mcp(bundle)
        bundle_resource = self.registry.register_bundle(structured, map_id=map_id)
        structured["bundle_uri"] = bundle_resource["uri"]
        provider_counts: dict[str, int] = {}
        for item in structured.get("items", []):
            provider = item.get("provider", "unknown")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        summary = (
            f"Knowledge query returned {len(structured.get('items', []))} item(s) "
            f"from {len(provider_counts)} provider(s)."
        )
        if structured.get("warnings"):
            summary += f" {len(structured['warnings'])} warning(s)."
        return success_result(
            structured=structured,
            text_summary=summary,
            trace_id=trace_id,
            resource_links=[
                {"uri": bundle_resource["uri"], "name": "knowledge bundle", "mimeType": "application/json"}
            ],
        )

    def _preview_for_role(self, artifacts: list[Mapping[str, Any]], role: str) -> dict[str, Any] | None:
        for artifact in artifacts:
            if artifact.get("role") != role:
                continue
            try:
                entry = self.registry.artifact_entry(str(artifact["uri"]))
            except McpToolError:
                return None
            return make_inline_preview(entry["path"], artifact_uri=str(artifact["uri"]))
        return None

    def _resolve_map_id(
        self,
        *,
        map_id: str | None,
        map_uri: str | None,
        required: bool = True,
    ) -> str | None:
        if map_id:
            self.registry.map_public(map_id)
            return map_id
        if map_uri:
            return self.registry.map_id_from_uri(map_uri)
        if required:
            raise McpToolError("artifact_not_found", "map_id or map_uri is required.")
        return None

    def _main_map_pixel_extent(self, map_id: str) -> list[float] | None:
        processing = self.registry.get_map_processing(map_id) or {}
        for detection in processing.get("regions", {}).get("main_map", []):
            bbox = detection.get("bbox")
            if bbox:
                return [float(value) for value in bbox]
        return None

    def _bundle_data(
        self,
        *,
        bundle_uri: str | None,
        bundle: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if bundle is not None:
            return redact_paths(dict(bundle))
        if bundle_uri is None:
            raise McpToolError("artifact_not_found", "bundle_uri or inline bundle is required.")
        content = self.registry.read_resource(bundle_uri)
        if "text" not in content:
            raise McpToolError("unsupported_media", "Bundle resource must be JSON text.")
        return json.loads(content["text"])

    def _knowledge(self) -> KnowledgeService:
        if self._knowledge_service is None:
            self._knowledge_service = self._knowledge_service_factory()
        return self._knowledge_service

    def _map(self) -> Any:
        if self._map_service is None:
            self._map_service = self._map_service_factory()
        return self._map_service

    @staticmethod
    def _default_map_service() -> Any:
        from ..map_processing import MapProcessingService

        return MapProcessingService()

    @staticmethod
    def _map_config() -> Any:
        from ..map_processing.config import MapProcessingConfig

        return MapProcessingConfig.from_env()


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _bounds_from_any(value: Mapping[str, Any] | Bounds | None) -> Bounds | None:
    if value is None:
        return None
    if isinstance(value, Bounds):
        return value
    return Bounds(**dict(value))


def _gcp_dict(value: Mapping[str, Any] | list[float] | tuple[float, float, float, float]) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {
            "pixel_x": float(value["pixel_x"]),
            "pixel_y": float(value["pixel_y"]),
            "world_x": float(value["world_x"]),
            "world_y": float(value["world_y"]),
        }
    pixel_x, pixel_y, world_x, world_y = value
    return {
        "pixel_x": float(pixel_x),
        "pixel_y": float(pixel_y),
        "world_x": float(world_x),
        "world_y": float(world_y),
    }


def _legend_labels_from_metadata(metadata: Mapping[str, Any]) -> list[str]:
    explicit = metadata.get("legend_labels")
    if explicit:
        return [str(label) for label in explicit if str(label).strip()]
    labels: list[str] = []
    for entry in metadata.get("legend", []) or []:
        text: Any = None
        if isinstance(entry, Mapping):
            text = entry.get("label") or entry.get("text")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[1], Mapping):
            text = entry[1].get("text") or entry[1].get("label")
        if text and str(text).strip():
            labels.append(str(text))
    return labels


def _bundle_from_dict(data: Mapping[str, Any]) -> KnowledgeBundle:
    bounds = _bounds_from_any(data.get("bounds"))
    items = [KnowledgeItem.from_dict(dict(item)) for item in data.get("items", [])]
    return KnowledgeBundle(
        bounds=bounds,
        items=items,
        selected_item_ids=data.get("selected_item_ids"),
        warnings=list(data.get("warnings", [])),
        provider_versions=dict(data.get("provider_versions", {})),
        trace=data.get("trace"),
        schema_version=str(data.get("schema_version", "knowledge/v2")),
    )


def _georef_from_dict(data: Mapping[str, Any]) -> Any:
    from ..georef import AffineTransform, GeoReference

    coefficients = data.get("affine", {}).get("coefficients")
    if not coefficients:
        coefficients = [data["affine"][key] for key in ("a", "b", "c", "d", "e", "f")]
    affine = AffineTransform(*[float(value) for value in coefficients], residual=float(data["residual"]))
    return GeoReference(
        crs=str(data["crs"]),
        affine=affine,
        bounds=Bounds(**data["bounds"]),
        residual=float(data["residual"]),
    )


__all__ = ["GeomapMcpAdapter"]

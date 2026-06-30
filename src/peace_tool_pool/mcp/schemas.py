"""JSON schemas and redaction/conversion helpers for MCP tool payloads."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from ..knowledge.types import SCHEMA_VERSION as KNOWLEDGE_SCHEMA_VERSION
from ..map_processing.types import SCHEMA_VERSION as MAP_PROCESSING_SCHEMA_VERSION
from .resources import ResourceRegistry


JSON_SCHEMA = "https://json-schema.org/draft/2020-12/schema"

PATH_KEYS = {
    "asset_path",
    "artifact_path",
    "image_path",
    "local_path",
    "path",
    "source_path",
}


def new_trace_id() -> str:
    return uuid.uuid4().hex


def redact_paths(value: Any) -> Any:
    """Recursively remove local filesystem paths from model-visible payloads."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_string = str(key)
            if key_string in PATH_KEYS and isinstance(item, (str, Path)):
                redacted[key_string] = "<redacted>"
            else:
                redacted[key_string] = redact_paths(item)
        return redacted
    if isinstance(value, list):
        return [redact_paths(item) for item in value]
    if isinstance(value, tuple):
        return [redact_paths(item) for item in value]
    if isinstance(value, Path):
        return "<redacted>"
    if isinstance(value, str) and _looks_like_local_path(value):
        return "<redacted>"
    return value


def map_processing_result_to_mcp(
    result: Any,
    *,
    registry: ResourceRegistry,
    map_id: str,
) -> dict[str, Any]:
    """Convert an SDK ``MapProcessingResult`` to a path-redacted MCP payload."""

    map_info = registry.map_public(map_id)
    artifacts: list[dict[str, Any]] = []
    artifact_by_path: dict[str, dict[str, Any]] = {}
    for artifact in result.artifacts:
        public = registry.register_artifact(
            artifact.path,
            role=artifact.role,
            stage=artifact.stage,
            map_id=map_id,
            bbox=list(artifact.bbox) if artifact.bbox is not None else None,
            label=artifact.label,
            mime_type=artifact.mime_type,
        )
        artifact_by_path[str(Path(artifact.path).resolve())] = public
        artifacts.append(public)

    regions: dict[str, list[dict[str, Any]]] = {}
    for label, detections in result.regions.items():
        converted_detections: list[dict[str, Any]] = []
        for detection in detections:
            data = detection.to_dict()
            raw_artifact_path = data.pop("artifact_path", None)
            if raw_artifact_path:
                canonical = str(Path(raw_artifact_path).resolve())
                public_artifact = artifact_by_path.get(canonical)
                if public_artifact is None:
                    public_artifact = registry.register_artifact(
                        raw_artifact_path,
                        role="component_crop",
                        stage="hie",
                        map_id=map_id,
                        bbox=data.get("bbox"),
                        label=label,
                    )
                    artifact_by_path[canonical] = public_artifact
                    artifacts.append(public_artifact)
                data["artifact_uri"] = public_artifact["uri"]
            converted_detections.append(data)
        regions[label] = converted_detections

    payload = {
        "schema_version": MAP_PROCESSING_SCHEMA_VERSION,
        "date": result.created_date,
        "name": result.name,
        "version": result.version,
        "source": result.source,
        "source_uri": map_info["source_uri"],
        "map_uri": map_info["map_uri"],
        "size": result.size.to_dict(),
        "regions": regions,
        "legend": [entry.to_dict() for entry in result.legend],
        "artifacts": artifacts,
        "information": redact_paths(result.information),
        "faults": redact_paths(result.faults),
        "source_path_redacted": True,
    }
    return redact_paths(payload)


def knowledge_bundle_to_mcp(bundle: Any) -> dict[str, Any]:
    data = bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
    return redact_paths(data)


def legend_enrichment_to_mcp(enrichment: Any) -> dict[str, Any]:
    data = enrichment.to_dict() if hasattr(enrichment, "to_dict") else dict(enrichment)
    return redact_paths(data)


def serialize_georef(
    georef: Any,
    *,
    pixel_extent: list[float] | tuple[float, float, float, float],
    gcps: list[Mapping[str, float]],
    trace_id: str,
) -> dict[str, Any]:
    coefficients = [float(value) for value in georef.affine.coefficients]
    return {
        "schema_version": "georef/v1",
        "crs": georef.crs,
        "affine": {
            "coefficients": coefficients,
            "a": coefficients[0],
            "b": coefficients[1],
            "c": coefficients[2],
            "d": coefficients[3],
            "e": coefficients[4],
            "f": coefficients[5],
        },
        "bounds": georef.bounds.to_dict(),
        "residual": float(georef.residual),
        "pixel_extent": [float(value) for value in pixel_extent],
        "gcps": [dict(gcp) for gcp in gcps],
        "gcp_count": len(gcps),
        "trace_id": trace_id,
        "warnings": [],
    }


def success_result(
    *,
    structured: Mapping[str, Any],
    text_summary: str,
    trace_id: str,
    content: list[dict[str, Any]] | None = None,
    resource_links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = dict(structured)
    output.setdefault("trace_id", trace_id)
    output.setdefault("text_summary", text_summary)
    if resource_links:
        output["resource_links"] = list(resource_links)
    result_content = [{"type": "text", "text": text_summary}]
    if content:
        result_content.extend(content)
    return {
        "content": result_content,
        "structuredContent": output,
        "isError": False,
    }


def tool_definitions() -> list[dict[str, Any]]:
    return [
        _tool(
            "geomap_list_capabilities",
            "List installed geomap MCP capabilities and limits.",
            {},
            read_only=True,
            idempotent=True,
        ),
        _tool(
            "geomap_register_map",
            "Register a local map image under configured allowed roots.",
            {"path": {"type": "string"}},
            required=("path",),
            read_only=True,
            idempotent=True,
        ),
        _tool(
            "geomap_process_image",
            "Run layout and legend extraction for a registered map.",
            {"map_id": {"type": "string"}, "map_uri": {"type": "string"}},
            read_only=False,
            idempotent=True,
        ),
        _tool(
            "geomap_georeference",
            "Fit a georeference from VLM-read ground control points.",
            {
                "map_id": {"type": "string"},
                "map_uri": {"type": "string"},
                "crs": {"type": ["string", "integer"]},
                "gcps": {"type": "array", "items": GCP_SCHEMA},
                "pixel_extent": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number"},
                },
                "main_map_artifact_uri": {"type": "string"},
            },
            required=("crs", "gcps"),
            read_only=False,
            idempotent=True,
        ),
        _tool(
            "geomap_query_knowledge",
            "Query geological knowledge providers by bounds, labels, or text; persists a local bundle resource.",
            KNOWLEDGE_REQUEST_PROPERTIES,
            read_only=False,
            idempotent=True,
        ),
        _tool(
            "geomap_query_map",
            "Query knowledge for registered map state or inline map metadata; persists a local bundle resource.",
            {
                "map_id": {"type": "string"},
                "map_uri": {"type": "string"},
                "metadata": {"type": "object"},
                "question": {"type": "string"},
                **KNOWLEDGE_REQUEST_PROPERTIES,
            },
            read_only=False,
            idempotent=True,
        ),
        _tool(
            "geomap_enrich_legend",
            "Enrich one legend label with rock type and stratigraphic age.",
            {"label": {"type": "string"}},
            required=("label",),
            read_only=True,
            idempotent=True,
        ),
        _tool(
            "geomap_render_knowledge_overlay",
            "Render a knowledge bundle as SVG and, when possible, annotated PNG overlay resources.",
            {
                "map_id": {"type": "string"},
                "map_uri": {"type": "string"},
                "bundle_uri": {"type": "string"},
                "bundle": {"type": "object"},
                "georef": GEOREFERENCE_SCHEMA,
            },
            read_only=False,
            idempotent=True,
        ),
    ]


BOUNDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["min_lon", "min_lat", "max_lon", "max_lat"],
    "properties": {
        "min_lon": {"type": "number"},
        "min_lat": {"type": "number"},
        "max_lon": {"type": "number"},
        "max_lat": {"type": "number"},
        "crs": {"type": "string"},
    },
}

GCP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["pixel_x", "pixel_y", "world_x", "world_y"],
    "properties": {
        "pixel_x": {"type": "number"},
        "pixel_y": {"type": "number"},
        "world_x": {"type": "number"},
        "world_y": {"type": "number"},
    },
}

GEOREFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"const": "georef/v1"},
        "crs": {"type": "string"},
        "affine": {
            "type": "object",
            "properties": {
                "coefficients": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 6,
                    "items": {"type": "number"},
                }
            },
        },
        "bounds": BOUNDS_SCHEMA,
        "residual": {"type": "number"},
        "pixel_extent": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "number"},
        },
        "gcps": {"type": "array", "items": GCP_SCHEMA},
        "gcp_count": {"type": "integer"},
    },
}

KNOWLEDGE_REQUEST_PROPERTIES: dict[str, Any] = {
    "bounds": {"anyOf": [BOUNDS_SCHEMA, {"type": "null"}]},
    "legend_labels": {"type": "array", "items": {"type": "string"}},
    "query_text": {"type": ["string", "null"]},
    "include": {"type": "array", "items": {"type": "string"}},
    "exclude": {"type": "array", "items": {"type": "string"}},
    "max_records": {"type": ["integer", "null"], "minimum": 0},
    "max_records_by_provider": {
        "type": "object",
        "additionalProperties": {"type": "integer", "minimum": 0},
    },
    "provider_options": {"type": "object"},
}

STRUCTURED_CONTENT_SCHEMA: dict[str, Any] = {
    "$schema": JSON_SCHEMA,
    "title": "GeomapStructuredContent",
    "description": (
        "Envelope-only contract. Tool-specific success payloads are intentionally "
        "permissive until stable output schemas are promoted."
    ),
    "type": "object",
    "required": ["trace_id", "text_summary"],
    "properties": {
        "trace_id": {"type": "string"},
        "text_summary": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "resource_links": {"type": "array"},
    },
    "additionalProperties": True,
}


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
    read_only: bool,
    idempotent: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "$schema": JSON_SCHEMA,
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
        "outputSchema": dict(STRUCTURED_CONTENT_SCHEMA),
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": False,
            "idempotentHint": idempotent,
        },
    }


def _looks_like_local_path(value: str) -> bool:
    if value.startswith(("geomap://", "http://", "https://")):
        return False
    return value.startswith("/") or value.startswith("~")


__all__ = [
    "STRUCTURED_CONTENT_SCHEMA",
    "GEOREFERENCE_SCHEMA",
    "JSON_SCHEMA",
    "KNOWLEDGE_SCHEMA_VERSION",
    "MAP_PROCESSING_SCHEMA_VERSION",
    "knowledge_bundle_to_mcp",
    "legend_enrichment_to_mcp",
    "map_processing_result_to_mcp",
    "new_trace_id",
    "redact_paths",
    "serialize_georef",
    "success_result",
    "tool_definitions",
]

"""Lightweight knowledge service contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bounds import Bounds


SCHEMA_VERSION = "knowledge/v1"


@dataclass
class KnowledgeRequest:
    bounds: Bounds | None = None
    legend_labels: list[str] = field(default_factory=list)
    query_text: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    max_records: int | None = None
    max_records_by_provider: dict[str, int] = field(default_factory=dict)
    trace_id: str | None = None

    def __post_init__(self) -> None:
        self.legend_labels = [str(label) for label in self.legend_labels]
        self.include = tuple(str(item) for item in self.include)
        self.exclude = tuple(str(item) for item in self.exclude)
        self.max_records_by_provider = {
            str(provider): int(limit) for provider, limit in self.max_records_by_provider.items()
        }
        if self.max_records is not None:
            self.max_records = int(self.max_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict() if self.bounds is not None else None,
            "legend_labels": list(self.legend_labels),
            "query_text": self.query_text,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "max_records": self.max_records,
            "max_records_by_provider": dict(self.max_records_by_provider),
            "trace_id": self.trace_id,
        }


@dataclass
class KnowledgeItem:
    id: str
    key: str
    provider: str
    value: Any
    summary: str | None = None
    source: str | None = None
    record_count: int | None = None
    truncated: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "provider": self.provider,
            "value": self.value,
            "summary": self.summary,
            "source": self.source,
            "record_count": self.record_count,
            "truncated": self.truncated,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeItem":
        return cls(
            id=str(data["id"]),
            key=str(data["key"]),
            provider=str(data["provider"]),
            value=data.get("value"),
            summary=data.get("summary"),
            source=data.get("source"),
            record_count=data.get("record_count"),
            truncated=bool(data.get("truncated", False)),
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass
class KnowledgeBundle:
    bounds: Bounds | None
    items: list[KnowledgeItem]
    selected_item_ids: list[str] | None
    warnings: list[str]
    provider_versions: dict[str, str]
    trace: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bounds": self.bounds.to_dict() if self.bounds is not None else None,
            "items": [item.to_dict() for item in self.items],
            "selected_item_ids": self.selected_item_ids,
            "warnings": list(self.warnings),
            "provider_versions": dict(self.provider_versions),
            "trace": self.trace,
        }

    def items_by_id(self) -> dict[str, KnowledgeItem]:
        return {item.id: item for item in self.items}

    def items_by_key(self) -> dict[str, list[KnowledgeItem]]:
        by_key: dict[str, list[KnowledgeItem]] = {}
        for item in self.items:
            by_key.setdefault(item.key, []).append(item)
        return by_key


@dataclass
class LegendEnrichment:
    label: str
    lithology: str | None
    stratigraphic_age: str | None
    items: list[KnowledgeItem]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "lithology": self.lithology,
            "stratigraphic_age": self.stratigraphic_age,
            "items": [item.to_dict() for item in self.items],
            "warnings": list(self.warnings),
        }


KNOWLEDGE_BUNDLE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "KnowledgeBundle",
    "type": "object",
    "required": ["schema_version", "items", "warnings", "provider_versions"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "bounds": {"type": ["object", "null"]},
        "items": {"type": "array"},
        "selected_item_ids": {"type": ["array", "null"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "provider_versions": {"type": "object"},
        "trace": {"type": ["object", "null"]},
    },
}


LEGEND_ENRICHMENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LegendEnrichment",
    "type": "object",
    "required": ["label", "items", "warnings"],
    "properties": {
        "label": {"type": "string"},
        "lithology": {"type": ["string", "null"]},
        "stratigraphic_age": {"type": ["string", "null"]},
        "items": {"type": "array"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

"""Lightweight result schema and PEACE compatibility export."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

BBox = tuple[int, int, int, int]

SCHEMA_VERSION = "map-processing/v1"

COMPONENT_LABELS = (
    "title",
    "main_map",
    "legend",
    "scale",
    "index_map",
    "cross_section",
    "stratigraphic_column",
    "others",
)

LEGEND_LABELS = ("color_bndbox", "text_bndbox")


def normalize_bbox(value: Any) -> BBox:
    """Convert a four-value bbox-like object into integer xyxy coordinates."""
    if isinstance(value, Mapping):
        value = value.get("bbox") or value.get("bndbox")
    if value is None or len(value) < 4:
        raise ValueError(f"Expected bbox with four values, got {value!r}")
    x0, y0, x1, y1 = value[:4]
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


def bbox_to_list(bbox: BBox | None) -> list[int] | None:
    if bbox is None:
        return None
    return [int(value) for value in bbox]


@dataclass
class ImageSize:
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"width": int(self.width), "height": int(self.height)}


@dataclass
class Detection:
    label: str
    bbox: BBox
    confidence: float | None = None
    artifact_path: str | None = None

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"label": self.label, "bbox": bbox_to_list(self.bbox)}
        if self.confidence is not None:
            data["confidence"] = float(self.confidence)
        if self.artifact_path is not None:
            data["artifact_path"] = self.artifact_path
        return data


@dataclass
class ArtifactRef:
    path: str | Path
    role: str
    stage: str
    bbox: BBox | None = None
    label: str | None = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        if self.bbox is not None:
            self.bbox = normalize_bbox(self.bbox)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": str(self.path),
            "role": self.role,
            "stage": self.stage,
            "mime_type": self.mime_type,
        }
        if self.bbox is not None:
            data["bbox"] = bbox_to_list(self.bbox)
        if self.label is not None:
            data["label"] = self.label
        return data


@dataclass
class LegendEntry:
    id: int
    color_bbox: BBox
    text_bbox: BBox
    color_rgb: tuple[int, int, int]
    color_hex: str
    color_name: str
    label: str = ""
    area_fraction: float = 0.0
    lithology: str | None = None
    stratigraphic_age: str | None = None

    def __post_init__(self) -> None:
        self.color_bbox = normalize_bbox(self.color_bbox)
        self.text_bbox = normalize_bbox(self.text_bbox)
        self.color_rgb = tuple(int(value) for value in self.color_rgb)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": int(self.id),
            "color_bbox": bbox_to_list(self.color_bbox),
            "text_bbox": bbox_to_list(self.text_bbox),
            "color_rgb": list(self.color_rgb),
            "color_hex": self.color_hex,
            "color_name": self.color_name,
            "label": self.label,
            "area_fraction": float(self.area_fraction),
        }
        if self.lithology is not None:
            data["lithology"] = self.lithology
        if self.stratigraphic_age is not None:
            data["stratigraphic_age"] = self.stratigraphic_age
        return data

    def to_peace_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "color_bndbox": bbox_to_list(self.color_bbox),
            "text_bndbox": bbox_to_list(self.text_bbox),
            "color": list(self.color_rgb),
            "color_name": self.color_name,
            "text": self.label,
            "area": float(self.area_fraction),
            "color_hex": self.color_hex,
        }
        if self.lithology is not None:
            data["lithology"] = self.lithology
        if self.stratigraphic_age is not None:
            data["stratigraphic_age"] = self.stratigraphic_age
        return data


@dataclass
class MapProcessingResult:
    name: str
    source: str
    image_path: str | Path
    size: ImageSize
    regions: dict[str, list[Detection]] = field(default_factory=dict)
    legend: list[LegendEntry] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    information: dict[str, Any] = field(default_factory=dict)
    faults: Any = None
    created_date: str = field(default_factory=lambda: date.today().strftime("%Y%m%d"))
    version: str = "v1.0"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "date": self.created_date,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "image_path": str(self.image_path),
            "size": self.size.to_dict(),
            "regions": {
                label: [detection.to_dict() for detection in detections]
                for label, detections in self.regions.items()
            },
            "legend": [entry.to_dict() for entry in self.legend],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "information": self.information,
            "faults": self.faults,
        }

    def to_peace_metadata(self) -> dict[str, Any]:
        labels = list(COMPONENT_LABELS)
        labels.extend(label for label in self.regions if label not in labels)
        return {
            "date": self.created_date,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "size": self.size.to_dict(),
            "regions": {
                label: [bbox_to_list(detection.bbox) for detection in self.regions.get(label, [])]
                for label in labels
            },
            "legend": {entry.id: entry.to_peace_dict() for entry in self.legend},
            "information": self.information,
            "faults": self.faults,
        }


MAP_PROCESSING_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "MapProcessingResult",
    "type": "object",
    "required": ["schema_version", "name", "source", "size", "regions", "legend"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "date": {"type": "string"},
        "name": {"type": "string"},
        "version": {"type": "string"},
        "source": {"type": "string"},
        "image_path": {"type": "string"},
        "size": {
            "type": "object",
            "required": ["width", "height"],
            "properties": {"width": {"type": "integer"}, "height": {"type": "integer"}},
        },
        "regions": {"type": "object"},
        "legend": {"type": "array"},
        "artifacts": {"type": "array"},
        "information": {"type": "object"},
        "faults": {},
    },
}

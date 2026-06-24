"""Detector protocols and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from ..types import Detection, normalize_bbox


class ComponentDetector(Protocol):
    def detect(self, image_path: str | Path) -> Mapping[str, Any]:
        """Detect map components in an image."""


class LegendDetector(Protocol):
    def detect(self, image_path: str | Path) -> Mapping[str, Any]:
        """Detect legend color/text boxes in an image."""


def _coerce_detection(label: str, value: Any) -> Detection:
    if isinstance(value, Detection):
        if value.label == label:
            return value
        return Detection(label=label, bbox=value.bbox, confidence=value.confidence)
    if isinstance(value, Mapping):
        return Detection(
            label=label,
            bbox=normalize_bbox(value),
            confidence=value.get("confidence") or value.get("conf"),
        )
    confidence = value[4] if hasattr(value, "__len__") and len(value) > 4 else None
    return Detection(label=label, bbox=normalize_bbox(value), confidence=confidence)


def normalize_detection_map(
    raw: Mapping[str, Any],
    default_labels: tuple[str, ...] = (),
) -> dict[str, list[Detection]]:
    detections: dict[str, list[Detection]] = {label: [] for label in default_labels}
    for label, values in raw.items():
        label_name = str(label)
        detections.setdefault(label_name, [])
        for value in values or []:
            detections[label_name].append(_coerce_detection(label_name, value))
    return detections

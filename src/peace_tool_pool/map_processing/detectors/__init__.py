"""Detector adapters for map processing."""

from .base import ComponentDetector, LegendDetector, normalize_detection_map
from .peace_yolov10 import YoloV10LegendDetector, YoloV10MapComponentDetector

__all__ = [
    "ComponentDetector",
    "LegendDetector",
    "YoloV10LegendDetector",
    "YoloV10MapComponentDetector",
    "normalize_detection_map",
]

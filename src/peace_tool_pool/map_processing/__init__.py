"""Local geologic map image processing tools."""

from .config import MapProcessingConfig
from .service import MapProcessingService
from .types import (
    ArtifactRef,
    Detection,
    ImageSize,
    LegendEntry,
    MapProcessingResult,
    MAP_PROCESSING_RESULT_SCHEMA,
)

__all__ = [
    "ArtifactRef",
    "Detection",
    "ImageSize",
    "LegendEntry",
    "MapProcessingConfig",
    "MapProcessingResult",
    "MapProcessingService",
    "MAP_PROCESSING_RESULT_SCHEMA",
]

"""Small batch helpers for local USGS processing smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .service import MapProcessingService
from .types import MapProcessingResult


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_image_paths(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def process_images(
    image_paths: Iterable[str | Path],
    service: MapProcessingService | None = None,
) -> list[MapProcessingResult]:
    processor = service or MapProcessingService()
    return [processor.process_image(path) for path in image_paths]


def process_directory(
    image_dir: str | Path,
    service: MapProcessingService | None = None,
    limit: int | None = None,
) -> list[MapProcessingResult]:
    image_paths = iter_image_paths(image_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    return process_images(image_paths, service=service)

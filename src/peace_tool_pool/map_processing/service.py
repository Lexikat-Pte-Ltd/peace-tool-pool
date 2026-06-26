"""Local map image processing service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import image_ops
from .cache import MapProcessingCache
from .config import MapProcessingConfig
from .detectors.base import normalize_detection_map
from .detectors.peace_yolov10 import YoloV10LegendDetector, YoloV10MapComponentDetector
from .legend import build_legend_entries
from .types import ArtifactRef, COMPONENT_LABELS, Detection, ImageSize, MapProcessingResult


class MapProcessingService:
    def __init__(
        self,
        config: MapProcessingConfig | None = None,
        component_detector: Any | None = None,
        legend_detector: Any | None = None,
    ):
        self.config = config or MapProcessingConfig.from_env()
        self.cache = MapProcessingCache(self.config)
        self.component_detector = component_detector or YoloV10MapComponentDetector(
            self.config.resolved_component_model_path,
            self.config.resolved_ultralytics_root,
        )
        self.legend_detector = legend_detector or YoloV10LegendDetector(
            self.config.resolved_legend_model_path,
            self.config.resolved_ultralytics_root,
        )

    def process_image(self, image_path: str | Path) -> MapProcessingResult:
        image_path = Path(image_path)
        image = image_ops.read_image(image_path)
        width, height = image_ops.image_size(image)
        name = image_path.stem

        raw_regions = self.component_detector.detect(image_path)
        regions = normalize_detection_map(raw_regions, COMPONENT_LABELS)
        result = MapProcessingResult(
            name=name,
            source=self.config.dataset_source,
            image_path=image_path,
            size=ImageSize(width=width, height=height),
            regions=regions,
        )

        region_artifacts = self._write_component_artifacts(image, result)
        self._extract_legend(result, region_artifacts)
        self._estimate_areas(result, region_artifacts)
        self._write_detection_overlay(image, result)

        if self.config.write_cache:
            self.cache.save_result(result)
        return result

    def _write_component_artifacts(
        self,
        image: Any,
        result: MapProcessingResult,
    ) -> dict[str, list[Path]]:
        artifact_paths: dict[str, list[Path]] = {}
        for label, detections in result.regions.items():
            artifact_paths.setdefault(label, [])
            for index, detection in enumerate(detections):
                artifact_path = self.cache.component_path(result.name, label, index)
                image_ops.crop_and_save_image(image, detection.bbox, artifact_path)
                detection.artifact_path = str(artifact_path)
                artifact_paths[label].append(artifact_path)
                result.artifacts.append(
                    ArtifactRef(
                        path=artifact_path,
                        role="component_crop",
                        stage="hie",
                        bbox=detection.bbox,
                        label=label,
                    )
                )
                if label == "main_map":
                    self._write_lonlat_artifact(result, artifact_path, index, artifact_paths)
                elif label == "index_map":
                    image_ops.annotate_image_with_directions(artifact_path, artifact_path)
        return artifact_paths

    def _write_detection_overlay(self, image: Any, result: MapProcessingResult) -> None:
        artifact_path = self.cache.visualization_path(result.name)
        image_ops.annotate_detections_on_image(
            image,
            result.regions,
            artifact_path,
            legend_entries=result.legend,
        )
        result.artifacts.append(
            ArtifactRef(
                path=artifact_path,
                role="detection_overlay",
                stage="hie",
            )
        )

    def _write_lonlat_artifact(
        self,
        result: MapProcessingResult,
        main_map_path: Path,
        index: int,
        artifact_paths: dict[str, list[Path]],
    ) -> None:
        lonlat_path = self.cache.component_path(result.name, "lonlat", index)
        image_ops.crop_corners_and_save_image(main_map_path, lonlat_path)
        artifact_paths.setdefault("lonlat", []).append(lonlat_path)
        result.artifacts.append(
            ArtifactRef(path=lonlat_path, role="lonlat_corner_crop", stage="hie", label="lonlat")
        )

    def _extract_legend(
        self,
        result: MapProcessingResult,
        artifact_paths: dict[str, list[Path]],
    ) -> None:
        legend_paths = artifact_paths.get("legend", [])
        legend_detections = result.regions.get("legend", [])
        if not legend_paths or not legend_detections:
            return
        legend_path = legend_paths[0]
        legend_detection: Detection = legend_detections[0]
        legend_image = image_ops.read_image(legend_path)
        legend_units = self.legend_detector.detect(legend_path)
        result.legend = build_legend_entries(legend_image, legend_units, legend_detection.bbox)

    def _estimate_areas(
        self,
        result: MapProcessingResult,
        artifact_paths: dict[str, list[Path]],
    ) -> None:
        main_map_paths = artifact_paths.get("main_map", [])
        if not main_map_paths or not result.legend:
            return
        image_ops.estimate_legend_area_fractions(main_map_paths[0], result.legend)

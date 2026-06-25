"""Optional Google Earth Engine knowledge providers."""

from __future__ import annotations

from typing import Any

from ..bounds import Bounds
from ..errors import OptionalDependencyError
from ..types import KnowledgeItem, KnowledgeRequest


LANDCOVER_CLASS_NAMES = {
    10: "Trees",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / Sparse Vegetation",
    70: "Snow and Ice",
    80: "Permanent Water Bodies",
    90: "Herbaceous Wetland",
    95: "Mangroves",
    100: "Moss and Lichen",
}


class _EarthEngineProviderBase:
    version = "1"

    def __init__(
        self,
        *,
        dataset_id: str,
        project: str | None = None,
        scale: int = 100,
        max_pixels: int = 100_000_000,
        ee_module: Any | None = None,
    ):
        self.dataset_id = dataset_id
        self.project = project
        self.scale = int(scale)
        self.max_pixels = int(max_pixels)
        self._ee_module = ee_module
        self._initialized = False
        self._dataset: Any | None = None

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        return f"{self.version}@earthengine:{self.dataset_id}"

    def cache_config(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "project": self.project,
            "scale": self.scale,
            "max_pixels": self.max_pixels,
        }

    def _ee(self) -> Any:
        if self._ee_module is None:
            try:
                import ee
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Earth Engine providers require `uv sync --extra knowledge-earthengine`."
                ) from exc
            self._ee_module = ee
        if not self._initialized:
            self._ee_module.Initialize(project=self.project)
            self._initialized = True
        return self._ee_module

    def _region(self, bounds: Bounds) -> Any:
        ee = self._ee()
        return ee.Geometry.Rectangle([bounds.min_lon, bounds.min_lat, bounds.max_lon, bounds.max_lat])

    def _image(self) -> Any:
        if self._dataset is None:
            self._dataset = self._ee().ImageCollection(self.dataset_id).mosaic()
        return self._dataset


class EarthEngineLandcoverProvider(_EarthEngineProviderBase):
    id = "landcover_distribution"
    name = "Landcover distribution"
    output_keys = ("landcover_distribution",)

    def __init__(
        self,
        *,
        dataset_id: str = "ESA/WorldCover/v200",
        project: str | None = None,
        scale: int = 100,
        max_pixels: int = 100_000_000,
        ee_module: Any | None = None,
    ):
        super().__init__(
            dataset_id=dataset_id,
            project=project,
            scale=scale,
            max_pixels=max_pixels,
            ee_module=ee_module,
        )

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        if request.bounds is None:
            return []
        ee = self._ee()
        region = self._region(request.bounds)
        histogram = (
            self._image()
            .clip(region)
            .reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=region,
                scale=self.scale,
                maxPixels=self.max_pixels,
                bestEffort=True,
            )
            .get("Map")
            .getInfo()
            or {}
        )
        total_pixels = max(1.0, sum(float(value) for value in histogram.values()))
        distribution: dict[str, float] = {}
        for code, count in sorted(histogram.items(), key=lambda item: self._class_code(item[0])):
            class_code = self._class_code(code)
            class_name = LANDCOVER_CLASS_NAMES.get(class_code, f"Class {code}")
            distribution[class_name] = round((float(count) / total_pixels) * 100.0, 3)
        return [
            KnowledgeItem(
                id=f"{self.id}:{self.id}",
                key=self.id,
                provider=self.id,
                value=distribution,
                summary=f"Computed landcover distribution for {len(distribution)} classes.",
                source=self.dataset_id,
                record_count=len(distribution),
                truncated=False,
                provenance=self.cache_config(),
            )
        ]

    def _class_code(self, code: Any) -> int:
        return int(float(code))


class EarthEnginePopulationDensityProvider(_EarthEngineProviderBase):
    id = "population_density"
    name = "Population density"
    output_keys = ("population_density",)

    def __init__(
        self,
        *,
        dataset_id: str = "WorldPop/GP/100m/pop",
        project: str | None = None,
        scale: int = 100,
        max_pixels: int = 100_000_000,
        ee_module: Any | None = None,
    ):
        super().__init__(
            dataset_id=dataset_id,
            project=project,
            scale=scale,
            max_pixels=max_pixels,
            ee_module=ee_module,
        )

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        if request.bounds is None:
            return []
        ee = self._ee()
        region = self._region(request.bounds)
        population_total = (
            self._image()
            .clip(region)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=self.scale,
                maxPixels=self.max_pixels,
                bestEffort=True,
            )
            .get("population")
            .getInfo()
            or 0
        )
        area_km2 = max(1e-6, float(region.area().getInfo()) / 1_000_000)
        density = round(float(population_total) / area_km2, 2)
        value = {
            "population_total": population_total,
            "area_km2": round(area_km2, 6),
            "density_people_per_km2": density,
            "label": f"{density} people/km^2",
        }
        return [
            KnowledgeItem(
                id=f"{self.id}:{self.id}",
                key=self.id,
                provider=self.id,
                value=value,
                summary=f"Computed population density as {value['label']}.",
                source=self.dataset_id,
                record_count=1,
                truncated=False,
                provenance=self.cache_config(),
            )
        ]

"""Bounds validation for knowledge providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import InvalidBoundsError


@dataclass
class Bounds:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        try:
            self.min_lon = float(self.min_lon)
            self.min_lat = float(self.min_lat)
            self.max_lon = float(self.max_lon)
            self.max_lat = float(self.max_lat)
        except (TypeError, ValueError) as exc:
            raise InvalidBoundsError("Bounds values must be decimal degree numbers.") from exc

        normalized_crs = str(self.crs).upper()
        if normalized_crs == "OGC:CRS84":
            normalized_crs = "EPSG:4326"
        if normalized_crs != "EPSG:4326":
            raise InvalidBoundsError(
                f"Unsupported bounds CRS {self.crs!r}; only EPSG:4326 is supported."
            )
        self.crs = normalized_crs

        if not -180 <= self.min_lon <= 180 or not -180 <= self.max_lon <= 180:
            raise InvalidBoundsError("Longitude bounds must be between -180 and 180 degrees.")
        if not -90 <= self.min_lat <= 90 or not -90 <= self.max_lat <= 90:
            raise InvalidBoundsError("Latitude bounds must be between -90 and 90 degrees.")
        if self.min_lon > self.max_lon:
            raise InvalidBoundsError(
                "Antimeridian-crossing bounds are not supported in this phase."
            )
        if self.min_lat > self.max_lat:
            raise InvalidBoundsError("min_lat must be less than or equal to max_lat.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_lon": self.min_lon,
            "min_lat": self.min_lat,
            "max_lon": self.max_lon,
            "max_lat": self.max_lat,
            "crs": self.crs,
        }

    def to_cache_dict(self, precision: int) -> dict[str, Any]:
        return {
            "min_lon": round(self.min_lon, precision),
            "min_lat": round(self.min_lat, precision),
            "max_lon": round(self.max_lon, precision),
            "max_lat": round(self.max_lat, precision),
            "crs": self.crs,
        }

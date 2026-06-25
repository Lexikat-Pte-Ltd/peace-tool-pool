"""Affine fit from ground control points and reprojection to EPSG:4326 bounds.

This is the deterministic core a VLM cannot do reliably: given grid tick labels
(GCPs: pixel position -> projected world coordinate) and the CRS, fit a
pixel->world affine, reproject the map extent corners to lon/lat, and return a
validated :class:`~peace_tool_pool.knowledge.bounds.Bounds`.

The agent supplies the GCPs (reading printed tick labels is its strength); this
module owns the projection math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..knowledge.bounds import Bounds
from .crs import resolve_crs
from .errors import AffineFitError, GeoReferenceError

try:  # numpy backs the >=3-GCP least-squares fit; part of the 'geo' extra.
    import numpy as _np
except ImportError as exc:  # pragma: no cover - exercised only without the extra.
    _np = None
    _NUMPY_IMPORT_ERROR: ImportError | None = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import pyproj as _pyproj
except ImportError as exc:  # pragma: no cover - exercised only without the extra.
    _pyproj = None
    _PYPROJ_IMPORT_ERROR: ImportError | None = exc
else:
    _PYPROJ_IMPORT_ERROR = None


@dataclass(frozen=True)
class GroundControlPoint:
    """A pixel position paired with its world (projected) coordinate."""

    pixel_x: float
    pixel_y: float
    world_x: float  # easting (projected CRS) or longitude
    world_y: float  # northing (projected CRS) or latitude


@dataclass(frozen=True)
class AffineTransform:
    """Pixel->world affine: ``wx = a*px + b*py + c``, ``wy = d*px + e*py + f``."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    residual: float = 0.0  # RMS fit error in world units (QA signal)

    def apply(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        wx = self.a * pixel_x + self.b * pixel_y + self.c
        wy = self.d * pixel_x + self.e * pixel_y + self.f
        return wx, wy

    @property
    def coefficients(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.e, self.f)


@dataclass
class GeoReference:
    """Result of georeferencing: resolved CRS, affine, lon/lat bounds, residual."""

    crs: str
    affine: AffineTransform
    bounds: Bounds
    residual: float
    transformer: object = field(default=None, repr=False, compare=False)

    def pixel_to_lonlat(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Map a pixel coordinate to (lon, lat) in EPSG:4326."""
        transformer = self.transformer or _build_transformer(self.crs)
        wx, wy = self.affine.apply(pixel_x, pixel_y)
        lon, lat = transformer.transform(wx, wy)
        return float(lon), float(lat)


def _fit_axis_aligned(p0: GroundControlPoint, p1: GroundControlPoint) -> AffineTransform:
    dpx = p1.pixel_x - p0.pixel_x
    dpy = p1.pixel_y - p0.pixel_y
    if dpx == 0 or dpy == 0:
        raise AffineFitError(
            "Two GCPs must differ in both pixel-x and pixel-y for an axis-aligned fit."
        )
    a = (p1.world_x - p0.world_x) / dpx
    e = (p1.world_y - p0.world_y) / dpy
    c = p0.world_x - a * p0.pixel_x
    f = p0.world_y - e * p0.pixel_y
    return AffineTransform(a=a, b=0.0, c=c, d=0.0, e=e, f=f, residual=0.0)


def fit_affine(gcps: Sequence[GroundControlPoint]) -> AffineTransform:
    """Fit a pixel->world affine. 2 GCPs -> axis-aligned; >=3 -> least squares."""
    points = list(gcps)
    if len(points) < 2:
        raise AffineFitError("At least 2 ground control points are required.")
    if len(points) == 2:
        return _fit_axis_aligned(points[0], points[1])

    if _np is None:  # pragma: no cover - exercised only without the extra.
        raise GeoReferenceError(
            "numpy is required to fit an affine from 3+ GCPs; install the 'geo' extra."
        ) from _NUMPY_IMPORT_ERROR

    matrix = _np.array([[p.pixel_x, p.pixel_y, 1.0] for p in points], dtype=float)
    if _np.linalg.matrix_rank(matrix) < 3:
        raise AffineFitError("Ground control points are collinear or degenerate.")

    world_x = _np.array([p.world_x for p in points], dtype=float)
    world_y = _np.array([p.world_y for p in points], dtype=float)
    coef_x, *_ = _np.linalg.lstsq(matrix, world_x, rcond=None)
    coef_y, *_ = _np.linalg.lstsq(matrix, world_y, rcond=None)
    rms = float(
        _np.sqrt(
            _np.mean((matrix @ coef_x - world_x) ** 2 + (matrix @ coef_y - world_y) ** 2)
        )
    )
    a, b, c = (float(v) for v in coef_x)
    d, e, f = (float(v) for v in coef_y)
    return AffineTransform(a=a, b=b, c=c, d=d, e=e, f=f, residual=rms)


def _build_transformer(epsg: str):
    if _pyproj is None:
        raise GeoReferenceError(
            "pyproj is required for reprojection; install the 'geo' extra."
        ) from _PYPROJ_IMPORT_ERROR
    return _pyproj.Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)


def georeference_bounds(
    *,
    crs: str | int,
    gcps: Sequence[GroundControlPoint],
    pixel_extent: tuple[float, float, float, float],
) -> GeoReference:
    """Georeference a pixel extent to an EPSG:4326 :class:`Bounds`.

    ``pixel_extent`` is ``(x0, y0, x1, y1)`` in pixels — typically the detected
    ``main_map`` bbox (or the full image). All four corners are reprojected so a
    rotated/skewed grid is handled correctly.
    """
    epsg = resolve_crs(crs)
    affine = fit_affine(gcps)
    transformer = _build_transformer(epsg)

    x0, y0, x1, y1 = pixel_extent
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    lons: list[float] = []
    lats: list[float] = []
    for px, py in corners:
        wx, wy = affine.apply(px, py)
        lon, lat = transformer.transform(wx, wy)
        lons.append(float(lon))
        lats.append(float(lat))

    bounds = Bounds(
        min_lon=min(lons), min_lat=min(lats), max_lon=max(lons), max_lat=max(lats)
    )
    return GeoReference(
        crs=epsg, affine=affine, bounds=bounds, residual=affine.residual, transformer=transformer
    )

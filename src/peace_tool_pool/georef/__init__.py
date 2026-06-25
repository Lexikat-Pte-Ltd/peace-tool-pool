"""Georeferencing: turn map grid tick GCPs + a CRS into an EPSG:4326 Bounds.

The agent reads printed grid tick labels and their pixel positions; this package
owns the deterministic projection math (CRS resolution, affine fit, reprojection)
that a VLM cannot do reliably. The resulting ``Bounds`` feeds
``KnowledgeService.query_bounds``.

Requires the ``geo`` extra (``pyproj``, ``numpy``).
"""

from __future__ import annotations

from .crs import resolve_crs
from .errors import AffineFitError, CRSResolutionError, GeoReferenceError
from .transform import (
    AffineTransform,
    GeoReference,
    GroundControlPoint,
    fit_affine,
    georeference_bounds,
)

__all__ = [
    "resolve_crs",
    "GeoReferenceError",
    "CRSResolutionError",
    "AffineFitError",
    "AffineTransform",
    "GeoReference",
    "GroundControlPoint",
    "fit_affine",
    "georeference_bounds",
]

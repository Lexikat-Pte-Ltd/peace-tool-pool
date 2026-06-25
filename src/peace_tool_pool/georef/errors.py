"""Errors for georeferencing."""

from __future__ import annotations


class GeoReferenceError(Exception):
    """Base class for georeferencing failures."""


class CRSResolutionError(GeoReferenceError):
    """Raised when a CRS specification cannot be resolved to an EPSG code."""


class AffineFitError(GeoReferenceError):
    """Raised when an affine transform cannot be fit from the given GCPs."""

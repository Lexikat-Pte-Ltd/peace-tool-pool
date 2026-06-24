"""Errors raised by map processing services."""


class MapProcessingError(RuntimeError):
    """Base error for map processing failures."""


class OptionalDependencyError(ImportError):
    """Raised when an optional processing dependency is not installed."""


class DetectorLoadError(MapProcessingError):
    """Raised when a detector backend cannot be loaded."""

"""Errors raised by knowledge services."""


class KnowledgeError(RuntimeError):
    """Base error for knowledge service failures."""


class InvalidBoundsError(KnowledgeError):
    """Raised when geographic bounds are invalid or unsupported."""


class MissingAssetError(KnowledgeError):
    """Raised when a configured local knowledge asset is missing."""


class ProviderError(KnowledgeError):
    """Raised when a provider cannot satisfy an explicit request."""


class OptionalDependencyError(ImportError):
    """Raised when an optional knowledge dependency is not installed."""

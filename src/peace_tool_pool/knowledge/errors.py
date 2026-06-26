"""Errors raised by knowledge services."""


class KnowledgeError(RuntimeError):
    """Base error for knowledge service failures."""


class InvalidBoundsError(KnowledgeError):
    """Raised when geographic bounds are invalid or unsupported."""


class MissingAssetError(KnowledgeError):
    """Raised when a configured local knowledge asset is missing."""


class ProviderError(KnowledgeError):
    """Raised when a provider cannot satisfy an explicit request."""


class ProviderOptionError(KnowledgeError):
    """Raised when provider-specific options are malformed or incompatible."""


class SourceRegistryError(KnowledgeError):
    """Raised when a knowledge source or source family is unknown."""


class SourceManifestError(KnowledgeError):
    """Raised when a source manifest cannot be parsed or validated."""


class SourceSyncError(KnowledgeError):
    """Raised when source acquisition or normalization fails."""


class SourceQueryError(SourceSyncError):
    """Raised when an upstream source query is invalid or too large."""


class SourceChecksumError(SourceSyncError):
    """Raised when an artifact checksum does not match expected metadata."""


class OptionalDependencyError(ImportError):
    """Raised when an optional knowledge dependency is not installed."""

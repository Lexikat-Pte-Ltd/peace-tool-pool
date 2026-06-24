"""Local knowledge provider implementations."""

from .base import KnowledgeProvider
from .earthquakes import EarthquakeHistoryProvider
from .faults import ActiveFaultProvider
from .rock import RockLookupProvider

__all__ = [
    "ActiveFaultProvider",
    "EarthquakeHistoryProvider",
    "KnowledgeProvider",
    "RockLookupProvider",
]

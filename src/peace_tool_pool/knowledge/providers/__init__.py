"""Local knowledge provider implementations."""

from .base import KnowledgeProvider
from .earthquakes import EarthquakeHistoryProvider
from .faults import ActiveFaultProvider
from .minerals import MineralOccurrenceProvider
from .rock import RockLookupProvider

__all__ = [
    "ActiveFaultProvider",
    "EarthquakeHistoryProvider",
    "KnowledgeProvider",
    "MineralOccurrenceProvider",
    "RockLookupProvider",
]

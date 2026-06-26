"""Reusable geological knowledge services."""

from .bounds import Bounds
from .config import KnowledgeConfig
from .service import KnowledgeService
from .types import (
    KNOWLEDGE_BUNDLE_SCHEMA,
    LEGEND_ENRICHMENT_SCHEMA,
    SCHEMA_VERSION,
    KnowledgeBundle,
    KnowledgeItem,
    KnowledgeRequest,
    LegendEnrichment,
)
from .visualization import (
    KNOWLEDGE_OVERLAY_COLORS_RGB,
    KnowledgeOverlay,
    KnowledgeOverlayFrame,
    KnowledgeOverlayItem,
    extract_knowledge_overlay,
    render_knowledge_overlay_svg,
)

__all__ = [
    "Bounds",
    "KNOWLEDGE_BUNDLE_SCHEMA",
    "KNOWLEDGE_OVERLAY_COLORS_RGB",
    "KnowledgeBundle",
    "KnowledgeConfig",
    "KnowledgeItem",
    "KnowledgeOverlay",
    "KnowledgeOverlayFrame",
    "KnowledgeOverlayItem",
    "KnowledgeRequest",
    "KnowledgeService",
    "LEGEND_ENRICHMENT_SCHEMA",
    "LegendEnrichment",
    "SCHEMA_VERSION",
    "extract_knowledge_overlay",
    "render_knowledge_overlay_svg",
]

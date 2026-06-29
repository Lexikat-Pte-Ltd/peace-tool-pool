"""Visualization helpers for bbox-backed knowledge lookups."""

from __future__ import annotations

import base64
import html
import io
import math
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bounds import Bounds
from .types import KnowledgeBundle, KnowledgeItem


KNOWLEDGE_OVERLAY_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "query_bounds": (37, 99, 235),
    "provider_bounds": (71, 85, 105),
    "active_faults": (220, 38, 38),
    "earthquake_history": (245, 158, 11),
    "mineral_occurrences": (22, 163, 74),
    "landcover_distribution": (8, 145, 178),
    "population_density": (147, 51, 234),
    "default": (71, 85, 105),
}


@dataclass
class KnowledgeOverlayItem:
    id: str
    kind: str
    label: str
    color_rgb: tuple[int, int, int]
    provider: str | None = None
    key: str | None = None
    bounds: Bounds | None = None
    lon: float | None = None
    lat: float | None = None
    source: str | None = None
    record_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "color_rgb": list(self.color_rgb),
        }
        if self.provider is not None:
            data["provider"] = self.provider
        if self.key is not None:
            data["key"] = self.key
        if self.bounds is not None:
            data["bounds"] = self.bounds.to_dict()
        if self.lon is not None and self.lat is not None:
            data["longitude"] = self.lon
            data["latitude"] = self.lat
        if self.source is not None:
            data["source"] = self.source
        if self.record_index is not None:
            data["record_index"] = self.record_index
        return data


@dataclass
class KnowledgeOverlayFrame:
    source: str = "geographic_canvas"
    crs: str = "EPSG:4326"
    bounds: Bounds | None = None
    bounds_parts: list[Bounds] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    image_path: str | None = None
    georef: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "crs": self.crs,
            "bounds": self.bounds.to_dict() if self.bounds is not None else None,
            "bounds_parts": [part.to_dict() for part in self.bounds_parts],
            "item_ids": list(self.item_ids),
        }
        if self.image_path is not None:
            data["image_path"] = self.image_path
        if self.georef is not None:
            data["georef"] = dict(self.georef)
        return data


@dataclass
class KnowledgeOverlay:
    frame: KnowledgeOverlayFrame
    items: list[KnowledgeOverlayItem] = field(default_factory=list)
    out_of_bounds: list[KnowledgeOverlayItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "out_of_bounds": [item.to_dict() for item in self.out_of_bounds],
            "warnings": list(self.warnings),
        }


def extract_knowledge_overlay(
    bundle: KnowledgeBundle,
    *,
    metadata: Mapping[str, Any] | None = None,
    include_provider_bounds: bool = True,
) -> KnowledgeOverlay:
    """Extract renderable lookup extents and result geometries from a bundle.

    The current knowledge bundle serializes bbox parts as dictionaries in trace
    and provenance, while provider values use provider-specific shapes. This
    function centralizes those shapes so renderers do not need provider logic.
    """

    bounds_parts = _bounds_parts_from_trace(bundle.trace)
    items: list[KnowledgeOverlayItem] = []

    for index, bounds in enumerate(bounds_parts):
        items.append(
            KnowledgeOverlayItem(
                id=f"query_bounds:part:{index}",
                kind="query_bounds",
                label=f"query part {index + 1}",
                bounds=bounds,
                color_rgb=KNOWLEDGE_OVERLAY_COLORS_RGB["query_bounds"],
                source="bundle.trace.bounds_parts",
            )
        )

    # The target map is the queried region. Result geometries that land outside it
    # cannot be a faithful annotation of that map -- they signal a coordinate/CRS
    # misalignment. Hide them from the plot but surface them as warnings so the
    # failure is never silent. Without query bounds there is nothing to validate
    # against, so every result is kept (and the frame falls back to its extent).
    out_of_bounds: list[KnowledgeOverlayItem] = []
    for knowledge_item in bundle.items:
        if include_provider_bounds:
            items.extend(_provider_bounds_items(knowledge_item))
        for geometry in _result_geometry_items(knowledge_item):
            if bounds_parts and not _result_within_parts(geometry, bounds_parts):
                out_of_bounds.append(geometry)
            else:
                items.append(geometry)

    frame_bounds = _bounds_union(bounds_parts) if bounds_parts else _bounds_for_items(items)
    overlay_warnings = _out_of_bounds_warnings(out_of_bounds, frame_bounds)
    for message in overlay_warnings:
        _warnings.warn(message, stacklevel=2)

    frame = KnowledgeOverlayFrame(
        source="geographic_canvas",
        crs="EPSG:4326",
        bounds=frame_bounds,
        bounds_parts=bounds_parts,
        item_ids=[item.id for item in items],
        image_path=_metadata_image_path(metadata),
        georef=_metadata_georef(metadata),
    )
    return KnowledgeOverlay(frame=frame, items=items, out_of_bounds=out_of_bounds, warnings=overlay_warnings)


def render_knowledge_overlay_svg(
    overlay: KnowledgeOverlay,
    output_path: str | Path,
    *,
    width: int = 1000,
    height: int = 700,
    title: str = "Knowledge lookup overlay",
) -> Path:
    """Render a geographic knowledge overlay as a standalone SVG artifact."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    width = int(width)
    height = int(height)
    bounds = overlay.frame.bounds
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #0f172a; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #475569; }",
        ".axis { font-size: 11px; fill: #64748b; }",
        ".warning { font-size: 12px; font-weight: 700; fill: #b91c1c; }",
        ".label { font-size: 12px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 3px; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="34" class="title">{_escape(title)}</text>',
    ]

    if bounds is None:
        lines.extend(
            [
                '<text x="24" y="64" class="subtitle">No geospatial overlay items were found.</text>',
                "</svg>",
            ]
        )
        target.write_text("\n".join(lines), encoding="utf-8")
        return target

    project = _projector(bounds, width, height)
    lines.append(
        '<text x="24" y="54" class="subtitle">'
        f'Frame: lon [{bounds.min_lon:.4f}, {bounds.max_lon:.4f}], '
        f'lat [{bounds.min_lat:.4f}, {bounds.max_lat:.4f}] · {overlay.frame.crs}'
        "</text>"
    )
    if overlay.warnings:
        note = (
            f"⚠ {len(overlay.out_of_bounds)} annotation(s) outside the target map "
            "bounds were hidden (possible CRS misalignment)."
        )
        lines.append(f'<text x="24" y="72" class="warning">{_escape(note)}</text>')

    # Draw the georeferenced input map as the plot background when available, so
    # annotations sit over the real map instead of a blank panel. The image is
    # cropped to the georef pixel_extent (the main-map area) and stretched to the
    # plot box, which the frame bounds describe; without an image we keep the plain
    # white panel. A faint scrim keeps coloured annotations legible over dark map ink.
    image_svg = _plot_image_svg(overlay, project)
    if image_svg:
        lines.append(image_svg)
        lines.append(_plot_scrim_svg(project))
    lines.extend(_grid_svg(bounds, project, fill_panel=not image_svg))

    for item in overlay.items:
        if item.kind in {"query_bounds", "provider_bounds", "result_bbox"} and item.bounds:
            lines.append(_bounds_item_svg(item, project))
        elif item.kind == "result_point" and item.lon is not None and item.lat is not None:
            lines.append(_point_item_svg(item, project))

    lines.append("</svg>")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def render_knowledge_overlay_on_image(
    overlay: KnowledgeOverlay,
    georef: Any,
    image_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Annotate the input map raster with the queried knowledge, in pixel space.

    The counterpart to :func:`render_knowledge_overlay_svg`: instead of plotting on
    a synthetic geographic canvas, each result point/bbox is projected back onto the
    original image via ``georef.lonlat_to_pixel`` and drawn over it (markers, boxes,
    a provider legend) -- so the lookup can be reviewed against the actual map.

    ``georef`` is duck-typed: any object exposing
    ``lonlat_to_pixel(lon, lat) -> (pixel_x, pixel_y)`` (e.g. a
    ``peace_tool_pool.georef.GeoReference``). Markers that land outside the image
    are skipped by the drawing primitive, mirroring the overlay bounds-guard. The
    cv2 drawing toolkit is imported lazily so importing this module stays light.
    """
    from ..map_processing.image_ops import annotate_points_on_image

    markers: list[tuple[float, float, Sequence[int]]] = []
    boxes: list[tuple[tuple[int, int, int, int], Sequence[int]]] = []
    plotted_by_provider: dict[str | None, int] = {}

    for item in overlay.items:
        if item.kind == "result_point" and item.lon is not None and item.lat is not None:
            pixel_x, pixel_y = georef.lonlat_to_pixel(item.lon, item.lat)
            markers.append((pixel_x, pixel_y, item.color_rgb))
            plotted_by_provider[item.provider] = plotted_by_provider.get(item.provider, 0) + 1
        elif item.kind == "result_bbox" and item.bounds is not None:
            box = item.bounds
            x0, y0 = georef.lonlat_to_pixel(box.min_lon, box.max_lat)
            x1, y1 = georef.lonlat_to_pixel(box.max_lon, box.min_lat)
            boxes.append(((round(x0), round(y0), round(x1), round(y1)), item.color_rgb))
            plotted_by_provider[item.provider] = plotted_by_provider.get(item.provider, 0) + 1

    legend = [
        (f"{provider or 'unknown'} ({count})", _color_for_provider(provider))
        for provider, count in sorted(plotted_by_provider.items(), key=lambda kv: kv[0] or "")
    ]
    annotate_points_on_image(
        image_path, markers, output_path, boxes=boxes, legend=legend, title=title
    )
    return Path(output_path)


def _bounds_parts_from_trace(trace: Mapping[str, Any] | None) -> list[Bounds]:
    if not isinstance(trace, Mapping):
        return []
    return _bounds_parts_from_any(trace.get("bounds_parts"))


def _bounds_parts_from_any(raw: Any) -> list[Bounds]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    bounds_parts: list[Bounds] = []
    for value in values:
        bounds = _bounds_from_any(value)
        if bounds is not None:
            bounds_parts.append(bounds)
    return bounds_parts


def _bounds_from_any(value: Any) -> Bounds | None:
    if isinstance(value, Bounds):
        return value
    try:
        if isinstance(value, Mapping):
            return Bounds(
                min_lon=value["min_lon"],
                min_lat=value["min_lat"],
                max_lon=value["max_lon"],
                max_lat=value["max_lat"],
                crs=str(value.get("crs", "EPSG:4326")),
            )
        if _is_sequence(value) and len(value) >= 4:
            return Bounds(
                min_lon=value[0],
                min_lat=value[1],
                max_lon=value[2],
                max_lat=value[3],
            )
    except Exception:  # noqa: BLE001 - malformed optional geometry is skipped.
        return None
    return None


def _provider_bounds_items(item: KnowledgeItem) -> list[KnowledgeOverlayItem]:
    bounds_parts = _bounds_parts_from_any(item.provenance.get("bounds_parts"))
    color = _color_for_provider(item.provider)
    return [
        KnowledgeOverlayItem(
            id=f"provider_bounds:{item.provider}:{index}",
            kind="provider_bounds",
            label=f"{item.provider} lookup {index + 1}",
            provider=item.provider,
            key=item.key,
            bounds=bounds,
            color_rgb=color,
            source="item.provenance.bounds_parts",
        )
        for index, bounds in enumerate(bounds_parts)
    ]


def _result_geometry_items(item: KnowledgeItem) -> list[KnowledgeOverlayItem]:
    if not isinstance(item.value, list):
        return []
    overlay_items: list[KnowledgeOverlayItem] = []
    color = _color_for_provider(item.provider)
    for index, record in enumerate(item.value):
        if not isinstance(record, Mapping):
            continue
        bounds = _bounds_from_any(record.get("geometry_bbox"))
        if bounds is not None:
            overlay_items.append(
                KnowledgeOverlayItem(
                    id=f"result_bbox:{item.provider}:{index}",
                    kind="result_bbox",
                    label=_record_label(record, item.provider),
                    provider=item.provider,
                    key=item.key,
                    bounds=bounds,
                    color_rgb=color,
                    source="item.value.geometry_bbox",
                    record_index=index,
                )
            )
            continue
        point = _point_from_record(record)
        if point is not None:
            lon, lat = point
            overlay_items.append(
                KnowledgeOverlayItem(
                    id=f"result_point:{item.provider}:{index}",
                    kind="result_point",
                    label=_record_label(record, item.provider),
                    provider=item.provider,
                    key=item.key,
                    lon=lon,
                    lat=lat,
                    color_rgb=color,
                    source="item.value.longitude_latitude",
                    record_index=index,
                )
            )
    return overlay_items


def _point_from_record(record: Mapping[str, Any]) -> tuple[float, float] | None:
    lon_value = record.get("longitude", record.get("lon"))
    lat_value = record.get("latitude", record.get("lat"))
    try:
        lon = float(lon_value)
        lat = float(lat_value)
    except (TypeError, ValueError):
        return None
    if not -180 <= lon <= 180 or not -90 <= lat <= 90:
        return None
    return lon, lat


def _record_label(record: Mapping[str, Any], fallback: str) -> str:
    for key in ("name", "place", "label", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def _bounds_for_items(items: Sequence[KnowledgeOverlayItem]) -> Bounds | None:
    lons: list[float] = []
    lats: list[float] = []
    for item in items:
        if item.bounds is not None:
            lons.extend([item.bounds.min_lon, item.bounds.max_lon])
            lats.extend([item.bounds.min_lat, item.bounds.max_lat])
        elif item.lon is not None and item.lat is not None:
            lons.append(item.lon)
            lats.append(item.lat)
    if not lons or not lats:
        return None
    min_lon = max(-180.0, min(lons))
    max_lon = min(180.0, max(lons))
    min_lat = max(-90.0, min(lats))
    max_lat = min(90.0, max(lats))
    if min_lon == max_lon:
        min_lon = max(-180.0, min_lon - 0.01)
        max_lon = min(180.0, max_lon + 0.01)
    if min_lat == max_lat:
        min_lat = max(-90.0, min_lat - 0.01)
        max_lat = min(90.0, max_lat + 0.01)
    return Bounds(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


def _bounds_union(parts: Sequence[Bounds]) -> Bounds | None:
    if not parts:
        return None
    return Bounds(
        min_lon=min(part.min_lon for part in parts),
        min_lat=min(part.min_lat for part in parts),
        max_lon=max(part.max_lon for part in parts),
        max_lat=max(part.max_lat for part in parts),
    )


def _bounds_contains_point(bounds: Bounds, lon: float, lat: float) -> bool:
    return (
        bounds.min_lon <= lon <= bounds.max_lon
        and bounds.min_lat <= lat <= bounds.max_lat
    )


def _bounds_intersects(a: Bounds, b: Bounds) -> bool:
    return not (
        a.max_lon < b.min_lon
        or a.min_lon > b.max_lon
        or a.max_lat < b.min_lat
        or a.min_lat > b.max_lat
    )


def _result_within_parts(item: KnowledgeOverlayItem, parts: Sequence[Bounds]) -> bool:
    """Whether a result geometry belongs on a map covering ``parts``.

    Points must be contained; result bounding boxes need only intersect a part so a
    feature straddling the map edge (e.g. a fault crossing the boundary) is retained.
    Membership is tested per part rather than against a merged box, which keeps
    antimeridian-split queries from accepting points in the unqueried middle.
    """

    if item.bounds is not None:
        return any(_bounds_intersects(part, item.bounds) for part in parts)
    if item.lon is not None and item.lat is not None:
        return any(_bounds_contains_point(part, item.lon, item.lat) for part in parts)
    return True


def _out_of_bounds_warnings(
    dropped: Sequence[KnowledgeOverlayItem],
    target: Bounds | None,
) -> list[str]:
    if not dropped or target is None:
        return []
    providers = sorted({item.provider or "unknown" for item in dropped})
    return [
        f"{len(dropped)} knowledge annotation(s) fell outside the target map bounds "
        f"[lon {target.min_lon:.4f}..{target.max_lon:.4f}, "
        f"lat {target.min_lat:.4f}..{target.max_lat:.4f}] and were hidden "
        f"(providers: {', '.join(providers)}). "
        "This usually indicates a coordinate or CRS misalignment in the lookup."
    ]


def _plot_image_svg(overlay: KnowledgeOverlay, project: Any) -> str:
    """SVG ``<image>`` of the input map filling the plot box, or "" when absent."""
    frame = overlay.frame
    if not frame.image_path:
        return ""
    data_uri = _load_map_image_data_uri(frame.image_path, frame.georef)
    if data_uri is None:
        return ""
    x, y, width, height = project.plot_box
    return (
        f'<image x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'href="{data_uri}" preserveAspectRatio="none"/>'
    )


def _plot_scrim_svg(project: Any) -> str:
    x, y, width, height = project.plot_box
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#ffffff" fill-opacity="0.12"/>'
    )


def _load_map_image_data_uri(
    image_path: str,
    georef: Mapping[str, Any] | None,
    *,
    max_dim: int = 1600,
) -> str | None:
    """Crop the input map to its georef pixel_extent and return a PNG data URI.

    Returns ``None`` (so the renderer falls back to a blank panel) when Pillow is
    unavailable or the image cannot be read -- a missing backdrop must never break
    the overlay. Cropping to ``pixel_extent`` aligns the main-map area with the
    frame bounds; the result is downscaled to keep the embedded SVG small.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            crop = _pixel_extent(georef, image.size)
            if crop is not None:
                image = image.crop(crop)
            longest = max(image.size)
            if longest > max_dim:
                scale = max_dim / longest
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                )
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
    except Exception:  # noqa: BLE001 - a malformed image degrades to no backdrop.
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pixel_extent(
    georef: Mapping[str, Any] | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(georef, Mapping):
        return None
    raw = georef.get("pixel_extent")
    if not (_is_sequence(raw) and len(raw) >= 4):
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(value))) for value in raw[:4])
    except (TypeError, ValueError):
        return None
    width, height = image_size
    x0, x1 = sorted((max(0, x0), min(width, x1)))
    y0, y1 = sorted((max(0, y0), min(height, y1)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1, y1)


def _metadata_image_path(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("image_path") or metadata.get("image")
    return str(value) if value not in (None, "") else None


def _metadata_georef(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("georef")
    return dict(value) if isinstance(value, Mapping) else None


def _color_for_provider(provider: str | None) -> tuple[int, int, int]:
    if provider is None:
        return KNOWLEDGE_OVERLAY_COLORS_RGB["default"]
    return KNOWLEDGE_OVERLAY_COLORS_RGB.get(
        provider,
        KNOWLEDGE_OVERLAY_COLORS_RGB["default"],
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _rgb_to_hex(rgb: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _projector(bounds: Bounds, width: int, height: int):
    left = 90.0
    right = max(left + 1.0, width - 30.0)
    top = 80.0
    bottom = max(top + 1.0, height - 70.0)
    plot_width = right - left
    plot_height = bottom - top
    mid_lat = (bounds.min_lat + bounds.max_lat) / 2.0
    lon_scale = max(math.cos(math.radians(mid_lat)), 0.05)
    data_width = max((bounds.max_lon - bounds.min_lon) * lon_scale, 1e-9)
    data_height = max(bounds.max_lat - bounds.min_lat, 1e-9)
    scale = min(plot_width / data_width, plot_height / data_height)
    drawn_width = data_width * scale
    drawn_height = data_height * scale
    x_offset = left + (plot_width - drawn_width) / 2.0
    y_offset = top + (plot_height - drawn_height) / 2.0

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = x_offset + ((lon - bounds.min_lon) * lon_scale) * scale
        y = y_offset + (bounds.max_lat - lat) * scale
        return x, y

    project.plot_box = (x_offset, y_offset, drawn_width, drawn_height)  # type: ignore[attr-defined]
    return project


def _grid_svg(bounds: Bounds, project: Any, *, fill_panel: bool = True) -> list[str]:
    x, y, width, height = project.plot_box
    panel_fill = "#ffffff" if fill_panel else "none"
    lines = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{panel_fill}" stroke="#cbd5e1" stroke-width="1"/>'
    ]
    for lon in _ticks(bounds.min_lon, bounds.max_lon, 4):
        x0, y0 = project(lon, bounds.min_lat)
        x1, y1 = project(lon, bounds.max_lat)
        lines.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(f'<text x="{x0 - 24:.2f}" y="{y + height + 20:.2f}" class="axis">{lon:.2f}</text>')
    for lat in _ticks(bounds.min_lat, bounds.max_lat, 4):
        x0, y0 = project(bounds.min_lon, lat)
        x1, y1 = project(bounds.max_lon, lat)
        lines.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(f'<text x="{max(4.0, x - 62):.2f}" y="{y0 + 4:.2f}" class="axis">{lat:.2f}</text>')
    lines.append(f'<text x="{x:.2f}" y="{y + height + 44:.2f}" class="axis">longitude</text>')
    lines.append(f'<text x="{max(4.0, x - 70):.2f}" y="{y - 12:.2f}" class="axis">latitude</text>')
    return lines


def _ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 0 or min_value == max_value:
        return [min_value]
    step = (max_value - min_value) / count
    return [min_value + step * index for index in range(count + 1)]


def _bounds_item_svg(item: KnowledgeOverlayItem, project: Any) -> str:
    assert item.bounds is not None
    x0, y0 = project(item.bounds.min_lon, item.bounds.max_lat)
    x1, y1 = project(item.bounds.max_lon, item.bounds.min_lat)
    x = min(x0, x1)
    y = min(y0, y1)
    width = max(1.0, abs(x1 - x0))
    height = max(1.0, abs(y1 - y0))
    color = _rgb_to_hex(item.color_rgb)
    stroke_width = 3 if item.kind == "query_bounds" else 2
    dash = ' stroke-dasharray="6 4"' if item.kind == "provider_bounds" else ""
    fill_opacity = "0.08" if item.kind == "result_bbox" else "0"
    return (
        f'<g data-id="{_escape(item.id)}" data-kind="{_escape(item.kind)}"'
        f'{_provider_attr(item)}>'
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" '
        f'stroke-width="{stroke_width}"{dash}/>'
        f'<text x="{x + 5:.2f}" y="{max(14.0, y - 6):.2f}" class="label" fill="{color}">'
        f'{_escape(item.label)}</text>'
        "</g>"
    )


def _point_item_svg(item: KnowledgeOverlayItem, project: Any) -> str:
    assert item.lon is not None and item.lat is not None
    x, y = project(item.lon, item.lat)
    color = _rgb_to_hex(item.color_rgb)
    return (
        f'<g data-id="{_escape(item.id)}" data-kind="{_escape(item.kind)}"'
        f'{_provider_attr(item)}>'
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="#ffffff" '
        'stroke-width="1.5"/>'
        f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" class="label" fill="{color}">'
        f'{_escape(item.label)}</text>'
        "</g>"
    )


def _provider_attr(item: KnowledgeOverlayItem) -> str:
    if item.provider is None:
        return ""
    return f' data-provider="{_escape(item.provider)}"'


__all__ = [
    "KNOWLEDGE_OVERLAY_COLORS_RGB",
    "KnowledgeOverlay",
    "KnowledgeOverlayFrame",
    "KnowledgeOverlayItem",
    "render_knowledge_overlay_on_image",
    "extract_knowledge_overlay",
    "render_knowledge_overlay_svg",
]

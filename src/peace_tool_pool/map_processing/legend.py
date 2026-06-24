"""Legend unit pairing and metadata extraction."""

from __future__ import annotations

from typing import Any, Mapping

from . import image_ops
from .types import BBox, LegendEntry, normalize_bbox


def distance(color_bbox: BBox, text_bbox: BBox) -> float:
    c_x0, c_y0, c_x1, c_y1 = color_bbox
    t_x0, t_y0, _t_x1, t_y1 = text_bbox
    color_x = c_x1
    color_y = (c_y0 + c_y1) / 2
    text_x = t_x0
    text_y = (t_y0 + t_y1) / 2
    return ((color_x - text_x) ** 2 + (color_y - text_y) ** 2) ** 0.5


def shrink_bbox(image: Any, bbox: BBox) -> BBox:
    cv2 = image_ops.require_cv2()
    x0, y0, x1, y1 = image_ops.clip_bbox(bbox, image.shape[1], image.shape[0])
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        return x0, y0, x1, y1
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    column_min = gray.min(axis=0)
    threshold = 32
    width = column_min.shape[0]

    dx0 = int(width * 0.01)
    while dx0 < width and column_min[dx0] >= 256 - threshold:
        dx0 += 1

    dx1 = int(width * 0.99) - 1
    while dx1 > dx0 and column_min[dx1] >= 256 - threshold:
        dx1 -= 1

    if dx0 >= width or dx1 <= dx0:
        return x0, y0, x1, y1

    return image_ops.clip_bbox((x0 + dx0 - 1, y0, x0 + dx1 + 1, y1), image.shape[1], image.shape[0])


def pair_legend_boxes(
    color_bboxes: list[BBox],
    text_bboxes: list[BBox],
    image: Any,
) -> list[tuple[BBox, BBox]]:
    remaining_text = list(text_bboxes)
    pairs: list[tuple[BBox, BBox]] = []
    for color_bbox in color_bboxes:
        threshold = color_bbox[3] - color_bbox[1]
        paired_text_bbox: BBox | None = None
        min_distance = float("inf")
        for text_bbox in remaining_text:
            current_distance = distance(color_bbox, text_bbox)
            if current_distance < min_distance:
                min_distance = current_distance
                paired_text_bbox = text_bbox
        if paired_text_bbox is None or min_distance > threshold:
            continue
        remaining_text.remove(paired_text_bbox)
        pairs.append((color_bbox, shrink_bbox(image, paired_text_bbox)))
    return pairs


def _extract_bboxes(detections: Mapping[str, Any], key: str) -> list[BBox]:
    bboxes: list[BBox] = []
    for value in detections.get(key, []):
        if hasattr(value, "bbox"):
            bboxes.append(normalize_bbox(value.bbox))
        else:
            bboxes.append(normalize_bbox(value))
    return bboxes


def _offset_bbox(bbox: BBox, parent_bbox: BBox) -> BBox:
    x0, y0, x1, y1 = bbox
    parent_x0, parent_y0, _parent_x1, _parent_y1 = parent_bbox
    return x0 + parent_x0, y0 + parent_y0, x1 + parent_x0, y1 + parent_y0


def build_legend_entries(
    legend_image: Any,
    detections: Mapping[str, Any],
    legend_bbox: BBox,
) -> list[LegendEntry]:
    color_bboxes = _extract_bboxes(detections, "color_bndbox")
    text_bboxes = _extract_bboxes(detections, "text_bndbox")
    pairs = pair_legend_boxes(color_bboxes, text_bboxes, legend_image)
    entries: list[LegendEntry] = []
    for index, (color_bbox, text_bbox) in enumerate(pairs):
        color_image = image_ops.crop_image(legend_image, color_bbox)
        color_rgb = image_ops.calc_image_rgb(color_image)
        entries.append(
            LegendEntry(
                id=index,
                color_bbox=_offset_bbox(color_bbox, legend_bbox),
                text_bbox=_offset_bbox(text_bbox, legend_bbox),
                color_rgb=color_rgb,
                color_hex=image_ops.rgb_to_hex(color_rgb),
                color_name=image_ops.rgb_to_color_name(color_rgb),
            )
        )
    return entries

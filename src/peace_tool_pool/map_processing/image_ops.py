"""Image operations ported from the PEACE HIE pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import OptionalDependencyError
from .types import BBox, LegendEntry, normalize_bbox


DETECTION_BOX_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "title": (37, 99, 235),
    "main_map": (22, 163, 74),
    "legend": (245, 158, 11),
    "scale": (147, 51, 234),
    "index_map": (8, 145, 178),
    "cross_section": (220, 38, 38),
    "stratigraphic_column": (219, 39, 119),
    "others": (71, 85, 105),
}


def require_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "OpenCV is required for map image processing. Install detector dependencies with "
            "`uv sync --extra detectors` or run with equivalent cv2/numpy packages."
        ) from exc
    return cv2


def require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "NumPy is required for map image processing. Install detector dependencies with "
            "`uv sync --extra detectors` or run with equivalent cv2/numpy packages."
        ) from exc
    return np


def read_image(path: str | Path) -> Any:
    cv2 = require_cv2()
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def image_size(image: Any) -> tuple[int, int]:
    if isinstance(image, (str, Path)):
        image = read_image(image)
    height, width = image.shape[:2]
    return width, height


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = normalize_bbox(bbox)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Invalid bbox {bbox!r} for image size {(width, height)!r}")
    return x0, y0, x1, y1


def _safe_clip_bbox(bbox: BBox, width: int, height: int) -> BBox | None:
    try:
        return clip_bbox(normalize_bbox(bbox), width, height)
    except ValueError:
        return None


def crop_image(image: Any, bbox: BBox) -> Any:
    if isinstance(image, (str, Path)):
        image = read_image(image)
    height, width = image.shape[:2]
    x0, y0, x1, y1 = clip_bbox(bbox, width, height)
    return image[y0:y1, x0:x1]


def save_image(path: str | Path, image: Any) -> None:
    cv2 = require_cv2()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise ValueError(f"Unable to write image: {target}")


def crop_and_save_image(image: Any, bbox: BBox, cropped_image_path: str | Path) -> None:
    save_image(cropped_image_path, crop_image(image, bbox))


def _rgb_to_bgr(color: Sequence[int]) -> tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def _text_color_for_background(rgb_color: Sequence[int]) -> tuple[int, int, int]:
    luminance = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def _blend_rectangle(cv2: Any, image: Any, bbox: BBox, color: Sequence[int], alpha: float) -> None:
    x0, y0, x1, y1 = bbox
    if x0 >= x1 or y0 >= y1:
        return
    roi = image[y0:y1, x0:x1]
    fill = roi.copy()
    fill[:] = color
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, dst=roi)


def annotate_detections_on_image(
    image: Any,
    detections_by_label: Mapping[str, Sequence[Any]],
    output_path: str | Path,
    *,
    legend_entries: Sequence[LegendEntry] = (),
) -> None:
    if isinstance(image, (str, Path)):
        image = read_image(image)
    cv2 = require_cv2()
    height, width = image.shape[:2]
    annotated = image.copy()
    thickness = max(2, round(min(width, height) / 500))
    font_scale = max(0.45, min(width, height) / 1400)
    font_thickness = max(1, thickness - 1)
    padding = max(3, thickness + 1)

    for label, detections in detections_by_label.items():
        rgb_color = DETECTION_BOX_COLORS_RGB.get(label, DETECTION_BOX_COLORS_RGB["others"])
        box_color = _rgb_to_bgr(rgb_color)
        text_color = _text_color_for_background(rgb_color)
        for detection in detections:
            bbox = getattr(detection, "bbox", detection)
            confidence = getattr(detection, "confidence", None)
            x0, y0, x1, y1 = clip_bbox(normalize_bbox(bbox), width, height)
            cv2.rectangle(
                annotated,
                (x0, y0),
                (x1, y1),
                box_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            caption = label.replace("_", " ")
            if confidence is not None:
                caption = f"{caption} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                caption,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
            label_width = min(width, text_width + padding * 2)
            label_height = text_height + baseline + padding * 2
            label_x0 = min(x0, max(0, width - label_width))
            label_x1 = min(width, label_x0 + label_width)
            if y0 - label_height >= 0:
                label_y0 = y0 - label_height
                label_y1 = y0
                text_y = y0 - baseline - padding
            else:
                label_y0 = y0
                label_y1 = min(height, y0 + label_height)
                text_y = min(height - padding - baseline, y0 + padding + text_height)

            cv2.rectangle(
                annotated,
                (label_x0, label_y0),
                (label_x1, label_y1),
                box_color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                caption,
                (label_x0 + padding, max(text_height, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

    legend_list = list(legend_entries)
    legend_box_heights: list[int] = []
    for entry in legend_list:
        for bbox in (entry.color_bbox, entry.text_bbox):
            clipped = _safe_clip_bbox(bbox, width, height)
            if clipped is None:
                continue
            legend_box_heights.append(clipped[3] - clipped[1])

    standard_legend_height = 12
    if legend_box_heights:
        standard_legend_height = sorted(legend_box_heights)[len(legend_box_heights) // 2]

    legend_thickness = 1
    (_base_text_width, base_text_height), base_baseline = cv2.getTextSize(
        "Ag",
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        1,
    )
    target_text_height = max(5, min(12, round(standard_legend_height * 0.45)))
    legend_font_scale = max(0.18, min(0.55, target_text_height / (base_text_height + base_baseline)))
    legend_font_thickness = 1
    legend_padding = max(1, min(3, round(standard_legend_height * 0.08)))
    for entry in legend_list:
        swatch_color = _rgb_to_bgr(entry.color_rgb)
        color_bbox = _safe_clip_bbox(entry.color_bbox, width, height)
        if color_bbox is not None:
            x0, y0, x1, y1 = color_bbox
            cv2.rectangle(
                annotated,
                (x0, y0),
                (x1, y1),
                swatch_color,
                thickness=legend_thickness,
                lineType=cv2.LINE_AA,
            )

        text_bbox = _safe_clip_bbox(entry.text_bbox, width, height)
        if text_bbox is not None:
            x0, y0, x1, y1 = text_bbox
            cv2.rectangle(
                annotated,
                (x0, y0),
                (x1, y1),
                swatch_color,
                thickness=legend_thickness,
                lineType=cv2.LINE_AA,
            )
            caption = f"#{entry.id} {entry.area_fraction:.1%}"
            (text_width, text_height), baseline = cv2.getTextSize(
                caption,
                cv2.FONT_HERSHEY_SIMPLEX,
                legend_font_scale,
                legend_font_thickness,
            )
            chip_size = max(5, min(12, text_height + baseline))
            chip_size = max(4, min(chip_size, round(standard_legend_height * 0.55)))
            label_width = min(width, text_width + chip_size + legend_padding * 4)
            label_height = max(text_height + baseline, chip_size) + legend_padding * 2
            label_x0 = min(x0, max(0, width - label_width))
            label_x1 = min(width, label_x0 + label_width)
            if y0 - label_height >= 0:
                label_y0 = y0 - label_height
            else:
                label_y0 = min(max(0, y1), max(0, height - label_height))
            label_y1 = min(height, label_y0 + label_height)
            _blend_rectangle(
                cv2,
                annotated,
                (label_x0, label_y0, label_x1, label_y1),
                _rgb_to_bgr((248, 250, 252)),
                alpha=0.7,
            )
            cv2.rectangle(
                annotated,
                (label_x0, label_y0),
                (label_x1, label_y1),
                _rgb_to_bgr((71, 85, 105)),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            chip_x0 = label_x0 + legend_padding
            chip_y0 = label_y0 + max(0, (label_height - chip_size) // 2)
            chip_x1 = min(label_x1 - legend_padding, chip_x0 + chip_size)
            chip_y1 = min(label_y1 - legend_padding, chip_y0 + chip_size)
            cv2.rectangle(
                annotated,
                (chip_x0, chip_y0),
                (chip_x1, chip_y1),
                swatch_color,
                thickness=-1,
            )
            cv2.rectangle(
                annotated,
                (chip_x0, chip_y0),
                (chip_x1, chip_y1),
                _rgb_to_bgr((15, 23, 42)),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            text_x = chip_x1 + legend_padding
            text_y = min(label_y1 - legend_padding - baseline, label_y0 + legend_padding + text_height)
            cv2.putText(
                annotated,
                caption,
                (text_x, max(text_height, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                legend_font_scale,
                _rgb_to_bgr((15, 23, 42)),
                thickness=legend_font_thickness,
                lineType=cv2.LINE_AA,
            )
    save_image(output_path, annotated)


def crop_corners_and_save_image(
    image: Any,
    cropped_image_path: str | Path,
    relative_size: float = 0.1,
) -> None:
    if isinstance(image, (str, Path)):
        image = read_image(image)
    np = require_numpy()
    height, width = image.shape[:2]
    corner_height = max(1, int(height * relative_size))
    corner_width = max(1, int(width * relative_size))
    top_left = image[0:corner_height, 0:corner_width]
    top_right = image[0:corner_height, width - corner_width : width]
    bottom_left = image[height - corner_height : height, 0:corner_width]
    bottom_right = image[height - corner_height : height, width - corner_width : width]
    top_combined = np.hstack((top_left, top_right))
    bottom_combined = np.hstack((bottom_left, bottom_right))
    save_image(cropped_image_path, np.vstack((top_combined, bottom_combined)))


def annotate_image_with_directions(
    image: Any,
    output_path: str | Path,
    font_scale: float = 0.8,
    offset: int = 50,
) -> None:
    if isinstance(image, (str, Path)):
        image = read_image(image)
    cv2 = require_cv2()
    height, width = image.shape[:2]
    annotated = cv2.copyMakeBorder(
        image,
        offset,
        offset,
        offset,
        offset,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    positions = {
        "N": (offset + width // 2, offset // 2),
        "S": (offset + width // 2, offset + height + offset // 2),
        "W": (offset // 3, offset + height // 2),
        "E": (offset + width + offset // 3, offset + height // 2),
        "NE": (offset + width + offset // 4, offset // 2),
        "NW": (offset // 4, offset // 2),
        "SE": (offset + width + offset // 4, offset + height + offset // 2),
        "SW": (offset // 4, offset + height + offset // 2),
    }
    for label, position in positions.items():
        cv2.putText(
            annotated,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    save_image(output_path, annotated)


def calc_image_rgb(image: Any) -> tuple[int, int, int]:
    np = require_numpy()
    pixels = image.reshape(-1, 3)
    not_black = ~((pixels[:, 0] < 16) & (pixels[:, 1] < 16) & (pixels[:, 2] < 16))
    pixel_list = pixels[not_black]
    if pixel_list.size == 0:
        pixel_list = pixels
    color_bgr = np.median(pixel_list, axis=0)
    return tuple(int(round(value)) for value in color_bgr[::-1])


def rgb_to_hex(rgb_color: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))


def rgb_to_color_name(rgb: Sequence[int]) -> str:
    color_map = {
        (255, 0, 0): "Red",
        (0, 255, 0): "Green",
        (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow",
        (0, 255, 255): "Cyan",
        (255, 0, 255): "Magenta",
        (255, 255, 255): "White",
        (0, 0, 0): "Black",
        (128, 128, 128): "Gray",
        (128, 0, 0): "Maroon",
        (128, 128, 0): "Olive",
        (0, 128, 0): "Dark Green",
        (128, 0, 128): "Purple",
        (0, 128, 128): "Teal",
        (0, 0, 128): "Navy",
        (255, 192, 203): "Pink",
        (255, 165, 0): "Orange",
        (0, 255, 127): "Spring Green",
        (255, 105, 180): "Hot Pink",
        (255, 69, 0): "Red-Orange",
        (102, 205, 170): "Medium Aquamarine",
        (173, 216, 230): "Light Blue",
        (240, 230, 140): "Khaki",
        (255, 20, 147): "Deep Pink",
        (255, 99, 71): "Tomato",
    }
    closest = min(color_map, key=lambda key: sum((a - b) ** 2 for a, b in zip(rgb, key)))
    return color_map[closest]


def _color_key(color: Sequence[int]) -> str:
    return f"{int(color[0])}_{int(color[1])}_{int(color[2])}"


def calculate_color_thresholds(colors: Iterable[Sequence[int]]) -> dict[str, float]:
    color_list = [tuple(int(value) for value in color) for color in colors]
    color_to_threshold: dict[str, float] = {}
    for color1 in color_list:
        color_to_threshold[_color_key(color1)] = 10
        min_distance = 256 * 3
        for color2 in color_list:
            if _color_key(color1) == _color_key(color2):
                continue
            distance = sum(abs(a - b) for a, b in zip(color1, color2))
            if distance < min_distance:
                min_distance = distance
                color_to_threshold[_color_key(color1)] = min_distance / 2
    return color_to_threshold


def estimate_legend_area_fractions(image: Any, legends: Sequence[LegendEntry]) -> None:
    if not legends:
        return
    if isinstance(image, (str, Path)):
        image = read_image(image)
    cv2 = require_cv2()
    np = require_numpy()
    scale = 4
    height, width = image.shape[:2]
    resized = cv2.resize(image, (max(1, width // scale), max(1, height // scale)), cv2.INTER_NEAREST)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.int16)
    image_area = float(resized_rgb.shape[0] * resized_rgb.shape[1])
    thresholds = calculate_color_thresholds([legend.color_rgb for legend in legends])
    for legend in legends:
        legend.color_hex = rgb_to_hex(legend.color_rgb)
        if list(legend.color_rgb) == [255, 255, 255]:
            legend.area_fraction = 0.0
            continue
        color = np.array(legend.color_rgb, dtype=np.int16)
        distances = np.sum(np.abs(resized_rgb - color), axis=-1)
        threshold = thresholds[_color_key(legend.color_rgb)]
        legend.area_fraction = round(int(np.sum(distances <= threshold)) / image_area, 6)

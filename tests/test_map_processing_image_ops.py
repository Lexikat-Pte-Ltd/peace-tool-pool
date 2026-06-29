import pytest

from peace_tool_pool.map_processing import image_ops
from peace_tool_pool.map_processing.types import Detection, LegendEntry

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_annotate_detections_empty_legend_entries_is_noop(tmp_path):
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    detections = {
        "main_map": [Detection(label="main_map", bbox=(5, 5, 80, 60), confidence=0.9)]
    }
    baseline_path = tmp_path / "baseline.png"
    empty_legend_path = tmp_path / "empty_legend.png"

    image_ops.annotate_detections_on_image(image, detections, baseline_path)
    image_ops.annotate_detections_on_image(
        image,
        detections,
        empty_legend_path,
        legend_entries=[],
    )

    baseline = cv2.imread(str(baseline_path))
    empty_legend = cv2.imread(str(empty_legend_path))
    assert baseline is not None
    assert empty_legend is not None
    assert np.array_equal(empty_legend, baseline)


def test_annotate_detections_draws_legend_entries_with_inline_swatch_tags(tmp_path):
    image = np.full((140, 220, 3), 255, dtype=np.uint8)
    detections = {
        "main_map": [Detection(label="main_map", bbox=(5, 5, 95, 95), confidence=0.92)]
    }
    legend_entries = [
        LegendEntry(
            id=0,
            color_bbox=(120, 20, 135, 35),
            text_bbox=(142, 20, 185, 35),
            color_rgb=(255, 0, 0),
            color_hex="#FF0000",
            color_name="Red",
            area_fraction=0.125,
        ),
        LegendEntry(
            id=1,
            color_bbox=(120, 45, 135, 60),
            text_bbox=(142, 45, 185, 60),
            color_rgb=(0, 255, 0),
            color_hex="#00FF00",
            color_name="Green",
            area_fraction=0.25,
        ),
    ]
    output_path = tmp_path / "legend_overlay.png"

    image_ops.annotate_detections_on_image(
        image,
        detections,
        output_path,
        legend_entries=legend_entries,
    )

    overlay = cv2.imread(str(output_path))
    assert overlay is not None
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay[20, 120], image[20, 120])
    assert not np.array_equal(overlay[20, 142], image[20, 142])

    first_tag_pixels = overlay[3:20, 140:216].reshape(-1, 3)
    second_tag_pixels = overlay[28:45, 140:216].reshape(-1, 3)
    assert np.any(np.all(first_tag_pixels == (0, 0, 255), axis=1))
    assert np.any(np.all(second_tag_pixels == (0, 255, 0), axis=1))

    unused_panel_region = overlay[96:136, 128:216]
    source_region = image[96:136, 128:216]
    assert np.array_equal(unused_panel_region, source_region)


def test_legend_inline_tag_scales_to_box_height_and_blends_background(tmp_path):
    image = np.full((900, 900, 3), (30, 80, 140), dtype=np.uint8)
    legend_entries = [
        LegendEntry(
            id=0,
            color_bbox=(100, 100, 110, 110),
            text_bbox=(120, 100, 170, 110),
            color_rgb=(255, 0, 0),
            color_hex="#FF0000",
            color_name="Red",
            area_fraction=0.125,
        )
    ]
    output_path = tmp_path / "scaled_legend_overlay.png"

    image_ops.annotate_detections_on_image(
        image,
        {},
        output_path,
        legend_entries=legend_entries,
    )

    overlay = cv2.imread(str(output_path))
    assert overlay is not None
    tag_region = overlay[70:100, 120:220]
    source_region = image[70:100, 120:220]
    changed_rows = np.where(np.any(tag_region != source_region, axis=2))[0]
    assert changed_rows.size > 0
    assert int(changed_rows.min()) + 70 >= 88
    assert not np.any(np.all(tag_region.reshape(-1, 3) == (252, 250, 248), axis=1))


def test_annotate_points_draws_markers_boxes_and_legend(tmp_path):
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "knowledge_points.png"

    image_ops.annotate_points_on_image(
        image,
        [(150.0, 100.0, (22, 163, 74))],  # green mineral-occurrence marker
        output_path,
        boxes=[((40, 40, 90, 90), (220, 38, 38))],  # red fault box
        legend=[("mineral_occurrences (1)", (22, 163, 74))],
        title="knowledge overlay",
    )

    overlay = cv2.imread(str(output_path))
    assert overlay is not None
    assert overlay.shape == image.shape
    # the marker centre is no longer white
    assert not np.array_equal(overlay[100, 150], (255, 255, 255))
    # a pixel of the provider colour (RGB 22,163,74 -> BGR 74,163,22) is present at the marker
    near_marker = overlay[90:111, 140:161].reshape(-1, 3)
    assert np.any(np.all(near_marker == (74, 163, 22), axis=1))


def test_annotate_points_skips_out_of_bounds_markers(tmp_path):
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "oob.png"

    # Markers off the image are skipped (the bounds-guard philosophy) and never raise.
    image_ops.annotate_points_on_image(
        image,
        [(999.0, 999.0, (22, 163, 74)), (-5.0, 10.0, (1, 2, 3))],
        output_path,
    )

    overlay = cv2.imread(str(output_path))
    assert overlay is not None
    assert np.array_equal(overlay, image)


def test_annotate_detections_skips_invalid_legend_bboxes(tmp_path):
    image = np.full((60, 60, 3), 255, dtype=np.uint8)
    output_path = tmp_path / "invalid_legend_overlay.png"
    legend_entries = [
        LegendEntry(
            id=0,
            color_bbox=(80, 80, 90, 90),
            text_bbox=(10, 10, 10, 20),
            color_rgb=(255, 0, 0),
            color_hex="#FF0000",
            color_name="Red",
            area_fraction=0.0,
        )
    ]

    image_ops.annotate_detections_on_image(
        image,
        {},
        output_path,
        legend_entries=legend_entries,
    )

    overlay = cv2.imread(str(output_path))
    assert overlay is not None
    assert overlay.shape == image.shape

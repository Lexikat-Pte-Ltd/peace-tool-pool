import pytest

from peace_tool_pool.map_processing.legend import build_legend_entries

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_build_legend_entries_pairs_boxes_and_offsets_bboxes():
    image = np.full((30, 80, 3), 255, dtype=np.uint8)
    image[5:15, 4:14] = (0, 0, 255)  # BGR red swatch.
    image[5:15, 25:45] = 0  # Synthetic text ink for shrink logic.

    entries = build_legend_entries(
        legend_image=image,
        detections={
            "color_bndbox": [[4, 5, 14, 15]],
            "text_bndbox": [[20, 4, 50, 16]],
        },
        legend_bbox=(100, 200, 180, 230),
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.color_bbox == (104, 205, 114, 215)
    assert entry.text_bbox[0] >= 119
    assert entry.text_bbox[2] <= 150
    assert entry.color_rgb == (255, 0, 0)
    assert entry.color_hex == "#FF0000"
    assert entry.color_name == "Red"


def test_build_legend_entries_skips_unpaired_color_boxes():
    image = np.full((30, 80, 3), 255, dtype=np.uint8)
    image[5:15, 4:14] = (0, 0, 255)

    entries = build_legend_entries(
        legend_image=image,
        detections={"color_bndbox": [[4, 5, 14, 15]], "text_bndbox": [[60, 4, 75, 16]]},
        legend_bbox=(0, 0, 80, 30),
    )

    assert entries == []

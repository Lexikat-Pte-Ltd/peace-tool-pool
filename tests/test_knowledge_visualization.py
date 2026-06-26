from pathlib import Path

import pytest

from peace_tool_pool.knowledge import Bounds, KnowledgeBundle, KnowledgeItem
from peace_tool_pool.knowledge.visualization import (
    KNOWLEDGE_OVERLAY_COLORS_RGB,
    extract_knowledge_overlay,
    render_knowledge_overlay_svg,
)


def _bundle() -> KnowledgeBundle:
    return KnowledgeBundle(
        bounds=Bounds(min_lon=-122.5, min_lat=37.0, max_lon=-90.3, max_lat=48.8),
        items=[
            KnowledgeItem(
                id="active_faults:active_faults",
                key="active_faults",
                provider="active_faults",
                value=[
                    {
                        "name": "Alpha Fault",
                        "slip_type": "strike-slip",
                        "geometry_bbox": [-122.0, 37.0, -121.0, 38.0],
                    },
                    {"name": "Fault without geometry"},
                ],
                provenance={
                    "bounds_parts": [
                        {
                            "min_lon": -122.5,
                            "min_lat": 37.0,
                            "max_lon": -121.5,
                            "max_lat": 38.0,
                            "crs": "EPSG:4326",
                        }
                    ]
                },
            ),
            KnowledgeItem(
                id="earthquake_history:earthquake_history",
                key="earthquake_history",
                provider="earthquake_history",
                value=[
                    {
                        "place": "Newer in bounds",
                        "longitude": -121.5,
                        "latitude": 37.5,
                        "mag": 5.2,
                    }
                ],
            ),
            KnowledgeItem(
                id="mineral_occurrences:mineral_occurrences",
                key="mineral_occurrences",
                provider="mineral_occurrences",
                value=[
                    {
                        "name": "GRANDE PORTAGE",
                        "longitude": -90.6,
                        "latitude": 48.5,
                        "primary_commodity": "GOLD",
                    }
                ],
            ),
            KnowledgeItem(
                id="rock_type:granite",
                key="rock_type",
                provider="rock_type",
                value={"value": "igneous"},
            ),
        ],
        selected_item_ids=None,
        warnings=[],
        provider_versions={},
        trace={
            "bounds_parts": [
                {
                    "min_lon": -122.5,
                    "min_lat": 37.0,
                    "max_lon": -90.3,
                    "max_lat": 48.8,
                    "crs": "EPSG:4326",
                }
            ],
            "raw_extent": None,
        },
    )


def test_extract_overlay_dispatches_provider_geometry_shapes() -> None:
    overlay = extract_knowledge_overlay(_bundle())

    assert overlay.frame.source == "geographic_canvas"
    assert overlay.frame.crs == "EPSG:4326"
    assert len(overlay.frame.bounds_parts) == 1
    assert set(overlay.frame.item_ids) == {item.id for item in overlay.items}

    query_boxes = [item for item in overlay.items if item.kind == "query_bounds"]
    provider_boxes = [item for item in overlay.items if item.kind == "provider_bounds"]
    result_boxes = [item for item in overlay.items if item.kind == "result_bbox"]
    result_points = [item for item in overlay.items if item.kind == "result_point"]

    assert len(query_boxes) == 1
    assert query_boxes[0].bounds is not None
    assert query_boxes[0].bounds.min_lon == -122.5
    assert [(item.provider, item.bounds.min_lon) for item in provider_boxes if item.bounds] == [
        ("active_faults", -122.5)
    ]
    assert [(item.provider, item.label) for item in result_boxes] == [
        ("active_faults", "Alpha Fault")
    ]
    assert {(item.provider, item.lon, item.lat) for item in result_points} == {
        ("earthquake_history", -121.5, 37.5),
        ("mineral_occurrences", -90.6, 48.5),
    }
    assert not any(item.provider == "rock_type" for item in overlay.items)


def test_extract_overlay_handles_split_trace_dicts_and_empty_trace() -> None:
    bundle = KnowledgeBundle(
        bounds=None,
        items=[],
        selected_item_ids=None,
        warnings=[],
        provider_versions={},
        trace={
            "bounds_parts": [
                {"min_lon": 170, "min_lat": -10, "max_lon": 180, "max_lat": 10},
                {"min_lon": -180, "min_lat": -10, "max_lon": -170, "max_lat": 10},
            ]
        },
    )

    overlay = extract_knowledge_overlay(bundle)

    assert len(overlay.frame.bounds_parts) == 2
    assert [item.label for item in overlay.items] == ["query part 1", "query part 2"]

    empty = extract_knowledge_overlay(
        KnowledgeBundle(
            bounds=None,
            items=[],
            selected_item_ids=None,
            warnings=[],
            provider_versions={},
            trace=None,
        )
    )
    assert empty.items == []
    assert empty.frame.bounds is None


def test_overlay_palette_is_stable_rgb() -> None:
    overlay = extract_knowledge_overlay(_bundle())

    colors_by_provider = {item.provider: item.color_rgb for item in overlay.items if item.provider}

    assert colors_by_provider["active_faults"] == KNOWLEDGE_OVERLAY_COLORS_RGB["active_faults"]
    assert colors_by_provider["earthquake_history"] == KNOWLEDGE_OVERLAY_COLORS_RGB[
        "earthquake_history"
    ]
    assert all(len(item.color_rgb) == 3 for item in overlay.items)
    assert all(0 <= channel <= 255 for item in overlay.items for channel in item.color_rgb)


def _bundle_with_stray_results() -> KnowledgeBundle:
    """A bundle whose target bounds sit in the US Midwest, plus two annotations
    that have strayed far outside (the signature of a CRS/coordinate misalignment)."""

    return KnowledgeBundle(
        bounds=Bounds(min_lon=-100.0, min_lat=40.0, max_lon=-90.0, max_lat=50.0),
        items=[
            KnowledgeItem(
                id="earthquake_history:earthquake_history",
                key="earthquake_history",
                provider="earthquake_history",
                value=[
                    {"place": "Inside", "longitude": -95.0, "latitude": 45.0, "mag": 4.0},
                    {"place": "Strayed", "longitude": 0.0, "latitude": 0.0, "mag": 4.0},
                ],
            ),
            KnowledgeItem(
                id="active_faults:active_faults",
                key="active_faults",
                provider="active_faults",
                value=[
                    # Crosses the western edge of the target -> still relevant, kept.
                    {"name": "Crossing Fault", "geometry_bbox": [-101.0, 44.0, -98.0, 46.0]},
                    # Entirely outside -> hidden.
                    {"name": "Far Fault", "geometry_bbox": [10.0, 10.0, 12.0, 12.0]},
                ],
            ),
        ],
        selected_item_ids=None,
        warnings=[],
        provider_versions={},
        trace={
            "bounds_parts": [
                {
                    "min_lon": -100.0,
                    "min_lat": 40.0,
                    "max_lon": -90.0,
                    "max_lat": 50.0,
                    "crs": "EPSG:4326",
                }
            ],
            "raw_extent": None,
        },
    )


def test_extract_overlay_hides_out_of_bounds_results_and_warns() -> None:
    with pytest.warns(UserWarning, match="outside the target map bounds"):
        overlay = extract_knowledge_overlay(_bundle_with_stray_results())

    plotted = {
        item.label for item in overlay.items if item.kind in {"result_point", "result_bbox"}
    }
    assert "Inside" in plotted
    assert "Crossing Fault" in plotted  # intersects the target edge -> kept
    assert "Strayed" not in plotted
    assert "Far Fault" not in plotted

    dropped = {item.label for item in overlay.out_of_bounds}
    assert dropped == {"Strayed", "Far Fault"}
    assert overlay.warnings
    assert any("misalign" in message.lower() for message in overlay.warnings)


def test_extract_overlay_frame_anchors_to_target_not_strays() -> None:
    with pytest.warns(UserWarning):
        overlay = extract_knowledge_overlay(_bundle_with_stray_results())

    assert overlay.frame.bounds is not None
    # The strayed (0, 0) point must NOT expand the frame off the target region.
    assert overlay.frame.bounds.min_lon == -100.0
    assert overlay.frame.bounds.max_lon == -90.0
    assert overlay.frame.bounds.min_lat == 40.0
    assert overlay.frame.bounds.max_lat == 50.0


def test_extract_overlay_without_target_bounds_keeps_all_results() -> None:
    bundle = KnowledgeBundle(
        bounds=None,
        items=[
            KnowledgeItem(
                id="earthquake_history:earthquake_history",
                key="earthquake_history",
                provider="earthquake_history",
                value=[{"place": "Anywhere", "longitude": 0.0, "latitude": 0.0}],
            )
        ],
        selected_item_ids=None,
        warnings=[],
        provider_versions={},
        trace=None,
    )

    overlay = extract_knowledge_overlay(bundle)

    assert any(item.label == "Anywhere" for item in overlay.items)
    assert overlay.out_of_bounds == []
    assert overlay.warnings == []


def test_render_overlay_notes_hidden_out_of_bounds(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        overlay = extract_knowledge_overlay(_bundle_with_stray_results())
    output_path = tmp_path / "guarded_overlay.svg"

    render_knowledge_overlay_svg(overlay, output_path, title="Guarded overlay")

    text = output_path.read_text(encoding="utf-8")
    assert "Strayed" not in text
    assert "Far Fault" not in text
    assert "outside the target map bounds" in text


def test_render_overlay_svg_writes_visual_artifact(tmp_path: Path) -> None:
    overlay = extract_knowledge_overlay(_bundle())
    output_path = tmp_path / "knowledge_overlay.svg"

    render_knowledge_overlay_svg(
        overlay,
        output_path,
        width=640,
        height=480,
        title="Fixture knowledge overlay",
    )

    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("<svg ")
    assert "Fixture knowledge overlay" in text
    assert "data-kind=\"query_bounds\"" in text
    assert "data-kind=\"result_bbox\"" in text
    assert "data-kind=\"result_point\"" in text
    assert "Alpha Fault" in text
    assert "GRANDE PORTAGE" in text

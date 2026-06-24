from peace_tool_pool.map_processing.types import (
    ArtifactRef,
    Detection,
    ImageSize,
    LegendEntry,
    MapProcessingResult,
    MAP_PROCESSING_RESULT_SCHEMA,
)


def test_result_exports_peace_metadata_shape():
    result = MapProcessingResult(
        name="sample",
        source="usgs",
        image_path="/maps/sample.jpg",
        size=ImageSize(width=200, height=100),
        regions={
            "main_map": [Detection(label="main_map", bbox=(0, 0, 100, 80), confidence=0.9)],
            "legend": [Detection(label="legend", bbox=(120, 10, 190, 90))],
        },
        legend=[
            LegendEntry(
                id=0,
                color_bbox=(125, 20, 135, 30),
                text_bbox=(140, 20, 180, 30),
                color_rgb=(255, 0, 0),
                color_hex="#FF0000",
                color_name="Red",
                label="sandstone",
                area_fraction=0.25,
            )
        ],
        artifacts=[
            ArtifactRef(
                path=".cache/usgs/map_processing/det/sample/main_map_0.png",
                role="component_crop",
                stage="hie",
                bbox=(0, 0, 100, 80),
                label="main_map",
            )
        ],
    )

    as_dict = result.to_dict()
    assert as_dict["schema_version"] == "map-processing/v1"
    assert as_dict["legend"][0]["label"] == "sandstone"

    peace = result.to_peace_metadata()
    assert peace["name"] == "sample"
    assert peace["source"] == "usgs"
    assert peace["size"] == {"width": 200, "height": 100}
    assert peace["regions"]["main_map"] == [[0, 0, 100, 80]]
    assert peace["legend"][0]["color_bndbox"] == [125, 20, 135, 30]
    assert peace["legend"][0]["text"] == "sandstone"
    assert peace["legend"][0]["area"] == 0.25


def test_result_schema_has_required_contract_keys():
    assert MAP_PROCESSING_RESULT_SCHEMA["type"] == "object"
    assert "schema_version" in MAP_PROCESSING_RESULT_SCHEMA["required"]
    assert "regions" in MAP_PROCESSING_RESULT_SCHEMA["properties"]
    assert "legend" in MAP_PROCESSING_RESULT_SCHEMA["properties"]

import pytest

from peace_tool_pool.knowledge import (
    Bounds,
    KnowledgeBundle,
    KnowledgeItem,
    KnowledgeRequest,
    LegendEnrichment,
    KNOWLEDGE_BUNDLE_SCHEMA,
    SCHEMA_VERSION,
)
from peace_tool_pool.knowledge.bounds import split_antimeridian
from peace_tool_pool.knowledge.errors import InvalidBoundsError


def test_bounds_validation_normalizes_crs_and_serializes():
    bounds = Bounds(min_lon=-122.5, min_lat=37.0, max_lon=-121.5, max_lat=38.0)
    assert bounds.to_dict() == {
        "min_lon": -122.5,
        "min_lat": 37.0,
        "max_lon": -121.5,
        "max_lat": 38.0,
        "crs": "EPSG:4326",
    }

    crs84 = Bounds(min_lon=-1, min_lat=-2, max_lon=1, max_lat=2, crs="OGC:CRS84")
    assert crs84.crs == "EPSG:4326"

    with pytest.raises(InvalidBoundsError):
        Bounds(min_lon=-181, min_lat=0, max_lon=1, max_lat=1)
    with pytest.raises(InvalidBoundsError):
        Bounds(min_lon=0, min_lat=-91, max_lon=1, max_lat=1)
    with pytest.raises(InvalidBoundsError):
        Bounds(min_lon=10, min_lat=0, max_lon=1, max_lat=1)
    with pytest.raises(InvalidBoundsError):
        Bounds(min_lon=0, min_lat=0, max_lon=1, max_lat=1, crs="EPSG:3857")


def test_split_antimeridian_keeps_bounds_non_wrapping():
    parts = split_antimeridian(min_lon=170, min_lat=-10, max_lon=-170, max_lat=10)

    assert [part.to_dict() for part in parts] == [
        {"min_lon": 170.0, "min_lat": -10.0, "max_lon": 180.0, "max_lat": 10.0, "crs": "EPSG:4326"},
        {"min_lon": -180.0, "min_lat": -10.0, "max_lon": -170.0, "max_lat": 10.0, "crs": "EPSG:4326"},
    ]
    assert split_antimeridian(min_lon=-10, min_lat=0, max_lon=10, max_lat=1) == [
        Bounds(min_lon=-10, min_lat=0, max_lon=10, max_lat=1)
    ]


def test_request_bundle_and_enrichment_serialization():
    bounds = Bounds(min_lon=-122, min_lat=37, max_lon=-121, max_lat=38)
    request = KnowledgeRequest(
        bounds=bounds,
        legend_labels=["Sandstone"],
        include=["Rock Type"],
        exclude=["active-faults"],
        max_records=3,
        max_records_by_provider={"earthquake_history": 1},
        provider_options={"earthquake_history": {"minmagnitude": "4.5"}},
        trace_id="trace-1",
    )
    assert request.include == ("Rock Type",)
    assert request.exclude == ("active-faults",)
    assert request.provider_options == {"earthquake_history": {"minmagnitude": "4.5"}}
    assert request.to_dict()["bounds"] == bounds.to_dict()
    assert request.to_dict()["provider_options"] == {
        "earthquake_history": {"minmagnitude": "4.5"}
    }

    item = KnowledgeItem(
        id="rock_type:rock_type",
        key="rock_type",
        provider="rock_type",
        value={"label": "Sandstone", "value": "sedimentary"},
        summary="Sandstone: sedimentary",
        source="fixture",
        record_count=1,
        provenance={"match_type": "exact"},
    )
    bundle = KnowledgeBundle(
        bounds=bounds,
        items=[item],
        selected_item_ids=None,
        warnings=["unsupported provider ignored"],
        provider_versions={"rock_type": "1@sha256:abcdef123456"},
        trace={"trace_id": "trace-1"},
    )

    as_dict = bundle.to_dict()
    assert as_dict["schema_version"] == SCHEMA_VERSION
    assert as_dict["items"][0]["id"] == "rock_type:rock_type"
    assert bundle.items_by_id()["rock_type:rock_type"] == item
    assert bundle.items_by_key()["rock_type"] == [item]

    enrichment = LegendEnrichment(
        label="Sandstone",
        lithology="sedimentary",
        stratigraphic_age="mesozoic",
        items=[item],
        warnings=[],
    )
    assert enrichment.to_dict()["lithology"] == "sedimentary"


def test_knowledge_schema_constant_is_stable():
    assert KNOWLEDGE_BUNDLE_SCHEMA["type"] == "object"
    assert KNOWLEDGE_BUNDLE_SCHEMA["properties"]["schema_version"]["const"] == "knowledge/v2"
    assert "items" in KNOWLEDGE_BUNDLE_SCHEMA["required"]

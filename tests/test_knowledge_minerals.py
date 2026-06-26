"""Tests for the Ontario Geological Survey mineral-occurrence prototype provider.

The OGS Mineral Deposit Inventory is a live ArcGIS REST service. No live network
is used here: a fake client returns canned ArcGIS GeoJSON, mirroring the USGS
adapter test approach. These tests pin (a) field normalization, (b) the live
bbox query path, (c) the region-routing coverage seam (the deferred Approach-C
"region-aware source selection"), and (d) the optional-dependency gate.
"""

from __future__ import annotations

import pytest

from peace_tool_pool.knowledge import Bounds, KnowledgeRequest, KnowledgeService
from peace_tool_pool.knowledge.errors import OptionalDependencyError, ProviderOptionError
from peace_tool_pool.knowledge.providers.minerals import MineralOccurrenceProvider
from peace_tool_pool.knowledge.sources import ogs_minerals
from peace_tool_pool.knowledge.sources.ogs_minerals import (
    OgsMineralOccurrenceAdapter,
    normalize_features,
)


# Two real-shaped OGS MDI records near the Shebandowan belt (osmani AOI).
OGS_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "NAME": "GRANDE PORTAGE",
                "P_COMMOD": "GOLD",
                "S_COMMOD": "SILVER",
                "DEPOSIT_ST": "OCCURRENCE",
                "DEP_CLASS": "VEIN",
            },
            "geometry": {"type": "Point", "coordinates": [-90.6, 48.5]},
        },
        {
            "type": "Feature",
            "properties": {
                "NAME": "SHEWAN COPPER",
                "P_COMMOD": "COPPER",
                "DEPOSIT_ST": "OCCURRENCE",
            },
            "geometry": {"type": "Point", "coordinates": [-90.5, 48.6]},
        },
    ],
}

ONTARIO = Bounds(min_lon=-95.2, min_lat=41.6, max_lon=-74.3, max_lat=56.9)  # coverage bbox
SHEBANDOWAN = Bounds(min_lon=-90.90, min_lat=48.33, max_lon=-90.35, max_lat=48.76)
# Unambiguously outside Ontario's bbox (the eval control region). NOTE: a Quebec
# James Bay box (~-77..-75 lon) is NOT used here because it overlaps Ontario's
# rectangular bbox (Ontario reaches -74.3 at its SE tip) -- separating adjacent
# provinces needs the deferred Approach-C authority-polygon routing, not a bbox.
CALIFORNIA = Bounds(min_lon=-122.5, min_lat=37.0, max_lon=-121.5, max_lat=38.0)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    """Records calls and returns canned ArcGIS GeoJSON; never touches the network."""

    def __init__(self, payload):
        self._payload = payload
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return FakeResponse(self._payload)


def _adapter(payload=OGS_GEOJSON):
    return OgsMineralOccurrenceAdapter(client=FakeClient(payload))


def test_normalize_features_maps_arcgis_fields_to_stable_keys():
    records = normalize_features(OGS_GEOJSON)

    assert [r["name"] for r in records] == ["GRANDE PORTAGE", "SHEWAN COPPER"]
    gold = records[0]
    assert gold["primary_commodity"] == "GOLD"
    assert gold["secondary_commodity"] == "SILVER"
    assert gold["status"] == "OCCURRENCE"
    assert gold["longitude"] == -90.6 and gold["latitude"] == 48.5
    # Raw properties preserved for audit / future field-level use.
    assert gold["raw_properties"]["P_COMMOD"] == "GOLD"


def test_provider_live_query_returns_occurrences_with_attribution():
    provider = MineralOccurrenceProvider(adapter=_adapter(), coverage_bounds=ONTARIO)

    item = provider.query(KnowledgeRequest(bounds=SHEBANDOWAN))[0]

    assert item.record_count == 2
    assert any(r["primary_commodity"] == "GOLD" for r in item.value)
    prov = item.provenance
    assert prov["source_id"] == "ontario_mineral_deposit_inventory"
    assert prov["source_family"] == "mineral_occurrences"
    assert prov["source_mode"] == "live"
    # Attribution sourced from the registry (single source of truth).
    assert "Ontario" in (prov["license"] or "")
    assert prov["attribution"]


def test_provider_truncates_to_max_records():
    provider = MineralOccurrenceProvider(adapter=_adapter(), coverage_bounds=ONTARIO)

    item = provider.query(
        KnowledgeRequest(
            bounds=SHEBANDOWAN,
            max_records_by_provider={"mineral_occurrences": 1},
        )
    )[0]

    assert item.record_count == 2
    assert item.truncated is True
    assert len(item.value) == 1


def test_provider_skips_network_outside_coverage_region_and_warns():
    """Region-routing seam: a bbox outside Ontario must NOT query the Ontario source."""
    fake = FakeClient(OGS_GEOJSON)
    provider = MineralOccurrenceProvider(
        adapter=OgsMineralOccurrenceAdapter(client=fake),
        coverage_bounds=ONTARIO,
        region_name="Ontario",
    )

    item = provider.query(KnowledgeRequest(bounds=CALIFORNIA))[0]

    assert item.record_count == 0
    assert fake.calls == []  # no network call for an out-of-region query
    assert any("ontario" in w.lower() for w in provider.last_warnings)


def test_provider_zero_result_in_region_warns_absence_is_not_evidence():
    empty = {"type": "FeatureCollection", "features": []}
    provider = MineralOccurrenceProvider(adapter=_adapter(empty), coverage_bounds=ONTARIO)

    item = provider.query(KnowledgeRequest(bounds=SHEBANDOWAN))[0]

    assert item.record_count == 0
    assert any("not evidence" in w.lower() for w in provider.last_warnings)


def test_provider_requires_network_extra_when_no_client(monkeypatch):
    monkeypatch.setattr(ogs_minerals, "_httpx_module", lambda: None)
    provider = MineralOccurrenceProvider(
        adapter=OgsMineralOccurrenceAdapter(client=None),
        coverage_bounds=ONTARIO,
    )

    with pytest.raises(OptionalDependencyError):
        provider.query(KnowledgeRequest(bounds=SHEBANDOWAN))


def test_provider_validate_options_rejects_unknown_and_non_live_mode():
    provider = MineralOccurrenceProvider(adapter=_adapter(), coverage_bounds=ONTARIO)

    with pytest.raises(ProviderOptionError):
        provider.validate_options({"bogus": True})
    with pytest.raises(ProviderOptionError):
        provider.validate_options({"source_mode": "legacy_asset"})
    assert provider.validate_options({"source_mode": "live"})["source_mode"] == "live"


def test_service_routes_mineral_occurrences_when_explicitly_included():
    provider = MineralOccurrenceProvider(adapter=_adapter(), coverage_bounds=ONTARIO)
    service = KnowledgeService(providers=[provider])

    bundle = service.query_bounds(SHEBANDOWAN, include=("mineral_occurrences",))

    item = bundle.items_by_key()["mineral_occurrences"][0]
    assert item.record_count == 2
    assert bundle.provider_versions["mineral_occurrences"]

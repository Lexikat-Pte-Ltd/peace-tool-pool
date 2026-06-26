"""Ontario Geological Survey Mineral Deposit Inventory (MDI) source adapter.

The MDI is exposed as an ArcGIS REST FeatureServer that accepts a bbox envelope
and returns GeoJSON. Unlike the GEM/USGS sources (static file / sync-to-mirror),
this is a live regional service, so the adapter does a single bbox query rather
than a full-province sync. Networking mirrors the USGS adapter: ``httpx`` is
imported lazily (the ``knowledge-network`` extra) and a client may be injected
for tests so no live network is touched in CI.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..bounds import Bounds
from ..errors import OptionalDependencyError, SourceQueryError, SourceSyncError
from .registry import OGS_MDI_ENDPOINT

NORMALIZER_VERSION = "1"

# Normalized key -> candidate ArcGIS property names (matched case-insensitively).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("NAME", "MDI_NAME"),
    "primary_commodity": ("P_COMMOD", "COMMODITY"),
    "secondary_commodity": ("S_COMMOD",),
    "status": ("DEPOSIT_ST", "STATUS"),
    "deposit_class": ("DEP_CLASS", "DEPOSIT_TYPE"),
    "detail": ("DETAIL",),
}
SELECTED_COLUMNS = (
    "name",
    "primary_commodity",
    "secondary_commodity",
    "status",
    "deposit_class",
)


def _httpx_module() -> Any | None:
    try:
        import httpx
    except ImportError:
        return None
    return httpx


class OgsMineralOccurrenceAdapter:
    source_id = "ontario_mineral_deposit_inventory"
    family = "mineral_occurrences"
    normalizer_version = NORMALIZER_VERSION

    def __init__(
        self,
        endpoint: str = OGS_MDI_ENDPOINT,
        client: Any | None = None,
        timeout: float = 20.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.client = client
        self.timeout = float(timeout)

    def query_bbox(self, bounds: Bounds) -> dict[str, Any]:
        params = {
            "where": "1=1",
            "geometry": f"{bounds.min_lon},{bounds.min_lat},{bounds.max_lon},{bounds.max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
        }
        data = self._response_json(self._get("query", params))
        if not isinstance(data, dict):
            raise SourceQueryError("OGS MDI query response was not a GeoJSON object.")
        return data

    def _get(self, endpoint_suffix: str, params: Mapping[str, Any]) -> Any:
        url = f"{self.endpoint}/{endpoint_suffix}"
        client = self.client
        close_client = False
        if client is None:
            httpx = _httpx_module()
            if httpx is None:
                raise OptionalDependencyError(
                    "OGS mineral source networking requires `uv sync --extra knowledge-network`."
                )
            client = httpx.Client(timeout=self.timeout)
            close_client = True
        try:
            response = client.get(url, params=dict(params), timeout=self.timeout)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return response
        except OptionalDependencyError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport clients vary by test/client.
            raise SourceSyncError(f"OGS mineral source request failed: {exc}") from exc
        finally:
            if close_client:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    def _response_json(self, response: Any) -> Any:
        json_method = getattr(response, "json", None)
        if callable(json_method):
            return json_method()
        text = getattr(response, "text", None)
        if text is not None:
            return json.loads(str(text))
        return response


def normalize_features(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map ArcGIS MDI GeoJSON features to stable normalized records.

    Field names are matched case-insensitively against ``_FIELD_ALIASES`` so the
    output is robust to ArcGIS casing. Raw properties are preserved for audit and
    future field-level use (Approach-C readiness). Deduplicated by name + point.
    """
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for feature in data.get("features") or []:
        if not isinstance(feature, Mapping):
            continue
        properties = dict(feature.get("properties") or {})
        lowered = {str(k).lower(): v for k, v in properties.items()}
        record: dict[str, Any] = {}
        for normalized_key, aliases in _FIELD_ALIASES.items():
            record[normalized_key] = _first_present(lowered, aliases)
        longitude, latitude = _point(feature.get("geometry") or {})
        if longitude is not None:
            record["longitude"] = longitude
            record["latitude"] = latitude
        record["raw_properties"] = properties
        dedupe_key = (record.get("name"), record.get("longitude"), record.get("latitude"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(record)
    return records


def _first_present(lowered_properties: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = lowered_properties.get(alias.lower())
        if value not in (None, ""):
            return value
    return None


def _point(geometry: Mapping[str, Any]) -> tuple[float | None, float | None]:
    coordinates = geometry.get("coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        first, second = coordinates[0], coordinates[1]
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            return float(first), float(second)
    return None, None

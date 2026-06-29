"""DISS seismogenic-source WFS adapter."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..bounds import Bounds
from ..errors import OptionalDependencyError, SourceQueryError, SourceSyncError
from .registry import DISS_DEFAULT_PROFILE, _validate_diss_profile


NORMALIZER_VERSION = "1"


def _httpx_module() -> Any | None:
    try:
        import httpx
    except ImportError:
        return None
    return httpx


class DissSeismogenicSourceAdapter:
    source_id = "diss_seismogenic_sources"
    family = "active_faults"
    normalizer_version = NORMALIZER_VERSION

    def __init__(
        self,
        endpoint: str | None = None,
        client: Any | None = None,
        timeout: float = 20.0,
    ):
        self.endpoint = (endpoint or DISS_DEFAULT_PROFILE["endpoint"]).rstrip("?")
        self.client = client
        self.timeout = float(timeout)

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        return _validate_diss_profile({**DISS_DEFAULT_PROFILE, **dict(profile or {})})

    def query_params(
        self,
        bounds: Bounds,
        *,
        type_name: str,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated = self.validate_profile(profile)
        srs_name = str(validated.get("srs_name") or "CRS:84")
        return {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": type_name,
            "outputFormat": "application/json",
            "srsName": srs_name,
            "bbox": (
                f"{bounds.min_lon},{bounds.min_lat},{bounds.max_lon},{bounds.max_lat},"
                f"{srs_name}"
            ),
        }

    def query_bbox(self, bounds: Bounds, profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
        validated = self.validate_profile(profile)
        features: list[dict[str, Any]] = []
        for type_name in validated["type_names"]:
            params = self.query_params(bounds, type_name=type_name, profile=validated)
            data = self._response_json(self._get(params))
            if not isinstance(data, dict):
                raise SourceQueryError("DISS WFS response was not a GeoJSON object.")
            features.extend(list(data.get("features") or []))
        return {"type": "FeatureCollection", "features": features}

    def normalize_geojson(self, data: Mapping[str, Any]) -> dict[str, Any]:
        features: list[dict[str, Any]] = []
        for feature in data.get("features") or []:
            if not isinstance(feature, Mapping):
                continue
            properties = dict(feature.get("properties") or {})
            normalized = dict(properties)
            normalized["source_id"] = self.source_id
            normalized["raw_properties"] = properties
            features.append(
                {
                    "type": "Feature",
                    "id": feature.get("id") or properties.get("id"),
                    "properties": normalized,
                    "geometry": feature.get("geometry"),
                    **({"bbox": feature["bbox"]} if "bbox" in feature else {}),
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def _get(self, params: Mapping[str, Any]) -> Any:
        client = self.client
        close_client = False
        if client is None:
            httpx = _httpx_module()
            if httpx is None:
                raise OptionalDependencyError(
                    "DISS source networking requires `uv sync --extra knowledge-network`."
                )
            client = httpx.Client(timeout=self.timeout)
            close_client = True
        try:
            response = client.get(self.endpoint, params=dict(params), timeout=self.timeout)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return response
        except OptionalDependencyError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport clients vary by tests.
            raise SourceSyncError(f"DISS source request failed: {exc}") from exc
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

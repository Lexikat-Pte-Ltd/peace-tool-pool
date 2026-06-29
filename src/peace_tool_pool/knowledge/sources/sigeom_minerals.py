"""Quebec SIGEOM mineral-occurrence WFS adapter."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..bounds import Bounds
from ..errors import OptionalDependencyError, SourceQueryError, SourceSyncError
from .registry import SIGEOM_DEFAULT_PROFILE, _validate_sigeom_profile


# v2: field aliases corrected to the real SGM:Substances_metalliques schema
# (NOM_CORPS_MINR / SUBST_PRINC / ETAT_CORPS_MINR). Bumping invalidates any v1
# cache entries that were normalized with the wrong (blank-label) aliases.
NORMALIZER_VERSION = "2"

# Keys are matched case-insensitively (see ``_first_present``). The production
# ``SGM:Substances_metalliques`` WFS layer uses French-coded field names
# (``NOM_CORPS_MINR``, ``SUBST_PRINC``, ``ETAT_CORPS_MINR``, ``ELMN_CHIM_PERD``);
# the generic ``NOM``/``SUBSTANCE`` names are kept as forward-compatible fallbacks.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("NOM_CORPS_MINR", "NOM", "NOM_GITE", "NOM_INDICE", "SITE_NOM"),
    "primary_commodity": (
        "SUBST_PRINC",
        "SUBSTANCE",
        "SUBSTANCES",
        "SUBSTANCE_PRINCIPALE",
        "ELMN_CHIM_PERD",
    ),
    "secondary_commodity": ("SUBSTANCE_SECONDAIRE", "SUBSTANCES_SECONDAIRES"),
    "status": ("ETAT_CORPS_MINR", "STATUT", "ETAT", "TYPE"),
    "deposit_class": ("CLASSE", "TYPE_GITE", "MODELE", "CODE_TYPE_ROCH_LITH"),
}


def _httpx_module() -> Any | None:
    try:
        import httpx
    except ImportError:
        return None
    return httpx


class SigeomMineralOccurrenceAdapter:
    source_id = "sigeom_mineral_occurrences"
    family = "mineral_occurrences"
    normalizer_version = NORMALIZER_VERSION

    def __init__(
        self,
        endpoint: str | None = None,
        client: Any | None = None,
        timeout: float = 20.0,
    ):
        self.endpoint = (endpoint or SIGEOM_DEFAULT_PROFILE["endpoint"]).rstrip("?")
        self.client = client
        self.timeout = float(timeout)

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        return _validate_sigeom_profile({**SIGEOM_DEFAULT_PROFILE, **dict(profile or {})})

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
        for type_name in validated["feature_types"]:
            params = self.query_params(bounds, type_name=type_name, profile=validated)
            data = self._response_json(self._get(params))
            if not isinstance(data, dict):
                raise SourceQueryError("SIGEOM WFS response was not a GeoJSON object.")
            features.extend(list(data.get("features") or []))
        return {"type": "FeatureCollection", "features": features}

    def _get(self, params: Mapping[str, Any]) -> Any:
        client = self.client
        close_client = False
        if client is None:
            httpx = _httpx_module()
            if httpx is None:
                raise OptionalDependencyError(
                    "SIGEOM mineral source networking requires `uv sync --extra knowledge-network`."
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
            raise SourceSyncError(f"SIGEOM mineral source request failed: {exc}") from exc
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


def normalize_sigeom_features(data: Mapping[str, Any]) -> list[dict[str, Any]]:
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

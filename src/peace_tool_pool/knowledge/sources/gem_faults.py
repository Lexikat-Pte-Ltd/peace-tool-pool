"""GEM Global Active Faults source adapter and coverage caveats."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ..bounds import Bounds
from ..cache import write_json_atomic
from ..errors import OptionalDependencyError, SourceChecksumError, SourceSyncError
from ..providers.base import file_sha256_digest
from .manifest import SourceManifest
from .registry import GEM_DEFAULT_PROFILE, _validate_gem_profile


NORMALIZER_VERSION = "1"
GEM_ATTRIBUTION = (
    "GEM Global Active Faults Database (CC BY-SA 4.0) - "
    "https://doi.org/10.5281/zenodo.3376300"
)
GEM_GAP_BBOXES = {
    "canada": (-141.0, 41.0, -52.0, 84.0),
    "madagascar": (43.0, -26.0, 51.0, -11.0),
    "malay_archipelago": (95.0, -11.0, 141.0, 8.0),
}
_TUPLE_PATTERN = re.compile(r"^\(([^,]+),([^,]+),([^,]+)\)$")


def _httpx_module() -> Any | None:
    try:
        import httpx
    except ImportError:
        return None
    return httpx


def coverage_caveats_for_bounds(bounds_parts: list[Bounds]) -> list[str]:
    caveats: list[str] = []
    for name, raw_bbox in GEM_GAP_BBOXES.items():
        gap = Bounds(*raw_bbox)
        if any(_bounds_intersect(part, gap) for part in bounds_parts):
            caveats.append(
                f"GEM active-fault coverage has a documented coarse gap near {name}; "
                "missing features may reflect source coverage rather than absence of active faults."
            )
    return caveats


class GemActiveFaultSourceAdapter:
    source_id = "gem_global_active_faults"
    family = "active_faults"
    normalizer_version = NORMALIZER_VERSION

    def __init__(self, client: Any | None = None, timeout: float = 30.0):
        self.client = client
        self.timeout = float(timeout)

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        return _validate_gem_profile({**GEM_DEFAULT_PROFILE, **dict(profile or {})})

    def sync(
        self,
        output_root: str | Path,
        profile: Mapping[str, Any] | None = None,
        version: str | None = None,
    ) -> SourceManifest:
        validated = self.validate_profile(profile)
        source_version = str(version or validated["source_version"])
        root = Path(output_root) / self.source_id / source_version.replace("/", "_")
        raw_root = root / "raw"
        normalized_root = root / "normalized"
        raw_root.mkdir(parents=True, exist_ok=True)
        normalized_root.mkdir(parents=True, exist_ok=True)

        data = self.fetch_geojson(validated)
        raw_path = raw_root / "gem_active_faults.geojson"
        write_json_atomic(raw_path, data)
        normalized = self.normalize_geojson(data)
        normalized_path = normalized_root / "faults.geojson"
        write_json_atomic(normalized_path, normalized)
        digest = file_sha256_digest(normalized_path, prefix=64)
        expected_digest = validated.get("sha256")
        if expected_digest and expected_digest != digest:
            raise SourceChecksumError(
                f"GEM normalized artifact SHA256 {digest} did not match expected {expected_digest}."
            )
        manifest = SourceManifest(
            source_id=self.source_id,
            family=self.family,
            retrieved_at=_format_time(datetime.now(UTC)),
            source_version=source_version,
            normalizer_version=self.normalizer_version,
            source_url=str(validated.get("source_url") or validated.get("local_path")),
            request=validated,
            record_count=len(normalized.get("features") or []),
            normalized_sha256=digest,
            license="CC BY-SA 4.0 (Creative Commons Attribution Share Alike 4.0 International)",
            citation="GEM Global Active Faults Database",
            attribution=GEM_ATTRIBUTION,
            coverage={
                "status": "broad-but-incomplete-global-dataset",
                "notes": [
                    "GEM README notes gaps including Malay Archipelago, Madagascar, and Canada."
                ],
            },
            artifacts={"normalized": "normalized/faults.geojson"},
        )
        manifest.write(root / "manifest.json")
        return manifest

    def fetch_geojson(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        local_path = profile.get("local_path")
        if local_path:
            return json.loads(Path(local_path).read_text(encoding="utf-8"))
        client = self.client
        close_client = False
        if client is None:
            httpx = _httpx_module()
            if httpx is None:
                raise OptionalDependencyError(
                    "GEM source networking requires `uv sync --extra knowledge-network`."
                )
            client = httpx.Client(timeout=self.timeout)
            close_client = True
        try:
            response = client.get(str(profile["source_url"]), timeout=self.timeout)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            json_method = getattr(response, "json", None)
            if callable(json_method):
                return json_method()
            return json.loads(str(getattr(response, "text")))
        except OptionalDependencyError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport clients vary by test/client.
            raise SourceSyncError(f"GEM active-fault source request failed: {exc}") from exc
        finally:
            if close_client:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    def normalize_geojson(self, data: Mapping[str, Any]) -> dict[str, Any]:
        features = []
        for feature in data.get("features") or []:
            if not isinstance(feature, Mapping):
                continue
            properties = dict(feature.get("properties") or {})
            normalized_properties = dict(properties)
            normalized_properties["raw_properties"] = dict(properties)
            for key, value in properties.items():
                parsed = _parse_tuple_value(value)
                if parsed is not None:
                    normalized_properties[f"{key}_uncertainty"] = parsed
            features.append(
                {
                    "type": "Feature",
                    "properties": normalized_properties,
                    "geometry": feature.get("geometry"),
                    **({"bbox": feature["bbox"]} if "bbox" in feature else {}),
                }
            )
        return {"type": "FeatureCollection", "features": features}


def _parse_tuple_value(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    match = _TUPLE_PATTERN.match(value.strip())
    if match is None:
        return None
    labels = ("most_likely", "min", "max")
    return {label: _coerce_tuple_part(part) for label, part in zip(labels, match.groups())}


def _coerce_tuple_part(value: str) -> Any:
    stripped = value.strip()
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    if number.is_integer():
        return int(number)
    return number


def _bounds_intersect(left: Bounds, right: Bounds) -> bool:
    return not (
        left.max_lon < right.min_lon
        or left.min_lon > right.max_lon
        or left.max_lat < right.min_lat
        or left.min_lat > right.max_lat
    )


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

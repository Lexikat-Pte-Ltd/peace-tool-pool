"""USGS FDSN Event source adapter and normalizer."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

from ..bounds import Bounds
from ..cache import write_json_atomic
from ..errors import OptionalDependencyError, SourceQueryError, SourceSyncError
from ..providers.base import file_sha256_digest
from .manifest import SourceManifest
from .registry import (
    EMSC_DEFAULT_PROFILE,
    USGS_DEFAULT_PROFILE,
    _validate_fdsn_event_profile,
    _validate_usgs_profile,
)


USGS_EVENT_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1"
EMSC_EVENT_BASE_URL = "https://www.seismicportal.eu/fdsnws/event/1"
NORMALIZER_VERSION = "1"
EARTHQUAKE_CSV_COLUMNS = (
    "identity_key",
    "event_id",
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "magType",
    "magSource",
    "net",
    "code",
    "ids",
    "sources",
    "place",
    "type",
    "status",
    "reviewstatus",
    "updated",
    "gap",
    "raw_properties",
)


def _httpx_module() -> Any | None:
    try:
        import httpx
    except ImportError:
        return None
    return httpx


def associated_id_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def event_identity_key(record: Mapping[str, Any]) -> str:
    ids = associated_id_set(record.get("ids"))
    if ids:
        return ",".join(sorted(ids))
    for key in ("identity_key", "event_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return "|".join(
        str(record.get(key, ""))
        for key in ("time", "latitude", "longitude", "depth", "mag")
    )


class UsgsFdsnEventAdapter:
    source_id = "usgs_fdsn_events"
    family = "earthquake_events"
    normalizer_version = NORMALIZER_VERSION

    def __init__(
        self,
        client: Any | None = None,
        base_url: str = USGS_EVENT_BASE_URL,
        timeout: float = 15.0,
    ):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        return _validate_usgs_profile({**USGS_DEFAULT_PROFILE, **dict(profile or {})})

    def query_params(
        self,
        profile: Mapping[str, Any] | None,
        bounds: Bounds | None = None,
        *,
        for_count: bool = False,
    ) -> dict[str, Any]:
        validated = self.validate_profile(profile)
        params = dict(validated)
        if bounds is not None:
            params.update(
                {
                    "minlatitude": bounds.min_lat,
                    "maxlatitude": bounds.max_lat,
                    "minlongitude": bounds.min_lon,
                    "maxlongitude": bounds.max_lon,
                }
            )
        if for_count:
            for key in ("format", "orderby", "limit", "offset"):
                params.pop(key, None)
        return params

    def count(self, profile: Mapping[str, Any] | None, bounds: Bounds | None = None) -> int:
        response = self._get("count", self.query_params(profile, bounds=bounds, for_count=True))
        text = getattr(response, "text", None)
        if text is not None:
            try:
                return int(str(text).strip())
            except ValueError:
                pass
        data = self._response_json(response)
        if isinstance(data, int):
            return data
        if isinstance(data, Mapping) and "count" in data:
            return int(data["count"])
        raise SourceQueryError("USGS count response did not contain an event count.")

    def query_geojson(
        self,
        profile: Mapping[str, Any] | None,
        bounds: Bounds | None = None,
    ) -> dict[str, Any]:
        params = self.query_params(profile, bounds=bounds)
        params["format"] = "geojson"
        data = self._response_json(self._get("query", params))
        if not isinstance(data, dict):
            raise SourceQueryError("USGS query response was not a GeoJSON object.")
        return data

    def split_time_window(
        self,
        profile: Mapping[str, Any],
        count_func: Callable[[dict[str, Any]], int],
    ) -> list[dict[str, Any]]:
        validated = self.validate_profile(profile)
        start = _parse_time(str(validated.get("starttime", "1970-01-01")))
        end_value = validated.get("endtime")
        end = _parse_time(str(end_value)) if end_value else datetime.now(UTC)
        windows: list[dict[str, Any]] = []
        for window_start, window_end in _yearly_windows(start, end):
            window_profile = {
                **validated,
                "starttime": _format_time(window_start),
                "endtime": _format_time(window_end),
            }
            windows.extend(self._split_until_under_limit(window_profile, count_func))
        return windows

    def live_records(
        self,
        profile: Mapping[str, Any] | None,
        bounds_parts: list[Bounds],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for part in bounds_parts:
            count = self.count(profile, bounds=part)
            if count > 20000:
                raise SourceQueryError(
                    "USGS live query returned more than 20000 events; narrow time, "
                    "magnitude, or bounds filters."
                )
            if count == 0:
                continue
            records.extend(normalize_geojson_events(self.query_geojson(profile, bounds=part)))
        return dedupe_event_records(records)

    def sync(
        self,
        output_root: str | Path,
        profile: Mapping[str, Any] | None = None,
        bounds: Bounds | None = None,
        version: str = "default",
    ) -> SourceManifest:
        validated = self.validate_profile(profile)
        root = Path(output_root) / self.source_id / version
        raw_root = root / "raw"
        normalized_root = root / "normalized"
        raw_root.mkdir(parents=True, exist_ok=True)
        normalized_root.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        page_index = 1
        windows = self.split_time_window(validated, lambda window: self.count(window, bounds=bounds))
        for window in windows:
            if self.count(window, bounds=bounds) == 0:
                continue
            page = self.query_geojson(window, bounds=bounds)
            raw_path = raw_root / f"query-{page_index:04d}.geojson"
            write_json_atomic(raw_path, page)
            records.extend(normalize_geojson_events(page))
            page_index += 1

        records = dedupe_event_records(records)
        normalized_path = normalized_root / "earthquakes.csv"
        write_earthquake_csv(normalized_path, records)
        digest = file_sha256_digest(normalized_path, prefix=64)
        manifest = SourceManifest(
            source_id=self.source_id,
            family=self.family,
            retrieved_at=_format_time(datetime.now(UTC)),
            source_version="usgs-fdsn-event-service",
            normalizer_version=self.normalizer_version,
            source_url=f"{self.base_url}/query",
            request=validated,
            record_count=len(records),
            normalized_sha256=digest,
            license="See USGS source policy",
            citation="USGS FDSN Event API",
            attribution="USGS Earthquake Hazards Program FDSN Event API",
            coverage={"status": "global-service", "notes": []},
            artifacts={"normalized": "normalized/earthquakes.csv"},
        )
        manifest.write(root / "manifest.json")
        return manifest

    def _split_until_under_limit(
        self,
        profile: dict[str, Any],
        count_func: Callable[[dict[str, Any]], int],
    ) -> list[dict[str, Any]]:
        count = count_func(profile)
        if count <= 20000:
            return [profile]
        start = _parse_time(str(profile["starttime"]))
        end = _parse_time(str(profile["endtime"]))
        if end - start < timedelta(days=1):
            raise SourceQueryError(
                "USGS event query still exceeds 20000 events below one-day granularity."
            )
        midpoint = start + (end - start) / 2
        first = {**profile, "endtime": _format_time(midpoint)}
        second = {**profile, "starttime": _format_time(midpoint)}
        return self._split_until_under_limit(first, count_func) + self._split_until_under_limit(
            second, count_func
        )

    def _get(self, endpoint: str, params: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}/{endpoint}"
        client = self.client
        close_client = False
        if client is None:
            httpx = _httpx_module()
            if httpx is None:
                raise OptionalDependencyError(
                    "USGS source networking requires `uv sync --extra knowledge-network`."
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
            raise SourceSyncError(f"USGS source request failed: {exc}") from exc
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


class FdsnEventSourceAdapter(UsgsFdsnEventAdapter):
    """GeoJSON/JSON FDSN event adapter for explicit federated sources.

    USGS keeps its source-specific adapter because its default mirror profile and
    historical chunking behavior are already pinned. This adapter covers other
    FDSN services that can return GeoJSON-like JSON, such as EMSC.
    """

    def __init__(
        self,
        source_id: str,
        base_url: str,
        default_profile: Mapping[str, Any] | None = None,
        client: Any | None = None,
        timeout: float = 15.0,
    ):
        super().__init__(client=client, base_url=base_url, timeout=timeout)
        self.source_id = source_id
        self.default_profile = dict(default_profile or EMSC_DEFAULT_PROFILE)

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        return _validate_fdsn_event_profile(
            {**self.default_profile, **dict(profile or {})},
            source_name=self.source_id,
        )

    def query_geojson(
        self,
        profile: Mapping[str, Any] | None,
        bounds: Bounds | None = None,
    ) -> dict[str, Any]:
        params = self.query_params(profile, bounds=bounds)
        params["format"] = str(params.get("format") or self.default_profile.get("format") or "json")
        data = self._response_json(self._get("query", params))
        if not isinstance(data, dict):
            raise SourceQueryError(f"{self.source_id} query response was not a GeoJSON object.")
        return data


def normalize_geojson_events(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in data.get("features") or []:
        if not isinstance(feature, Mapping):
            continue
        properties = dict(feature.get("properties") or {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        ids = associated_id_set(properties.get("ids"))
        event_id = str(feature.get("id") or properties.get("code") or "")
        identity_key = ",".join(sorted(ids)) or event_id
        depth = coordinates[2] if len(coordinates) > 2 else properties.get("depth")
        records.append(
            {
                "identity_key": identity_key,
                "event_id": event_id,
                "time": properties.get("time"),
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "depth": depth,
                "mag": properties.get("mag"),
                "magType": properties.get("magType"),
                "magSource": properties.get("magSource"),
                "net": properties.get("net"),
                "code": properties.get("code"),
                "ids": ",".join(sorted(ids)),
                "sources": properties.get("sources"),
                "place": properties.get("place"),
                "type": properties.get("type"),
                "status": properties.get("status"),
                "reviewstatus": properties.get("reviewstatus"),
                "updated": properties.get("updated"),
                "gap": properties.get("gap"),
                "raw_properties": json.dumps(properties, sort_keys=True, separators=(",", ":")),
            }
        )
    return dedupe_event_records(records)


def dedupe_event_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    identity_sets: list[set[str]] = []
    for record in records:
        ids = associated_id_set(record.get("ids"))
        if not ids:
            ids = {event_identity_key(record)}
        replacement_index: int | None = None
        for index, existing_ids in enumerate(identity_sets):
            if ids & existing_ids:
                replacement_index = index
                break
        if replacement_index is None:
            deduped.append(record)
            identity_sets.append(ids)
        else:
            deduped[replacement_index] = record
            identity_sets[replacement_index] |= ids
    return deduped


def write_earthquake_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=EARTHQUAKE_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _parse_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    if "T" not in normalized and len(normalized) == 10:
        normalized = f"{normalized}T00:00:00+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _yearly_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        next_year = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
        window_end = min(next_year, end)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows

"""Local earthquake history provider."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path
from typing import Any

from ..errors import OptionalDependencyError
from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class EarthquakeHistoryProvider:
    id = "earthquake_history"
    name = "Earthquake history"
    version = "1"
    output_keys = ("earthquake_history",)

    _selected_columns = (
        "time",
        "latitude",
        "longitude",
        "place",
        "mag",
        "magType",
        "depth",
        "type",
        "updated",
        "gap",
    )

    def __init__(
        self,
        asset_path: str | Path,
        default_max_records: int = 50,
        margin_degrees: float = 0.05,
        engine: str = "auto",
    ):
        self.asset_path = Path(asset_path)
        self.default_max_records = default_max_records
        self.margin_degrees = float(margin_degrees)
        self.engine = engine
        self._rows: list[dict[str, str]] | None = None
        self._frame: Any | None = None
        self._digest: str | None = None
        self._active_engine = "csv"

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        return source_version(self.version, self._digest)

    def cache_config(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "resolved_engine": self._resolved_engine_for_cache(),
            "margin_degrees": self.margin_degrees,
        }

    def _resolved_engine_for_cache(self) -> str:
        if self.engine == "auto":
            return "pandas" if _dependency_available("pandas") else "csv"
        return self.engine

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        self.source_version()
        if request.bounds is None:
            return []
        matching = self._matching_rows(request)
        matching.sort(
            key=lambda row: (str(row.get("time") or ""), self._float_or_default(row.get("mag"))),
            reverse=True,
        )
        limit = max_records_for_request(self.id, request, self.default_max_records)
        limited = matching[:limit]
        records = [self._shape_row(row) for row in limited]
        total = len(matching)
        truncated = total > len(records)
        if total:
            summary = f"Found {total} earthquakes within bounds; returning {len(records)} records."
        else:
            summary = "No earthquakes with configured filters were found within bounds."
        return [
            KnowledgeItem(
                id=f"{self.id}:{self.id}",
                key=self.id,
                provider=self.id,
                value=records,
                summary=summary,
                source=str(self.asset_path),
                record_count=total,
                truncated=truncated,
                provenance={
                    "asset_path": str(self.asset_path),
                    "engine": self._active_engine,
                    "margin_degrees": self.margin_degrees,
                },
            )
        ]

    def _matching_rows(self, request: KnowledgeRequest) -> list[dict[str, Any]]:
        if self.engine in {"auto", "pandas"}:
            try:
                return self._matching_rows_pandas(request)
            except OptionalDependencyError:
                if self.engine == "pandas":
                    raise
        if self.engine not in {"auto", "csv", "pandas"}:
            raise ValueError(f"Unsupported earthquake provider engine: {self.engine!r}")
        self._active_engine = "csv"
        rows = self._load_rows()
        return [row for row in rows if self._row_in_bounds(row, request)]

    def _matching_rows_pandas(self, request: KnowledgeRequest) -> list[dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise OptionalDependencyError(
                "EarthquakeHistoryProvider pandas engine requires `uv sync --extra knowledge-local`."
            ) from exc

        frame = self._load_frame(pd)
        if request.bounds is None or "latitude" not in frame.columns or "longitude" not in frame.columns:
            self._active_engine = "pandas"
            return []
        latitude = pd.to_numeric(frame["latitude"], errors="coerce")
        longitude = pd.to_numeric(frame["longitude"], errors="coerce")
        bounds = request.bounds
        mask = (
            (latitude >= bounds.min_lat - self.margin_degrees)
            & (latitude <= bounds.max_lat + self.margin_degrees)
            & (longitude >= bounds.min_lon - self.margin_degrees)
            & (longitude <= bounds.max_lon + self.margin_degrees)
        )
        matching = frame.loc[mask].copy()
        self._active_engine = "pandas"
        return matching.to_dict(orient="records")

    def _load_frame(self, pandas_module: Any) -> Any:
        if self._frame is None:
            self._frame = pandas_module.read_csv(self.asset_path)
        return self._frame

    def _load_rows(self) -> list[dict[str, str]]:
        if self._rows is not None:
            return self._rows
        with self.asset_path.open("r", encoding="utf-8", newline="") as file_obj:
            self._rows = [dict(row) for row in csv.DictReader(file_obj)]
        return self._rows

    def _row_in_bounds(self, row: dict[str, str], request: KnowledgeRequest) -> bool:
        bounds = request.bounds
        if bounds is None:
            return False
        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            bounds.min_lat - self.margin_degrees <= latitude <= bounds.max_lat + self.margin_degrees
            and bounds.min_lon - self.margin_degrees
            <= longitude
            <= bounds.max_lon + self.margin_degrees
        )

    def _shape_row(self, row: dict[str, Any]) -> dict[str, Any]:
        columns = [column for column in self._selected_columns if column in row]
        if not columns:
            columns = list(row)
        return {column: self._coerce_value(row.get(column)) for column in columns}

    def _coerce_value(self, value: Any) -> Any:
        if value is None:
            return None
        item_method = getattr(value, "item", None)
        if callable(item_method) and not isinstance(value, (str, bytes)):
            value = item_method()
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        value = str(value)
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        if number.is_integer() and "." not in stripped and "e" not in stripped.lower():
            return int(number)
        return number

    def _float_or_default(self, value: str | None) -> float:
        try:
            return float(value or 0)
        except ValueError:
            return 0.0

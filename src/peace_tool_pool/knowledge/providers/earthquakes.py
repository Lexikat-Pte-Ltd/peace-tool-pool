"""Local earthquake history provider."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


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
    ):
        self.asset_path = Path(asset_path)
        self.default_max_records = default_max_records
        self.margin_degrees = float(margin_degrees)
        self._rows: list[dict[str, str]] | None = None
        self._digest: str | None = None

    def supports(self, request: KnowledgeRequest) -> bool:
        return request.bounds is not None

    def source_version(self) -> str:
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        return source_version(self.version, self._digest)

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        self.source_version()
        if request.bounds is None:
            return []
        rows = self._load_rows()
        matching = [row for row in rows if self._row_in_bounds(row, request)]
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
                    "margin_degrees": self.margin_degrees,
                },
            )
        ]

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

    def _shape_row(self, row: dict[str, str]) -> dict[str, Any]:
        columns = [column for column in self._selected_columns if column in row]
        if not columns:
            columns = list(row)
        return {column: self._coerce_value(row.get(column)) for column in columns}

    def _coerce_value(self, value: str | None) -> Any:
        if value is None:
            return None
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

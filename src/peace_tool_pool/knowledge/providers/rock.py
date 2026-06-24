"""Local K2 rock type and age lookup providers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, source_version


@dataclass(frozen=True)
class _RockEntry:
    name: str
    value: str


class RockLookupProvider:
    version = "1"

    def __init__(
        self,
        asset_path: str | Path,
        provider_id: str,
        output_key: str,
        name: str,
    ):
        self.asset_path = Path(asset_path)
        self.id = provider_id
        self.output_key = output_key
        self.name = name
        self.output_keys = (output_key,)
        self._entries: list[_RockEntry] | None = None
        self._digest: str | None = None

    def supports(self, request: KnowledgeRequest) -> bool:
        return bool(request.legend_labels)

    def source_version(self) -> str:
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        return source_version(self.version, self._digest)

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        self.source_version()
        entries = self._load_entries()
        labels = request.legend_labels
        items: list[KnowledgeItem] = []
        for index, label in enumerate(labels):
            value, matched_name, match_type = self._match(label, entries)
            item_id = f"{self.id}:{self.output_key}" if len(labels) == 1 else f"{self.id}:{self.output_key}:{index}"
            record_count = 0 if value == "unknown" else 1
            summary = f"{label}: {value}" if record_count else f"No {self.name.lower()} match for {label}."
            items.append(
                KnowledgeItem(
                    id=item_id,
                    key=self.output_key,
                    provider=self.id,
                    value={
                        "label": label,
                        "value": value,
                        "matched_name": matched_name,
                        "match_type": match_type,
                    },
                    summary=summary,
                    source=str(self.asset_path),
                    record_count=record_count,
                    truncated=False,
                    provenance={
                        "asset_path": str(self.asset_path),
                        "matched_name": matched_name,
                        "match_type": match_type,
                    },
                )
            )
        return items

    def _load_entries(self) -> list[_RockEntry]:
        if self._entries is not None:
            return self._entries
        data = json.loads(self.asset_path.read_text(encoding="utf-8"))
        entries: list[_RockEntry] = []
        for item in data:
            rock_name = str(item["rock_name"]).lower().strip()
            rock_value = str(item["rock_value"]).lower().strip()
            if rock_name:
                entries.append(_RockEntry(name=rock_name, value=rock_value))
        self._entries = entries
        return entries

    def _match(self, label: str, entries: list[_RockEntry]) -> tuple[str, str | None, str | None]:
        names = self._split_rock_name(label)
        by_name = {entry.name: entry for entry in entries}
        for name in names:
            exact = by_name.get(name)
            if exact is not None:
                return exact.value, exact.name, "exact"

        candidates: list[tuple[int, int, str, str]] = []
        for entry in entries:
            for name in names:
                if name in entry.name or entry.name in name:
                    overlap = min(len(name), len(entry.name))
                    candidates.append((overlap, len(entry.name), entry.name, entry.value))
        if not candidates:
            return "unknown", None, None
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        _, _, matched_name, value = candidates[0]
        return value, matched_name, "substring"

    def _split_rock_name(self, rock_name: str | None) -> list[str]:
        if rock_name is None:
            return []
        keywords = [",", "、", "-", " and ", "和", "及", "或", "\n", "/", "("]
        pattern = "|".join(map(re.escape, keywords))
        names = re.split(pattern, rock_name.lower().strip())
        cleaned = [self._clean_rock_name(name.strip().strip(")")) for name in names]
        return [name for name in cleaned if name]

    def _clean_rock_name(self, name: str) -> str:
        for key in ["脉", "?", ")", "member", "."]:
            name = name.replace(key, "").strip()
        for key in ["夹", "（", "。", ":"]:
            if key in name:
                name = name[: name.find(key)].strip()
        for key in ["的", "色", "—"]:
            if key in name:
                name = name[name.find(key) + 1 :].strip()
        return name.strip()

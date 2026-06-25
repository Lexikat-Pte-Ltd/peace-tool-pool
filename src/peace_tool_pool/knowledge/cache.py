"""Filesystem cache helpers for knowledge provider outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .config import KnowledgeConfig
from .types import KnowledgeItem, SCHEMA_VERSION


def write_json_atomic(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as temp_file:
            temp_name = temp_file.name
            json.dump(data, temp_file, indent=2, ensure_ascii=False, sort_keys=True)
        Path(temp_name).replace(target)
    except Exception:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class KnowledgeCache:
    def __init__(self, config: KnowledgeConfig):
        self.config = config

    def provider_dir(self, provider_id: str) -> Path:
        return self.config.cache_namespace_root / "providers" / provider_id

    def provider_path(self, provider_id: str, cache_key: str) -> Path:
        return self.provider_dir(provider_id) / f"{cache_key}.json"

    def read_provider_items(
        self,
        provider_id: str,
        cache_key: str,
        provider_version: str,
    ) -> list[KnowledgeItem] | None:
        path = self.provider_path(provider_id, cache_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        if data.get("provider_version") != provider_version:
            return None
        try:
            return [KnowledgeItem.from_dict(item) for item in data.get("items", [])]
        except (KeyError, TypeError, ValueError):
            return None

    def write_provider_items(
        self,
        provider_id: str,
        cache_key: str,
        provider_version: str,
        items: list[KnowledgeItem],
    ) -> None:
        write_json_atomic(
            self.provider_path(provider_id, cache_key),
            {
                "schema_version": SCHEMA_VERSION,
                "provider": provider_id,
                "provider_version": provider_version,
                "items": [item.to_dict() for item in items],
            },
        )

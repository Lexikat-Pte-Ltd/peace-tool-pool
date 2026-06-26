"""Knowledge provider protocol and shared provider helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..bounds import Bounds

from ..errors import MissingAssetError
from ..types import KnowledgeItem, KnowledgeRequest


class KnowledgeProvider(Protocol):
    id: str
    name: str
    version: str

    def supports(self, request: KnowledgeRequest) -> bool:
        """Return whether the provider can satisfy this request shape."""

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        """Return knowledge items for the request."""

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Validate provider-specific request options."""

    def query_bounds_parts(
        self,
        request: KnowledgeRequest,
        bounds_parts: list[Bounds],
    ) -> list[KnowledgeItem]:
        """Return one merged item for already-normalized bounds parts."""


def file_sha256_digest(path: str | Path, prefix: int = 12) -> str:
    source = Path(path)
    if not source.exists():
        raise MissingAssetError(f"Knowledge asset does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:prefix]


def source_version(code_version: str, digest: str) -> str:
    return f"{code_version}@sha256:{digest}"


def max_records_for_request(
    provider_id: str,
    request: KnowledgeRequest,
    default_max_records: int,
) -> int:
    limit = request.max_records_by_provider.get(provider_id)
    if limit is None:
        limit = request.max_records
    if limit is None:
        limit = default_max_records
    return max(0, int(limit))

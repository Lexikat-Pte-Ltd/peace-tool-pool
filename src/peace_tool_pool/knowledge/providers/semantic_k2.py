"""Optional semantic K2 retrieval providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from ..cache import stable_hash
from ..errors import OptionalDependencyError, ProviderError
from ..types import KnowledgeItem, KnowledgeRequest
from .base import file_sha256_digest, max_records_for_request, source_version


@dataclass(frozen=True)
class SemanticSearchResult:
    index: int
    score: float


class SemanticBackend(Protocol):
    def embed_corpus(self, corpus_texts: list[str], *, batch_size: int) -> Any:
        """Embed corpus texts once per provider."""

    def search(
        self,
        query_text: str,
        corpus_embeddings: Any,
        *,
        top_k: int,
        batch_size: int,
    ) -> list[SemanticSearchResult]:
        """Return ranked corpus indexes for a query."""


def resolve_embedding_device(
    configured: str | None,
    *,
    cuda_available: Callable[[], bool],
) -> str:
    requested = (configured or "auto").strip().lower()
    if requested in {"", "auto", "cuda_if_available"}:
        return "cuda" if cuda_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available():
            raise ProviderError(
                f"Configured semantic K2 device {configured!r}, but CUDA is unavailable."
            )
        return requested
    raise ProviderError(f"Unsupported semantic K2 device {configured!r}.")


class SentenceTransformerSemanticBackend:
    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.configured_device = device
        self.local_files_only = local_files_only
        self.resolved_device: str | None = None
        self._model: Any | None = None
        self._semantic_search: Any | None = None
        self._dot_score: Any | None = None

    def embed_corpus(self, corpus_texts: list[str], *, batch_size: int) -> Any:
        model = self._load_model()
        return model.encode(
            corpus_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=self.resolved_device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(
        self,
        query_text: str,
        corpus_embeddings: Any,
        *,
        top_k: int,
        batch_size: int,
    ) -> list[SemanticSearchResult]:
        if top_k <= 0:
            return []
        model = self._load_model()
        query_embedding = model.encode(
            [query_text],
            batch_size=batch_size,
            convert_to_tensor=True,
            device=self.resolved_device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        hits = self._semantic_search(
            query_embedding,
            corpus_embeddings,
            top_k=top_k,
            score_function=self._dot_score,
        )[0]
        return [
            SemanticSearchResult(index=int(hit["corpus_id"]), score=float(hit["score"]))
            for hit in hits
        ]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.util import dot_score, semantic_search
        except ImportError as exc:
            raise OptionalDependencyError(
                "Semantic K2 providers require `uv sync --extra knowledge-semantic`."
            ) from exc

        self.resolved_device = resolve_embedding_device(
            self.configured_device,
            cuda_available=torch.cuda.is_available,
        )
        kwargs: dict[str, Any] = {
            "device": self.resolved_device,
            "trust_remote_code": False,
            "local_files_only": self.local_files_only,
        }
        if self.model_revision is not None:
            kwargs["revision"] = self.model_revision
        self._model = SentenceTransformer(self.model_name, **kwargs)
        self._semantic_search = semantic_search
        self._dot_score = dot_score
        return self._model


@dataclass(frozen=True)
class _SemanticEntry:
    key: str
    answer: Any


class SemanticK2Provider:
    version = "1"

    def __init__(
        self,
        *,
        provider_id: str,
        name: str,
        output_key: str,
        asset_path: str | Path,
        query_field: str,
        answer_field: str,
        query_template: str,
        backend_factory: Callable[[], SemanticBackend],
        default_top_k: int = 5,
        min_score: float | None = None,
        batch_size: int = 32,
        model_name: str,
        model_revision: str | None = None,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        self.id = provider_id
        self.name = name
        self.output_keys = (output_key,)
        self.output_key = output_key
        self.asset_path = Path(asset_path)
        self.query_field = query_field
        self.answer_field = answer_field
        self.query_template = query_template
        self.backend_factory = backend_factory
        self.default_top_k = int(default_top_k)
        self.min_score = min_score
        self.batch_size = int(batch_size)
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.local_files_only = local_files_only
        self._digest: str | None = None
        self._entries: list[_SemanticEntry] | None = None
        self._corpus_embeddings: Any | None = None

    def supports(self, request: KnowledgeRequest) -> bool:
        return bool(request.query_text)

    def source_version(self) -> str:
        if self._digest is None:
            self._digest = file_sha256_digest(self.asset_path)
        config_suffix = stable_hash(self.cache_config())[:12]
        return (
            f"{source_version(self.version, self._digest)}:model={self.model_name}:"
            f"revision={self.model_revision or 'none'}:config={config_suffix}"
        )

    def cache_config(self) -> dict[str, Any]:
        return {
            "query_field": self.query_field,
            "answer_field": self.answer_field,
            "query_template": self.query_template,
            "default_top_k": self.default_top_k,
            "min_score": self.min_score,
            "batch_size": self.batch_size,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "local_files_only": self.local_files_only,
        }

    def query(self, request: KnowledgeRequest) -> list[KnowledgeItem]:
        self.source_version()
        if not request.query_text:
            return []
        entries = self._load_entries()
        limit = max_records_for_request(self.id, request, self.default_top_k)
        rendered_query = self.query_template.format(query=request.query_text)
        matches = self._search(rendered_query, limit)
        records = self._records_for_matches(entries, matches)
        truncated = self._is_truncated(entries=entries, matches=matches, records=records, limit=limit)
        if records:
            summary = f"Found {len(records)} semantic K2 matches for query text."
        else:
            summary = "No semantic K2 matches met the configured score threshold."
        return [
            KnowledgeItem(
                id=f"{self.id}:{self.output_key}",
                key=self.output_key,
                provider=self.id,
                value=records,
                summary=summary,
                source=str(self.asset_path),
                record_count=len(records),
                truncated=truncated,
                provenance={
                    "asset_path": str(self.asset_path),
                    "corpus_count": len(entries),
                    "model": self.model_name,
                    "model_revision": self.model_revision,
                    "device": self.device,
                    "query_template": self.query_template,
                    "min_score": self.min_score,
                    "top_k": limit,
                },
            )
        ]

    def _is_truncated(
        self,
        *,
        entries: list[_SemanticEntry],
        matches: list[SemanticSearchResult],
        records: list[dict[str, Any]],
        limit: int,
    ) -> bool:
        if limit <= 0:
            return bool(entries)
        return len(matches) >= limit and len(records) == len(matches) and len(entries) > limit

    def _load_entries(self) -> list[_SemanticEntry]:
        if self._entries is not None:
            return self._entries
        data = json.loads(self.asset_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ProviderError(f"Semantic K2 asset must contain a JSON list: {self.asset_path}")
        entries: list[_SemanticEntry] = []
        for raw_entry in data:
            if not isinstance(raw_entry, dict):
                continue
            key = raw_entry.get(self.query_field)
            if key is None:
                continue
            entries.append(_SemanticEntry(key=str(key), answer=raw_entry.get(self.answer_field)))
        self._entries = entries
        return entries

    def _search(self, rendered_query: str, limit: int) -> list[SemanticSearchResult]:
        if limit <= 0:
            return []
        backend = self.backend_factory()
        if self._corpus_embeddings is None:
            corpus_texts = [entry.key for entry in self._load_entries()]
            self._corpus_embeddings = backend.embed_corpus(corpus_texts, batch_size=self.batch_size)
        return backend.search(
            rendered_query,
            self._corpus_embeddings,
            top_k=limit,
            batch_size=self.batch_size,
        )

    def _records_for_matches(
        self,
        entries: list[_SemanticEntry],
        matches: Iterable[SemanticSearchResult],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for rank, match in enumerate(matches, start=1):
            if match.index < 0 or match.index >= len(entries):
                continue
            if self.min_score is not None and match.score < self.min_score:
                continue
            entry = entries[match.index]
            records.append(
                {
                    "key": entry.key,
                    "answer": entry.answer,
                    "score": round(float(match.score), 6),
                    "rank": rank,
                }
            )
        return records

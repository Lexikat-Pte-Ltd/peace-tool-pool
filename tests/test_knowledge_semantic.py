import json

import pytest

from peace_tool_pool.knowledge import KnowledgeRequest
from peace_tool_pool.knowledge.errors import ProviderError
from peace_tool_pool.knowledge.providers.semantic_k2 import (
    SemanticK2Provider,
    SemanticSearchResult,
    resolve_embedding_device,
)


class FakeSemanticBackend:
    def __init__(self):
        self.corpus_texts = []
        self.queries = []

    def embed_corpus(self, corpus_texts, *, batch_size):
        self.corpus_texts.append((list(corpus_texts), batch_size))
        return list(corpus_texts)

    def search(self, query_text, corpus_embeddings, *, top_k, batch_size):
        self.queries.append((query_text, list(corpus_embeddings), top_k, batch_size))
        return [
            SemanticSearchResult(index=1, score=0.92),
            SemanticSearchResult(index=0, score=0.1),
        ]


def test_resolve_embedding_device_prefers_cuda_for_auto():
    assert resolve_embedding_device(None, cuda_available=lambda: True) == "cuda"
    assert resolve_embedding_device("auto", cuda_available=lambda: False) == "cpu"
    assert resolve_embedding_device("cpu", cuda_available=lambda: True) == "cpu"
    assert resolve_embedding_device("cuda:1", cuda_available=lambda: True) == "cuda:1"

    with pytest.raises(ProviderError):
        resolve_embedding_device("cuda", cuda_available=lambda: False)

    with pytest.raises(ProviderError):
        resolve_embedding_device("mps", cuda_available=lambda: True)


def test_semantic_provider_uses_prompt_template_fake_backend_and_score_filter(tmp_path):
    asset_path = tmp_path / "k2_usage.json"
    asset_path.write_text(
        json.dumps(
            [
                {"question": "What is a legend?", "answer": "A key for map symbols."},
                {"question": "What is a scale bar?", "answer": "A distance reference."},
            ]
        ),
        encoding="utf-8",
    )
    backend = FakeSemanticBackend()
    provider = SemanticK2Provider(
        provider_id="component_usage_knowledge",
        name="Component usage knowledge",
        output_key="component_usage_knowledge",
        asset_path=asset_path,
        query_field="question",
        answer_field="answer",
        query_template="What is the function of {query} in geologic maps?",
        backend_factory=lambda: backend,
        default_top_k=2,
        min_score=0.5,
        batch_size=16,
        model_name="fixture/model",
        model_revision="abc123",
        device="cuda",
    )

    item = provider.query(KnowledgeRequest(query_text="scale bar"))[0]

    assert backend.corpus_texts == [(["What is a legend?", "What is a scale bar?"], 16)]
    assert backend.queries == [
        (
            "What is the function of scale bar in geologic maps?",
            ["What is a legend?", "What is a scale bar?"],
            2,
            16,
        )
    ]
    assert item.key == "component_usage_knowledge"
    assert item.value == [
        {
            "key": "What is a scale bar?",
            "answer": "A distance reference.",
            "score": 0.92,
            "rank": 1,
        }
    ]
    assert item.record_count == 1
    assert item.truncated is False
    assert item.provenance["model"] == "fixture/model"
    assert item.provenance["device"] == "cuda"
    assert "model=" in provider.source_version()

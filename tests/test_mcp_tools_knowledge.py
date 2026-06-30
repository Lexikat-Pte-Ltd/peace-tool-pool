import json

from peace_tool_pool.knowledge import Bounds, KnowledgeConfig, KnowledgeService
from peace_tool_pool.knowledge.types import KnowledgeItem
from peace_tool_pool.mcp.adapter import GeomapMcpAdapter
from peace_tool_pool.mcp.resources import ResourceRegistry


class EchoProvider:
    id = "echo"
    name = "Echo"
    output_keys = ("echo",)
    version = "fixture-v1"
    last_warnings: list[str] = []

    def __init__(self, captured, secret_path):
        self.captured = captured
        self.secret_path = secret_path

    def supports(self, request):
        return True

    def validate_options(self, options):
        return dict(options)

    def query(self, request):
        self.captured["request"] = request
        return [
            KnowledgeItem(
                id="echo-1",
                key="echo",
                provider="echo",
                value={"legend_labels": list(request.legend_labels)},
                summary="echo summary",
                source=str(self.secret_path),
                record_count=1,
                provenance={"asset_path": str(self.secret_path)},
            )
        ]


def _adapter(tmp_path, monkeypatch, captured):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    cache_root.mkdir()
    monkeypatch.setenv("GEOMAP_DATA_ROOT", str(data_root))
    monkeypatch.setenv("GEOMAP_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("GEOMAP_MCP_ALLOWED_ROOTS", f"{data_root}:{cache_root}")
    registry = ResourceRegistry.from_env(base_dir=tmp_path)
    provider = EchoProvider(captured, tmp_path / "secret" / "asset.json")
    config = KnowledgeConfig(
        data_root=data_root,
        knowledge_root=tmp_path / "knowledge",
        cache_root=cache_root,
        write_cache=False,
    )
    service = KnowledgeService(config=config, providers=[provider])
    return GeomapMcpAdapter(registry=registry, knowledge_service_factory=lambda: service)


def test_query_knowledge_preserves_full_request_and_persists_bundle(tmp_path, monkeypatch):
    captured = {}
    adapter = _adapter(tmp_path, monkeypatch, captured)

    result = adapter.query_knowledge(
        bounds={"min_lon": -122, "min_lat": 37, "max_lon": -121, "max_lat": 38},
        legend_labels=["sandstone"],
        query_text="what is here?",
        include=["echo"],
        exclude=["unused"],
        max_records=5,
        max_records_by_provider={"echo": 3},
        provider_options={"echo": {"mode": "fixture"}},
    )

    request = captured["request"]
    assert isinstance(request.bounds, Bounds)
    assert request.legend_labels == ["sandstone"]
    assert request.query_text == "what is here?"
    assert request.include == ("echo",)
    assert request.exclude == ("unused",)
    assert request.max_records == 5
    assert request.max_records_by_provider == {"echo": 3}
    assert request.provider_options == {"echo": {"mode": "fixture"}}

    structured = result["structuredContent"]
    assert structured["bundle_uri"].startswith("geomap://bundles/")
    assert structured["items"][0]["source"] == "<redacted>"
    assert structured["items"][0]["provenance"]["asset_path"] == "<redacted>"
    assert str(tmp_path) not in json.dumps(result)

    bundle_resource = adapter.read_resource(structured["bundle_uri"])
    assert bundle_resource["mimeType"] == "application/json"
    assert json.loads(bundle_resource["text"])["items"][0]["id"] == "echo-1"

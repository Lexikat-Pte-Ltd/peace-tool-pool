import asyncio

import pytest

from peace_tool_pool.mcp.errors import McpToolError
from peace_tool_pool.mcp.server import create_server


class FailingAdapter:
    def list_capabilities(self):
        raise McpToolError(
            code="disallowed_path",
            message="Path is outside allowed roots.",
            trace_id="trace-fixture",
        )


def test_call_tool_preserves_typed_errors_over_server_handler():
    mcp_types = pytest.importorskip("mcp.types")

    async def invoke():
        server = create_server(adapter=FailingAdapter())
        handler = server.request_handlers[mcp_types.CallToolRequest]
        return await handler(
            mcp_types.CallToolRequest(
                method="tools/call",
                params={"name": "geomap_list_capabilities", "arguments": {}},
            )
        )

    result = asyncio.run(invoke()).model_dump()

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "disallowed_path"
    assert result["structuredContent"]["error"]["trace_id"] == "trace-fixture"
    assert result["structuredContent"]["code"] == "disallowed_path"

"""MCP SDK entry point for the geomap adapter.

The SDK import is intentionally lazy so importing ``peace_tool_pool.mcp`` or
installing the base package does not require the optional ``mcp`` extra.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from typing import Any, Mapping

from .adapter import GeomapMcpAdapter
from .errors import McpToolError
from .schemas import tool_definitions


def create_server(adapter: GeomapMcpAdapter | None = None) -> Any:
    try:
        from mcp.server import Server  # type: ignore[import-not-found]
        from mcp.server.lowlevel.helper_types import ReadResourceContents
        from mcp.types import (
            CallToolRequest,
            CallToolResult,
            ImageContent,
            ResourceLink,
            ServerResult,
            TextContent,
            Tool,
            ToolAnnotations,
        )
        from jsonschema import ValidationError, validate
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "The MCP server requires the optional 'mcp' extra. Install with "
            "`uv sync --extra mcp` or `pip install peace-tool-pool[mcp]`."
        ) from exc

    adapter = adapter or GeomapMcpAdapter()
    server = Server("peace-tool-pool")
    definitions = {definition["name"]: definition for definition in tool_definitions()}

    @server.list_tools()
    async def list_tools() -> list[Any]:
        tools = []
        for definition in definitions.values():
            tools.append(
                Tool(
                    name=definition["name"],
                    description=definition["description"],
                    inputSchema=definition["inputSchema"],
                    outputSchema=definition["outputSchema"],
                    annotations=ToolAnnotations(**definition["annotations"]),
                )
            )
        return tools

    async def call_tool(request: Any) -> Any:
        name = request.params.name
        arguments = request.params.arguments or {}
        definition = definitions.get(name)
        if definition is None:
            return server._make_error_result(f"Unknown geomap MCP tool: {name}")
        try:
            validate(instance=arguments, schema=definition["inputSchema"])
        except ValidationError as exc:
            return server._make_error_result(f"Input validation error: {exc.message}")
        try:
            result = _call_adapter(adapter, name, arguments)
        except McpToolError as exc:
            err = exc.to_dict()
            structured = {"isError": True, "error": err, **err, "text_summary": exc.message}
            return ServerResult(
                CallToolResult(
                    content=[TextContent(type="text", text=exc.message)],
                    structuredContent=structured,
                    isError=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - match MCP SDK tool-error behavior.
            return server._make_error_result(str(exc))
        content = []
        for item in result.get("content", []):
            if item.get("type") == "image":
                content.append(
                    ImageContent(
                        type="image",
                        data=item["data"],
                        mimeType=item["mimeType"],
                    )
                )
            else:
                content.append(TextContent(type="text", text=str(item.get("text", ""))))
        for link in result.get("structuredContent", {}).get("resource_links", []):
            content.append(
                ResourceLink(
                    type="resource_link",
                    uri=link["uri"],
                    name=link.get("name") or link["uri"],
                    mimeType=link.get("mimeType"),
                    description=link.get("description"),
                    size=link.get("size"),
                )
            )
        structured = result["structuredContent"]
        try:
            validate(instance=structured, schema=definition["outputSchema"])
        except ValidationError as exc:
            return server._make_error_result(f"Output validation error: {exc.message}")
        return ServerResult(
            CallToolResult(
                content=content,
                structuredContent=structured,
                isError=False,
            )
        )

    server.request_handlers[CallToolRequest] = call_tool

    @server.read_resource()
    async def read_resource(uri: Any) -> list[Any]:
        content = adapter.read_resource(str(uri))
        if "text" in content:
            return [
                ReadResourceContents(
                    content=content["text"],
                    mime_type=content.get("mimeType"),
                )
            ]
        return [
            ReadResourceContents(
                content=base64.b64decode(content["blob"]),
                mime_type=content.get("mimeType"),
            )
        ]

    return server


def main() -> None:
    try:
        server = create_server()
    except RuntimeError as exc:  # pragma: no cover - console-script behavior.
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    asyncio.run(_run_stdio(server))


async def _run_stdio(server: Any) -> None:
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _call_adapter(adapter: GeomapMcpAdapter, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    args = dict(arguments)
    if name == "geomap_list_capabilities":
        return adapter.list_capabilities()
    if name == "geomap_register_map":
        return adapter.register_map(args["path"])
    if name == "geomap_process_image":
        return adapter.process_image(map_id=args.get("map_id"), map_uri=args.get("map_uri"))
    if name == "geomap_georeference":
        return adapter.georeference(
            crs=args["crs"],
            gcps=args["gcps"],
            pixel_extent=args.get("pixel_extent"),
            map_id=args.get("map_id"),
            map_uri=args.get("map_uri"),
            main_map_artifact_uri=args.get("main_map_artifact_uri"),
        )
    if name == "geomap_query_knowledge":
        return adapter.query_knowledge(
            bounds=args.get("bounds"),
            legend_labels=args.get("legend_labels"),
            query_text=args.get("query_text"),
            include=args.get("include"),
            exclude=args.get("exclude"),
            max_records=args.get("max_records"),
            max_records_by_provider=args.get("max_records_by_provider"),
            provider_options=args.get("provider_options"),
        )
    if name == "geomap_query_map":
        return adapter.query_map(
            map_id=args.get("map_id"),
            map_uri=args.get("map_uri"),
            metadata=args.get("metadata"),
            question=args.get("question"),
            include=args.get("include"),
            exclude=args.get("exclude"),
            max_records=args.get("max_records"),
            max_records_by_provider=args.get("max_records_by_provider"),
            provider_options=args.get("provider_options"),
            bounds=args.get("bounds"),
            legend_labels=args.get("legend_labels"),
            query_text=args.get("query_text"),
        )
    if name == "geomap_enrich_legend":
        return adapter.enrich_legend(args["label"])
    if name == "geomap_render_knowledge_overlay":
        return adapter.render_knowledge_overlay(
            map_id=args.get("map_id"),
            map_uri=args.get("map_uri"),
            bundle_uri=args.get("bundle_uri"),
            bundle=args.get("bundle"),
            georef=args.get("georef"),
        )
    raise ValueError(f"Unknown geomap MCP tool: {name}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""Preview generation helpers for agent-facing image content."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from .resources import mime_type_for_path


def make_inline_preview(
    path: str | Path,
    *,
    artifact_uri: str,
    max_long_edge: int = 1536,
    max_encoded_bytes: int = 1_000_000,
) -> dict[str, Any] | None:
    """Return MCP image content plus metadata for a bounded preview.

    Pillow is preferred and is part of the MCP extra. If unavailable, small files
    are still inlined unchanged so lightweight installs can expose useful previews.
    """

    source = Path(path)
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        raw = source.read_bytes()
        if len(raw) > max_encoded_bytes:
            return None
        mime_type = mime_type_for_path(source)
        return _preview_payload(
            raw,
            artifact_uri=artifact_uri,
            mime_type=mime_type,
            source_width=None,
            source_height=None,
            width=None,
            height=None,
            downsampled=False,
        )

    with Image.open(source) as image:
        source_width, source_height = image.size
        preview = image.copy()
        downsampled = False
        long_edge = max(source_width, source_height)
        if long_edge > max_long_edge:
            preview.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
            downsampled = True
        if preview.mode not in {"RGB", "RGBA"}:
            preview = preview.convert("RGBA" if "A" in preview.getbands() else "RGB")
        encoded = _encode_png(preview)
        while len(encoded) > max_encoded_bytes and max(preview.size) > 256:
            next_size = (max(1, int(preview.width * 0.8)), max(1, int(preview.height * 0.8)))
            preview = preview.resize(next_size, Image.Resampling.LANCZOS)
            downsampled = True
            encoded = _encode_png(preview)
        if len(encoded) > max_encoded_bytes:
            return None
        return _preview_payload(
            encoded,
            artifact_uri=artifact_uri,
            mime_type="image/png",
            source_width=source_width,
            source_height=source_height,
            width=preview.width,
            height=preview.height,
            downsampled=downsampled,
        )


def _encode_png(image: Any) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _preview_payload(
    payload: bytes,
    *,
    artifact_uri: str,
    mime_type: str,
    source_width: int | None,
    source_height: int | None,
    width: int | None,
    height: int | None,
    downsampled: bool,
) -> dict[str, Any]:
    data = base64.b64encode(payload).decode("ascii")
    metadata = {
        "artifact_uri": artifact_uri,
        "mime_type": mime_type,
        "source_width": source_width,
        "source_height": source_height,
        "width": width,
        "height": height,
        "downsampled": downsampled,
        "encoded_bytes": len(payload),
    }
    return {
        "content": {"type": "image", "data": data, "mimeType": mime_type},
        "metadata": metadata,
    }

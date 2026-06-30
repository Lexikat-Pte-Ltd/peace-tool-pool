"""Private registry for model-safe ``geomap://`` resources."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import mimetypes
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import McpToolError


REGISTRY_SCHEMA_VERSION = "mcp-registry/v1"
IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/webp",
    "image/gif",
}
TEXT_MIME_TYPES = {"application/json", "image/svg+xml", "text/plain"}


def write_json_atomic(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
    ) as file_obj:
        json.dump(data, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
        temp_name = file_obj.name
    os.replace(temp_name, target)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _path_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(os.pathsep) if part.strip()]


def mime_type_for_path(path: str | Path) -> str:
    source = Path(path)
    guessed, _ = mimetypes.guess_type(source.name)
    if guessed == "image/jpg":
        return "image/jpeg"
    if guessed:
        return guessed
    suffix = source.suffix.lower()
    if suffix == ".svg":
        return "image/svg+xml"
    if suffix == ".json":
        return "application/json"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class ResourceRegistry:
    """Map private local paths to stable, redacted ``geomap://`` URIs."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        cache_root: str | Path,
        allowed_roots: list[str | Path] | tuple[str | Path, ...],
        registry_path: str | Path | None = None,
        max_source_bytes: int = 200 * 1024 * 1024,
        max_resource_read_bytes: int = 50 * 1024 * 1024,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.allowed_roots = [Path(root).expanduser().resolve() for root in allowed_roots]
        self.registry_path = Path(registry_path or self.cache_root / "mcp" / "v1" / "registry.json")
        self.max_source_bytes = int(max_source_bytes)
        self.max_resource_read_bytes = int(max_resource_read_bytes)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._defer_depth = 0
        self._dirty = False
        self._data = self._load()

    @classmethod
    def from_env(cls, base_dir: str | Path | None = None) -> "ResourceRegistry":
        root = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        data_root = _resolve_path(os.getenv("GEOMAP_DATA_ROOT", "./data"), root)
        cache_root = _resolve_path(os.getenv("GEOMAP_CACHE_ROOT", ".cache"), root)
        raw_allowed = _path_list(os.getenv("GEOMAP_MCP_ALLOWED_ROOTS"))
        allowed = [_resolve_path(value, root) for value in raw_allowed]
        if not allowed:
            allowed = [data_root, cache_root]
        return cls(data_root=data_root, cache_root=cache_root, allowed_roots=allowed)

    @property
    def limits(self) -> dict[str, int]:
        return {
            "max_source_bytes": self.max_source_bytes,
            "max_resource_read_bytes": self.max_resource_read_bytes,
        }

    def allowed_root_labels(self) -> list[dict[str, str]]:
        return [{"label": f"root_{index + 1}"} for index, _ in enumerate(self.allowed_roots)]

    def register_map(self, path: str | Path) -> dict[str, Any]:
        source = self._validate_existing_file(path, error_code="disallowed_path")
        mime_type = mime_type_for_path(source)
        if mime_type not in IMAGE_MIME_TYPES:
            raise McpToolError("unsupported_media", f"Unsupported map media type: {mime_type}")
        size = source.stat().st_size
        if size > self.max_source_bytes:
            raise McpToolError("oversize_image", "Source image exceeds the configured byte limit.")
        source_key = str(source)
        for map_id, entry in self._data["maps"].items():
            if entry.get("source_path") == source_key:
                return self._public_map(map_id, entry)
        map_id = uuid.uuid4().hex
        entry = {
            "source_path": source_key,
            "source_mime_type": mime_type,
            "created_at": _utc_now(),
            "processing": None,
            "georef": None,
            "bundles": [],
        }
        self._data["maps"][map_id] = entry
        self._save()
        return self._public_map(map_id, entry)

    def register_artifact(
        self,
        path: str | Path,
        *,
        role: str,
        stage: str,
        map_id: str | None = None,
        bbox: list[int] | tuple[int, int, int, int] | None = None,
        label: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        artifact_path = self._validate_existing_file(path, error_code="disallowed_path")
        canonical = str(artifact_path)
        for artifact_id, entry in self._data["artifacts"].items():
            if entry.get("path") == canonical:
                self._merge_artifact_metadata(entry, role, stage, map_id, bbox, label)
                self._save()
                return self._public_artifact(artifact_id, entry)
        artifact_id = uuid.uuid4().hex
        detected_mime = mime_type or mime_type_for_path(artifact_path)
        suffix = _resource_suffix(artifact_path, detected_mime)
        entry = {
            "path": canonical,
            "uri": f"geomap://artifacts/{artifact_id}{suffix}",
            "mime_type": detected_mime,
            "role": str(role),
            "stage": str(stage),
            "map_id": map_id,
            "bbox": list(bbox) if bbox is not None else None,
            "label": label,
            "created_at": _utc_now(),
        }
        self._data["artifacts"][artifact_id] = entry
        self._save()
        return self._public_artifact(artifact_id, entry)

    def register_bundle(
        self,
        data: Mapping[str, Any],
        *,
        map_id: str | None = None,
    ) -> dict[str, Any]:
        bundle_id = uuid.uuid4().hex
        path = self.cache_root / "mcp" / "v1" / "bundles" / f"{bundle_id}.json"
        write_json_atomic(path, data)
        entry = {
            "path": str(path.resolve()),
            "uri": f"geomap://bundles/{bundle_id}.json",
            "mime_type": "application/json",
            "map_id": map_id,
            "created_at": _utc_now(),
        }
        self._data["bundles"][bundle_id] = entry
        if map_id and map_id in self._data["maps"]:
            self._data["maps"][map_id].setdefault("bundles", []).append(bundle_id)
        self._save()
        return self._public_json_resource(bundle_id, entry)

    def register_overlay(
        self,
        path: str | Path,
        *,
        map_id: str | None = None,
        role: str = "knowledge_overlay",
    ) -> dict[str, Any]:
        overlay_path = self._validate_existing_file(path, error_code="disallowed_path")
        overlay_id = uuid.uuid4().hex
        mime_type = mime_type_for_path(overlay_path)
        suffix = _resource_suffix(overlay_path, mime_type)
        entry = {
            "path": str(overlay_path),
            "uri": f"geomap://overlays/{overlay_id}{suffix}",
            "mime_type": mime_type,
            "map_id": map_id,
            "role": role,
            "created_at": _utc_now(),
        }
        self._data["overlays"][overlay_id] = entry
        self._save()
        return self._public_json_resource(overlay_id, entry)

    def set_map_processing(self, map_id: str, processing: Mapping[str, Any]) -> None:
        self._require_map(map_id)["processing"] = dict(processing)
        self._save()

    def get_map_processing(self, map_id: str) -> dict[str, Any] | None:
        processing = self._require_map(map_id).get("processing")
        return dict(processing) if isinstance(processing, Mapping) else None

    def set_map_georef(self, map_id: str, georef: Mapping[str, Any]) -> dict[str, Any]:
        self._require_map(map_id)
        path = self.cache_root / "mcp" / "v1" / "maps" / map_id / "georef.json"
        write_json_atomic(path, georef)
        entry = {
            "path": str(path.resolve()),
            "uri": f"geomap://maps/{map_id}/georef.json",
            "mime_type": "application/json",
            "created_at": _utc_now(),
        }
        self._data["maps"][map_id]["georef"] = entry
        self._save()
        return {"uri": entry["uri"], "mime_type": entry["mime_type"], "source_path_redacted": True}

    def get_map_georef(self, map_id: str) -> dict[str, Any] | None:
        entry = self._require_map(map_id).get("georef")
        if not isinstance(entry, Mapping):
            return None
        path = Path(str(entry["path"]))
        if not path.exists():
            raise McpToolError("artifact_not_found", "Stored georef resource is missing.")
        return json.loads(path.read_text(encoding="utf-8"))

    def source_path(self, map_id: str) -> Path:
        return Path(str(self._require_map(map_id)["source_path"]))

    def map_public(self, map_id: str) -> dict[str, Any]:
        return self._public_map(map_id, self._require_map(map_id))

    def map_id_from_uri(self, uri: str) -> str:
        if not uri.startswith("geomap://maps/"):
            raise McpToolError("artifact_not_found", f"Unsupported map URI: {uri}")
        suffix = uri.removeprefix("geomap://maps/")
        map_id = suffix.split("/", 1)[0]
        self._require_map(map_id)
        return map_id

    def artifact_entry(self, uri: str) -> dict[str, Any]:
        kind, entry = self._entry_for_uri(uri)
        if kind != "artifact":
            raise McpToolError("artifact_not_found", f"Resource is not an artifact: {uri}")
        return dict(entry)

    def read_resource(self, uri: str) -> dict[str, Any]:
        _, entry = self._entry_for_uri(uri)
        path = Path(str(entry["path"]))
        if not path.exists():
            raise McpToolError("artifact_not_found", "Resource backing file is missing.")
        resolved = path.resolve()
        if not self._inside_allowed_root(resolved):
            raise McpToolError("disallowed_path", "Resource backing file is outside allowed roots.")
        size = resolved.stat().st_size
        if size > self.max_resource_read_bytes:
            raise McpToolError("oversize_image", "Resource exceeds the configured read byte limit.")
        mime_type = str(entry.get("mime_type") or mime_type_for_path(resolved))
        content: dict[str, Any] = {"uri": uri, "mimeType": mime_type, "size": size}
        if mime_type in TEXT_MIME_TYPES or mime_type.startswith("text/"):
            content["text"] = resolved.read_text(encoding="utf-8")
        else:
            content["blob"] = base64.b64encode(resolved.read_bytes()).decode("ascii")
        return content

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return _empty_registry()
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return _empty_registry()
        data.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
        data.setdefault("maps", {})
        data.setdefault("artifacts", {})
        data.setdefault("bundles", {})
        data.setdefault("overlays", {})
        return data

    def _save(self) -> None:
        if self._defer_depth > 0:
            self._dirty = True
            return
        self._write()

    def _write(self) -> None:
        with _registry_file_lock(self.registry_path):
            self._data = _merge_registry_data(self._load(), self._data)
            write_json_atomic(self.registry_path, self._data)

    @contextmanager
    def deferred_save(self):
        """Coalesce nested registry mutations into one locked merge-write.

        A multi-artifact tool call (e.g. ``process_image``) registers many
        artifacts; without this each registration would take the file lock and
        re-merge the whole registry. Nested scopes are reference counted so only
        the outermost scope flushes, and only if something was actually written.
        """

        self._defer_depth += 1
        try:
            yield
        finally:
            self._defer_depth -= 1
            if self._defer_depth == 0 and self._dirty:
                self._dirty = False
                self._write()

    def _validate_existing_file(self, path: str | Path, *, error_code: str) -> Path:
        source = Path(path).expanduser()
        try:
            resolved = source.resolve(strict=True)
        except FileNotFoundError as exc:
            raise McpToolError("artifact_not_found", f"File does not exist: {source.name}") from exc
        if not resolved.is_file():
            raise McpToolError(error_code, "Path is not a regular file.")
        if not self._inside_allowed_root(resolved):
            raise McpToolError(error_code, "Path is outside configured MCP allowed roots.")
        return resolved

    def _inside_allowed_root(self, path: Path) -> bool:
        return any(_is_relative_to(path, root) for root in self.allowed_roots)

    def _require_map(self, map_id: str) -> dict[str, Any]:
        try:
            entry = self._data["maps"][map_id]
        except KeyError as exc:
            raise McpToolError("artifact_not_found", f"Unknown map id: {map_id}") from exc
        return entry

    def _entry_for_uri(self, uri: str) -> tuple[str, Mapping[str, Any]]:
        if uri.startswith("geomap://maps/"):
            suffix = uri.removeprefix("geomap://maps/")
            map_id, _, tail = suffix.partition("/")
            entry = self._require_map(map_id)
            if tail == "source":
                return "source", {
                    "path": entry["source_path"],
                    "mime_type": entry.get("source_mime_type", "image/png"),
                }
            if tail == "georef.json" and isinstance(entry.get("georef"), Mapping):
                return "georef", entry["georef"]
            if tail == "":
                payload = self._public_map(map_id, entry)
                path = self.cache_root / "mcp" / "v1" / "maps" / map_id / "map.json"
                write_json_atomic(path, payload)
                return "map", {"path": str(path), "mime_type": "application/json"}
        for kind in ("artifacts", "bundles", "overlays"):
            for entry in self._data[kind].values():
                if entry.get("uri") == uri:
                    return kind[:-1], entry
        raise McpToolError("artifact_not_found", f"Unknown geomap resource URI: {uri}")

    def _public_map(self, map_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        source = Path(str(entry["source_path"]))
        return {
            "map_id": map_id,
            "map_uri": f"geomap://maps/{map_id}",
            "source_uri": f"geomap://maps/{map_id}/source",
            "mime_type": entry.get("source_mime_type", mime_type_for_path(source)),
            "source_path_redacted": True,
        }

    def _public_artifact(self, artifact_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = {
            "uri": entry["uri"],
            "role": entry.get("role"),
            "stage": entry.get("stage"),
            "mime_type": entry.get("mime_type"),
            "source_path_redacted": True,
        }
        if entry.get("bbox") is not None:
            data["bbox"] = list(entry["bbox"])
        if entry.get("label") is not None:
            data["label"] = entry["label"]
        if entry.get("map_id") is not None:
            data["map_id"] = entry["map_id"]
        path = Path(str(entry["path"]))
        if path.exists():
            data["size"] = path.stat().st_size
        data["artifact_id"] = artifact_id
        return data

    def _public_json_resource(self, resource_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        data = {
            "id": resource_id,
            "uri": entry["uri"],
            "mime_type": entry.get("mime_type"),
            "source_path_redacted": True,
        }
        if entry.get("map_id") is not None:
            data["map_id"] = entry["map_id"]
        return data

    def _merge_artifact_metadata(
        self,
        entry: dict[str, Any],
        role: str,
        stage: str,
        map_id: str | None,
        bbox: list[int] | tuple[int, int, int, int] | None,
        label: str | None,
    ) -> None:
        entry["role"] = entry.get("role") or str(role)
        entry["stage"] = entry.get("stage") or str(stage)
        entry["map_id"] = entry.get("map_id") or map_id
        entry["bbox"] = entry.get("bbox") or (list(bbox) if bbox is not None else None)
        entry["label"] = entry.get("label") or label


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "maps": {},
        "artifacts": {},
        "bundles": {},
        "overlays": {},
    }


@contextmanager
def _registry_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            import fcntl
        except ModuleNotFoundError:  # pragma: no cover - non-POSIX fallback.
            yield
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _merge_registry_data(existing: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    merged = _empty_registry()
    merged["schema_version"] = str(
        current.get("schema_version") or existing.get("schema_version") or REGISTRY_SCHEMA_VERSION
    )
    for key in ("artifacts", "bundles", "overlays"):
        merged[key] = {**dict(existing.get(key, {})), **dict(current.get(key, {}))}
    existing_maps = dict(existing.get("maps", {}))
    current_maps = dict(current.get("maps", {}))
    merged["maps"] = {**existing_maps, **current_maps}
    for map_id in existing_maps.keys() & current_maps.keys():
        merged["maps"][map_id] = _merge_map_entry(existing_maps[map_id], current_maps[map_id])
    return merged


def _merge_map_entry(existing: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**dict(existing), **dict(current)}
    for key in ("processing", "georef"):
        if merged.get(key) is None and existing.get(key) is not None:
            merged[key] = existing[key]
    bundles = []
    for bundle_id in [*existing.get("bundles", []), *current.get("bundles", [])]:
        if bundle_id not in bundles:
            bundles.append(bundle_id)
    merged["bundles"] = bundles
    return merged


def _resource_suffix(path: Path, mime_type: str) -> str:
    if path.suffix:
        return path.suffix.lower()
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/svg+xml":
        return ".svg"
    if mime_type == "application/json":
        return ".json"
    return ""

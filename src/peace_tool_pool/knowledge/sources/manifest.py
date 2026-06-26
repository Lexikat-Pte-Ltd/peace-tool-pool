"""Source manifest contract for normalized knowledge mirrors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..cache import stable_hash, write_json_atomic
from ..errors import SourceManifestError


SOURCE_MANIFEST_SCHEMA_VERSION = "knowledge-source/v1"


@dataclass
class SourceManifest:
    source_id: str
    family: str
    retrieved_at: str
    source_version: str
    normalizer_version: str
    source_url: str
    request: dict[str, Any]
    record_count: int
    normalized_sha256: str
    license: str | None = None
    citation: str | None = None
    attribution: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    manifest_hash_material_version: str = "1"
    schema_version: str = SOURCE_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "family": self.family,
            "retrieved_at": self.retrieved_at,
            "source_version": self.source_version,
            "normalizer_version": self.normalizer_version,
            "source_url": self.source_url,
            "request": dict(self.request),
            "record_count": int(self.record_count),
            "normalized_sha256": self.normalized_sha256,
            "manifest_hash_material_version": self.manifest_hash_material_version,
            "license": self.license,
            "citation": self.citation,
            "attribution": self.attribution,
            "coverage": dict(self.coverage),
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceManifest":
        if data.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise SourceManifestError(
                f"Unsupported source manifest schema version: {data.get('schema_version')!r}"
            )
        try:
            return cls(
                source_id=str(data["source_id"]),
                family=str(data["family"]),
                retrieved_at=str(data["retrieved_at"]),
                source_version=str(data["source_version"]),
                normalizer_version=str(data["normalizer_version"]),
                source_url=str(data["source_url"]),
                request=dict(data.get("request") or {}),
                record_count=int(data.get("record_count", 0)),
                normalized_sha256=str(data["normalized_sha256"]),
                license=data.get("license"),
                citation=data.get("citation"),
                attribution=data.get("attribution"),
                coverage=dict(data.get("coverage") or {}),
                artifacts={str(k): str(v) for k, v in dict(data.get("artifacts") or {}).items()},
                manifest_hash_material_version=str(
                    data.get("manifest_hash_material_version", "1")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceManifestError(f"Malformed source manifest: {exc}") from exc

    @classmethod
    def from_path(cls, path: str | Path) -> "SourceManifest":
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceManifestError(f"Could not read source manifest {source}: {exc}") from exc
        return cls.from_dict(data)

    def write(self, path: str | Path) -> None:
        write_json_atomic(path, self.to_dict())

    def hash_material(self) -> dict[str, Any]:
        material = self.to_dict()
        material.pop("retrieved_at", None)
        return material

    def stable_hash(self) -> str:
        return stable_hash(self.hash_material())

    def normalized_artifact_path(self, manifest_path: str | Path) -> Path:
        relative = self.artifacts.get("normalized")
        if relative is None:
            if self.family == "earthquake_events":
                relative = "normalized/earthquakes.csv"
            elif self.family == "active_faults":
                relative = "normalized/faults.geojson"
            else:
                raise SourceManifestError(
                    f"Source manifest {self.source_id!r} has no normalized artifact path."
                )
        return Path(manifest_path).parent / relative


def find_latest_manifest(
    sources_root: str | Path,
    source_id: str,
    preferred_version: str | None = None,
) -> Path | None:
    root = Path(sources_root) / source_id
    if preferred_version:
        candidate = root / preferred_version / "manifest.json"
        return candidate if candidate.exists() else None
    default_candidate = root / "default" / "manifest.json"
    if default_candidate.exists():
        return default_candidate
    candidates = sorted(root.glob("*/manifest.json"), key=lambda path: path.parent.name)
    if not candidates:
        return None
    return candidates[-1]

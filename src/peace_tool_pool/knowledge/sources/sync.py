"""Command-line entrypoint for syncing knowledge source mirrors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..bounds import Bounds
from ..config import KnowledgeConfig
from ..errors import SourceRegistryError
from .gem_faults import GemActiveFaultSourceAdapter
from .registry import default_source_registry
from .usgs_events import UsgsFdsnEventAdapter


def sync_source(
    source_id: str,
    output_root: str | Path,
    profile: dict[str, Any] | None = None,
    bounds: Bounds | None = None,
    version: str | None = None,
) -> Path:
    registry = default_source_registry()
    definition = registry.get(source_id)
    validated_profile = definition.validate_profile(profile or {})
    if source_id == "usgs_fdsn_events":
        UsgsFdsnEventAdapter().sync(
            output_root,
            profile=validated_profile,
            bounds=bounds,
            version=version or "default",
        )
        return Path(output_root) / source_id / (version or "default") / "manifest.json"
    if source_id == "gem_global_active_faults":
        source_version = str(version or validated_profile["source_version"]).replace("/", "_")
        GemActiveFaultSourceAdapter().sync(
            output_root,
            profile=validated_profile,
            version=version,
        )
        return Path(output_root) / source_id / source_version / "manifest.json"
    raise SourceRegistryError(f"No sync adapter is implemented for source {source_id!r}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync a normalized knowledge source mirror.")
    parser.add_argument("source_id", choices=("usgs_fdsn_events", "gem_global_active_faults"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-lon", type=float, default=None)
    parser.add_argument("--min-lat", type=float, default=None)
    parser.add_argument("--max-lon", type=float, default=None)
    parser.add_argument("--max-lat", type=float, default=None)
    args = parser.parse_args(argv)

    config = KnowledgeConfig.from_env()
    output_root = args.output_root or config.knowledge_sources_root
    if output_root is None:
        raise SystemExit("No output root configured for knowledge source mirrors.")
    profile = _read_profile(args.profile_json)
    bounds = _bounds_from_args(args)
    manifest_path = sync_source(
        args.source_id,
        output_root,
        profile=profile,
        bounds=bounds,
        version=args.version,
    )
    print(manifest_path)
    return 0


def _read_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bounds_from_args(args: argparse.Namespace) -> Bounds | None:
    values = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise SystemExit("Bounds require all of --min-lon, --min-lat, --max-lon, --max-lat.")
    return Bounds(args.min_lon, args.min_lat, args.max_lon, args.max_lat)


if __name__ == "__main__":  # pragma: no cover - exercised through manual operator use.
    raise SystemExit(main())

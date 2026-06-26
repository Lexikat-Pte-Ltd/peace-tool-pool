"""Registry of authoritative knowledge source definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import SourceRegistryError


USGS_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "geojson",
    "eventtype": "earthquake",
    "starttime": "1970-01-01",
    "minmagnitude": 4.5,
    "orderby": "time",
    "limit": 20000,
    "offset": 1,
}

GEM_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "geojson",
    "source_version": "zenodo:3376300",
    "source_url": "https://doi.org/10.5281/zenodo.3376300",
}

# Ontario Geological Survey Mineral Deposit Inventory (MDI), served as an ArcGIS
# REST FeatureServer queryable by bbox envelope. PROTOTYPE ENDPOINT ONLY: this
# public layer is named ``MDI_March_01_2013`` (a third-party-republished March 1
# 2013 snapshot, ~19k points), not the authoritative OGS feed. The production
# source is the Ontario Ministry of Mines "Ontario Mineral Inventory" (OMI),
# continuously updated, distributed via the GeologyOntario hub / data.ontario.ca
# under OGL-Ontario. See the validation section of
# docs/design/generalized-seismic-knowledge-sources.md for the promotion steps.
OGS_MDI_ENDPOINT = (
    "https://services2.arcgis.com/NzAqpeLbvbvoP3qh/arcgis/rest/services/"
    "Ontario_Mineral_Deposit_Inventory/FeatureServer/0"
)
OGS_MDI_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "geojson",
    "endpoint": OGS_MDI_ENDPOINT,
    "region": "Ontario",
    # Coverage extent (lon/lat) used for region routing: the Province of Ontario.
    "coverage_bounds": [-95.2, 41.6, -74.3, 56.9],
}


@dataclass
class SourceDefinition:
    id: str
    family: str
    authority: str
    homepage_url: str
    default_format: str
    license: str | None = None
    citation: str | None = None
    attribution: str | None = None
    coverage_notes: list[str] = field(default_factory=list)
    default_profile: dict[str, Any] = field(default_factory=dict)

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = {**self.default_profile, **dict(profile or {})}
        if self.id == "usgs_fdsn_events":
            return _validate_usgs_profile(merged)
        if self.id == "gem_global_active_faults":
            return _validate_gem_profile(merged)
        if self.id == "ontario_mineral_deposit_inventory":
            return _validate_ogs_mdi_profile(merged)
        return merged


class SourceRegistry:
    def __init__(self, definitions: list[SourceDefinition]):
        self._definitions = {definition.id: definition for definition in definitions}

    def get(self, source_id: str) -> SourceDefinition:
        try:
            return self._definitions[source_id]
        except KeyError as exc:
            raise SourceRegistryError(f"Unknown knowledge source: {source_id!r}") from exc

    def resolve(
        self,
        *,
        family: str,
        source_id: str | None = None,
        bounds: Any | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> list[SourceDefinition]:
        del bounds, options
        if source_id is not None:
            definition = self.get(source_id)
            if definition.family != family:
                raise SourceRegistryError(
                    f"Knowledge source {source_id!r} has family {definition.family!r}, "
                    f"not {family!r}."
                )
            return [definition]
        selection = [definition for definition in self._definitions.values() if definition.family == family]
        if not selection:
            raise SourceRegistryError(f"No knowledge source registered for family {family!r}.")
        return selection


def default_source_registry() -> SourceRegistry:
    return SourceRegistry(
        [
            SourceDefinition(
                id="usgs_fdsn_events",
                family="earthquake_events",
                authority="USGS Earthquake Hazards Program",
                homepage_url="https://earthquake.usgs.gov/fdsnws/event/1/",
                default_format="geojson",
                license="See USGS source policy",
                citation="USGS FDSN Event API",
                attribution="USGS Earthquake Hazards Program FDSN Event API",
                coverage_notes=[],
                default_profile=dict(USGS_DEFAULT_PROFILE),
            ),
            SourceDefinition(
                id="gem_global_active_faults",
                family="active_faults",
                authority="GEM Foundation",
                homepage_url="https://github.com/GEMScienceTools/gem-global-active-faults",
                default_format="geojson",
                license="CC BY-SA 4.0 (Creative Commons Attribution Share Alike 4.0 International)",
                citation="GEM Global Active Faults Database",
                attribution=(
                    "GEM Global Active Faults Database (CC BY-SA 4.0) - "
                    "https://doi.org/10.5281/zenodo.3376300"
                ),
                coverage_notes=[
                    "GEM covers most deforming continental regions but has documented gaps."
                ],
                default_profile=dict(GEM_DEFAULT_PROFILE),
            ),
            SourceDefinition(
                id="ontario_mineral_deposit_inventory",
                family="mineral_occurrences",
                authority="Ontario Geological Survey",
                homepage_url="https://www.hub.geologyontario.mines.gov.on.ca/",
                default_format="geojson",
                license="Open Government Licence - Ontario",
                citation="Ontario Geological Survey, Mineral Deposit Inventory (MDI)",
                attribution=(
                    "Mineral Deposit Inventory (c) King's Printer for Ontario, "
                    "Ontario Geological Survey (Open Government Licence - Ontario)"
                ),
                coverage_notes=[
                    "Coverage is the Province of Ontario only.",
                    "The public ArcGIS FeatureServer is a 2013 MDI snapshot; the "
                    "authoritative continuously-updated inventory is distributed via "
                    "data.ontario.ca / GeologyOntario.",
                ],
                default_profile=dict(OGS_MDI_DEFAULT_PROFILE),
            ),
        ]
    )


def source_attribution(source_id: str) -> dict[str, str | None]:
    """Return ``{license, citation, attribution}`` for a source id.

    The registry is the single source of truth for attribution metadata. Providers
    use this to attribute data even when reading a legacy bundled asset, because
    that asset is still derived from the registered upstream source (and licences
    such as GEM's CC BY-SA require attribution wherever the data is surfaced).
    Returns ``None`` values for unregistered ids rather than raising.
    """
    try:
        definition = default_source_registry().get(source_id)
    except SourceRegistryError:
        return {"license": None, "citation": None, "attribution": None}
    return {
        "license": definition.license,
        "citation": definition.citation,
        "attribution": definition.attribution,
    }


def _validate_usgs_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "format",
        "eventtype",
        "starttime",
        "endtime",
        "updatedafter",
        "minmagnitude",
        "maxmagnitude",
        "mindepth",
        "maxdepth",
        "reviewstatus",
        "catalog",
        "contributor",
        "orderby",
        "limit",
        "offset",
    }
    unknown = set(profile) - allowed
    if unknown:
        raise SourceRegistryError(f"Unknown USGS FDSN profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "geojson"))
    if validated["format"] not in {"geojson", "csv"}:
        raise SourceRegistryError("USGS FDSN source only supports geojson or csv profiles.")
    validated["eventtype"] = str(validated.get("eventtype", "earthquake"))
    if validated["eventtype"] != "earthquake":
        raise SourceRegistryError("USGS earthquake source requires eventtype='earthquake'.")
    for key in ("minmagnitude", "maxmagnitude", "mindepth", "maxdepth"):
        if key in validated and validated[key] not in (None, ""):
            validated[key] = float(validated[key])
    for key in ("limit", "offset"):
        if key in validated and validated[key] not in (None, ""):
            validated[key] = int(validated[key])
    if int(validated.get("limit", 20000)) > 20000:
        raise SourceRegistryError("USGS FDSN Event API rejects limit values above 20000.")
    if int(validated.get("offset", 1)) < 1:
        raise SourceRegistryError("USGS FDSN offset is one-based and must be >= 1.")
    return validated


def _validate_gem_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"format", "source_version", "source_url", "local_path", "sha256"}
    unknown = set(profile) - allowed
    if unknown:
        raise SourceRegistryError(f"Unknown GEM active-fault profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "geojson"))
    if validated["format"] not in {"geojson", "gpkg", "kml", "gmt", "shapefile"}:
        raise SourceRegistryError("Unsupported GEM active-fault profile format.")
    if not validated.get("source_version"):
        raise SourceRegistryError("GEM active-fault profiles must pin source_version.")
    if not validated.get("source_url") and not validated.get("local_path"):
        raise SourceRegistryError("GEM active-fault profiles require source_url or local_path.")
    return validated


def _validate_ogs_mdi_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"format", "endpoint", "region", "coverage_bounds"}
    unknown = set(profile) - allowed
    if unknown:
        raise SourceRegistryError(f"Unknown OGS MDI profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "geojson"))
    if validated["format"] != "geojson":
        raise SourceRegistryError("OGS MDI source only supports the geojson format.")
    if not validated.get("endpoint"):
        raise SourceRegistryError("OGS MDI profiles require an ArcGIS REST endpoint.")
    coverage = validated.get("coverage_bounds")
    if coverage is not None and len(list(coverage)) != 4:
        raise SourceRegistryError("OGS MDI coverage_bounds must be [min_lon, min_lat, max_lon, max_lat].")
    return validated

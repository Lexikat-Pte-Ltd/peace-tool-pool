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

EMSC_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "json",
    "eventtype": "earthquake",
    "orderby": "time",
    "limit": 20000,
    "offset": 1,
}

GEM_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "geojson",
    "source_version": "zenodo:3376300",
    "source_url": "https://doi.org/10.5281/zenodo.3376300",
}

DISS_ENDPOINT = "https://services.seismofaults.eu/DISS331/ows"
DISS_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "geojson",
    "endpoint": DISS_ENDPOINT,
    "source_version": "DISS3.3.1",
    "type_names": [
        "DISS331:iss331",
        "DISS331:csspln331",
        "DISS331:dss331",
        "DISS331:subdzon331",
    ],
    "srs_name": "CRS:84",
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
}

SIGEOM_ENDPOINT = "https://servicesvectoriels.atlas.gouv.qc.ca/IDS_SGM_WFS/service.svc/get"
SIGEOM_DEFAULT_PROFILE: dict[str, Any] = {
    "format": "geojson",
    "endpoint": SIGEOM_ENDPOINT,
    "region": "Quebec",
    "feature_types": ["SGM:Substances_metalliques"],
    "srs_name": "CRS:84",
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
    selection_rank: int = 100
    coverage_bounds: tuple[float, float, float, float] | None = None

    def validate_profile(self, profile: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = {**self.default_profile, **dict(profile or {})}
        if self.id == "usgs_fdsn_events":
            return _validate_usgs_profile(merged)
        if self.id == "emsc_fdsn_events":
            return _validate_fdsn_event_profile(merged, source_name="EMSC FDSN")
        if self.id == "gem_global_active_faults":
            return _validate_gem_profile(merged)
        if self.id == "diss_seismogenic_sources":
            return _validate_diss_profile(merged)
        if self.id == "ontario_mineral_deposit_inventory":
            return _validate_ogs_mdi_profile(merged)
        if self.id == "sigeom_mineral_occurrences":
            return _validate_sigeom_profile(merged)
        return merged

    def intersects_bounds(self, bounds: Any | None) -> bool:
        if bounds is None or self.coverage_bounds is None:
            return True
        min_lon, min_lat, max_lon, max_lat = self.coverage_bounds
        return not (
            float(getattr(bounds, "max_lon")) < min_lon
            or float(getattr(bounds, "min_lon")) > max_lon
            or float(getattr(bounds, "max_lat")) < min_lat
            or float(getattr(bounds, "min_lat")) > max_lat
        )


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
        source_ids: list[str] | tuple[str, ...] | None = None,
        bounds: Any | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> list[SourceDefinition]:
        selected_ids = _resolve_requested_source_ids(source_id, source_ids, options)
        if selected_ids is not None:
            selection = [self.get(item) for item in selected_ids]
            for definition in selection:
                if definition.family != family:
                    raise SourceRegistryError(
                        f"Knowledge source {definition.id!r} has family {definition.family!r}, "
                        f"not {family!r}."
                    )
            return sorted(selection, key=lambda definition: definition.selection_rank)
        selection = [definition for definition in self._definitions.values() if definition.family == family]
        selection = [definition for definition in selection if definition.intersects_bounds(bounds)]
        selection = sorted(selection, key=lambda definition: definition.selection_rank)
        if not selection:
            # Regional families can legitimately have no source covering a request.
            family_definitions = [
                definition for definition in self._definitions.values() if definition.family == family
            ]
            if family_definitions and all(
                definition.coverage_bounds is not None for definition in family_definitions
            ):
                return []
            raise SourceRegistryError(f"No knowledge source registered for family {family!r}.")
        return selection


def _resolve_requested_source_ids(
    source_id: str | None,
    source_ids: list[str] | tuple[str, ...] | None,
    options: Mapping[str, Any] | None,
) -> list[str] | None:
    if source_id is not None and source_ids is not None:
        raise SourceRegistryError("Use either source_id or source_ids, not both.")
    if source_ids is not None:
        return _coerce_source_ids(source_ids)
    if source_id is not None:
        return [source_id]
    if not options:
        return None
    if "source" in options and "sources" in options:
        raise SourceRegistryError("Use either 'source' or 'sources', not both.")
    if "sources" in options:
        return _coerce_source_ids(options.get("sources"))
    source = options.get("source")
    if source in (None, "", "all"):
        return None
    return [str(source)]


def _coerce_source_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    try:
        return [str(part).strip() for part in value if str(part).strip()]
    except TypeError as exc:
        raise SourceRegistryError("sources must be a string or iterable of source IDs.") from exc


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
                selection_rank=10,
            ),
            SourceDefinition(
                id="emsc_fdsn_events",
                family="earthquake_events",
                authority="Euro-Mediterranean Seismological Centre",
                homepage_url="https://www.seismicportal.eu/fdsn-wsevent.html",
                default_format="json",
                license="See EMSC SeismicPortal source policy",
                citation="EMSC SeismicPortal FDSN Event service",
                attribution="EMSC SeismicPortal FDSN Event service",
                coverage_notes=["Global/regional FDSN event service; queried only when explicit."],
                default_profile=dict(EMSC_DEFAULT_PROFILE),
                selection_rank=20,
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
                selection_rank=10,
            ),
            SourceDefinition(
                id="diss_seismogenic_sources",
                family="active_faults",
                authority="Istituto Nazionale di Geofisica e Vulcanologia",
                homepage_url="https://diss.ingv.it/data",
                default_format="geojson",
                license="See DISS source terms",
                citation="Database of Individual Seismogenic Sources (DISS), version 3.3.1",
                attribution="DISS Working Group, INGV",
                coverage_notes=[
                    "DISS is a regional seismogenic-source database for Italy and surrounding areas."
                ],
                default_profile=dict(DISS_DEFAULT_PROFILE),
                selection_rank=20,
                coverage_bounds=(5.0, 35.0, 20.0, 48.5),
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
                selection_rank=10,
                coverage_bounds=(-95.2, 41.6, -74.3, 56.9),
            ),
            SourceDefinition(
                id="sigeom_mineral_occurrences",
                family="mineral_occurrences",
                authority="Ministere des Ressources naturelles et des Forets du Quebec",
                homepage_url="https://sigeom.mines.gouv.qc.ca/signet/classes/I0000_serviceWeb",
                default_format="geojson",
                license="CC BY 4.0 (verify on exact resource metadata before redistribution)",
                citation="SIGEOM mineral occurrences WFS",
                attribution="SIGEOM, Gouvernement du Quebec",
                coverage_notes=["Coverage is the Province of Quebec."],
                default_profile=dict(SIGEOM_DEFAULT_PROFILE),
                selection_rank=20,
                coverage_bounds=(-79.8, 44.8, -57.1, 62.7),
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


def _validate_fdsn_event_profile(profile: Mapping[str, Any], *, source_name: str) -> dict[str, Any]:
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
        raise SourceRegistryError(f"Unknown {source_name} profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "json"))
    if validated["format"] not in {"json", "geojson", "csv"}:
        raise SourceRegistryError(f"{source_name} source only supports json, geojson, or csv profiles.")
    validated["eventtype"] = str(validated.get("eventtype", "earthquake"))
    if validated["eventtype"] != "earthquake":
        raise SourceRegistryError(f"{source_name} earthquake source requires eventtype='earthquake'.")
    for key in ("minmagnitude", "maxmagnitude", "mindepth", "maxdepth"):
        if key in validated and validated[key] not in (None, ""):
            validated[key] = float(validated[key])
    for key in ("limit", "offset"):
        if key in validated and validated[key] not in (None, ""):
            validated[key] = int(validated[key])
    if int(validated.get("limit", 20000)) > 20000:
        raise SourceRegistryError(f"{source_name} source rejects limit values above 20000.")
    if int(validated.get("offset", 1)) < 1:
        raise SourceRegistryError(f"{source_name} offset is one-based and must be >= 1.")
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
    allowed = {"format", "endpoint", "region"}
    unknown = set(profile) - allowed
    if unknown:
        raise SourceRegistryError(f"Unknown OGS MDI profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "geojson"))
    if validated["format"] != "geojson":
        raise SourceRegistryError("OGS MDI source only supports the geojson format.")
    if not validated.get("endpoint"):
        raise SourceRegistryError("OGS MDI profiles require an ArcGIS REST endpoint.")
    return validated


def _validate_diss_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"format", "endpoint", "source_version", "type_names", "srs_name"}
    unknown = set(profile) - allowed
    if unknown:
        raise SourceRegistryError(f"Unknown DISS profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "geojson"))
    if validated["format"] not in {"geojson", "json"}:
        raise SourceRegistryError("DISS profiles currently support only GeoJSON/JSON.")
    if not validated.get("endpoint"):
        raise SourceRegistryError("DISS profiles require an OGC WFS endpoint.")
    if not validated.get("source_version"):
        raise SourceRegistryError("DISS profiles must pin source_version.")
    type_names = validated.get("type_names") or []
    if isinstance(type_names, str):
        type_names = [part.strip() for part in type_names.split(",") if part.strip()]
    validated["type_names"] = [str(item) for item in type_names]
    if not validated["type_names"]:
        raise SourceRegistryError("DISS profiles require at least one WFS type name.")
    validated["srs_name"] = str(validated.get("srs_name") or "CRS:84")
    return validated


def _validate_sigeom_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"format", "endpoint", "region", "feature_types", "srs_name"}
    unknown = set(profile) - allowed
    if unknown:
        raise SourceRegistryError(f"Unknown SIGEOM profile keys: {sorted(unknown)}")
    validated = dict(profile)
    validated["format"] = str(validated.get("format", "geojson"))
    if validated["format"] not in {"geojson", "json"}:
        raise SourceRegistryError("SIGEOM profiles currently support only GeoJSON/JSON.")
    if not validated.get("endpoint"):
        raise SourceRegistryError("SIGEOM profiles require a WFS endpoint.")
    feature_types = validated.get("feature_types") or []
    if isinstance(feature_types, str):
        feature_types = [part.strip() for part in feature_types.split(",") if part.strip()]
    validated["feature_types"] = [str(item) for item in feature_types]
    if not validated["feature_types"]:
        raise SourceRegistryError("SIGEOM profiles require at least one WFS feature type.")
    validated["srs_name"] = str(validated.get("srs_name") or "CRS:84")
    return validated

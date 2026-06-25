"""Resolve free-text or EPSG CRS specifications to a canonical ``EPSG:<code>``.

A geologic map states its CRS as printed text (e.g. ``"UTM N83 Zone 15"``). The
agent reads that string; this module turns it into an EPSG code so the agent does
not have to know UTM/EPSG arithmetic. Only the common UTM and explicit-EPSG forms
are handled deterministically; anything else raises :class:`CRSResolutionError`.
"""

from __future__ import annotations

import re

from .errors import CRSResolutionError

# UTM zone EPSG code = base + zone. NAD83/NAD27 are North American (northern
# hemisphere) datums with no standard southern UTM codes.
_UTM_NORTH_BASE = {"nad83": 26900, "nad27": 26700, "wgs84": 32600}
_UTM_SOUTH_BASE = {"wgs84": 32700}

# Datum aliases as they appear on real maps (e.g. "N83" for NAD83).
_DATUM_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("nad83", "nad 83", "n83"), "nad83"),
    (("nad27", "nad 27", "n27"), "nad27"),
    (("wgs84", "wgs 84", "wgs1984", "w84"), "wgs84"),
)


def _normalize_datum(text: str) -> str | None:
    for aliases, canonical in _DATUM_ALIASES:
        if any(alias in text for alias in aliases):
            return canonical
    return None


def resolve_crs(spec: str | int) -> str:
    """Return a canonical ``"EPSG:<code>"`` for a CRS specification.

    Accepts an EPSG integer (``26915``), an ``"EPSG:xxxx"`` string, or free-text
    UTM such as ``"UTM N83 Zone 15"`` / ``"WGS 84 / UTM zone 15S"``. A bare
    ``"UTM Zone 15N"`` with no datum defaults to WGS84.
    """
    if isinstance(spec, bool):  # bool is an int subclass; reject explicitly.
        raise CRSResolutionError(f"Invalid CRS specification: {spec!r}")
    if isinstance(spec, int):
        return f"EPSG:{spec}"
    if not isinstance(spec, str) or not spec.strip():
        raise CRSResolutionError(f"Invalid CRS specification: {spec!r}")

    text = spec.strip().lower()

    epsg_match = re.fullmatch(r"epsg:\s*(\d+)", text)
    if epsg_match:
        return f"EPSG:{int(epsg_match.group(1))}"

    if "utm" in text:
        zone_match = re.search(r"zone\s*0*(\d{1,2})", text)
        if not zone_match:
            raise CRSResolutionError(f"Could not find a UTM zone in {spec!r}")
        zone = int(zone_match.group(1))
        if not 1 <= zone <= 60:
            raise CRSResolutionError(f"UTM zone {zone} out of range (1-60) in {spec!r}")

        south = bool(re.search(r"\b\d{1,2}\s*s\b", text)) or "south" in text
        datum = _normalize_datum(text) or "wgs84"
        base_table = _UTM_SOUTH_BASE if south else _UTM_NORTH_BASE
        if datum not in base_table:
            hemisphere = "southern" if south else "northern"
            raise CRSResolutionError(
                f"No standard UTM EPSG code for datum {datum!r} in the "
                f"{hemisphere} hemisphere ({spec!r})"
            )
        return f"EPSG:{base_table[datum] + zone}"

    raise CRSResolutionError(f"Unrecognized CRS specification: {spec!r}")

"""CRS string resolution for georeferencing.

A geologic map states its CRS as free text (e.g. "UTM N83 Zone 15"). The agent
should not have to know EPSG arithmetic, so resolve_crs turns common UTM/EPSG
strings into a canonical "EPSG:<code>".
"""

import pytest

from peace_tool_pool.georef import resolve_crs
from peace_tool_pool.georef.errors import CRSResolutionError


def test_resolve_crs_from_epsg_int():
    assert resolve_crs(26915) == "EPSG:26915"


def test_resolve_crs_from_epsg_string_is_normalized():
    assert resolve_crs("epsg:26915") == "EPSG:26915"
    assert resolve_crs("EPSG: 4326") == "EPSG:4326"


def test_resolve_crs_utm_nad83_full_name():
    assert resolve_crs("UTM NAD83 Zone 15") == "EPSG:26915"


def test_resolve_crs_utm_n83_alias_matches_test_maps():
    # The Huronian test maps literally print "UTM N83 Zone 15".
    assert resolve_crs("UTM N83 Zone 15") == "EPSG:26915"


def test_resolve_crs_nad83_slash_form_with_hemisphere():
    assert resolve_crs("NAD83 / UTM zone 17N") == "EPSG:26917"


def test_resolve_crs_wgs84_utm_north():
    assert resolve_crs("WGS84 UTM Zone 15N") == "EPSG:32615"


def test_resolve_crs_wgs84_utm_south():
    assert resolve_crs("WGS 84 / UTM zone 15S") == "EPSG:32715"


def test_resolve_crs_nad27_utm_north():
    assert resolve_crs("NAD27 UTM Zone 15") == "EPSG:26715"


def test_resolve_crs_bare_utm_defaults_to_wgs84_north():
    # No datum token -> default to the most common modern default, WGS84 north.
    assert resolve_crs("UTM Zone 15N") == "EPSG:32615"


def test_resolve_crs_unparseable_raises():
    with pytest.raises(CRSResolutionError):
        resolve_crs("banana republic")


def test_resolve_crs_zone_out_of_range_raises():
    with pytest.raises(CRSResolutionError):
        resolve_crs("UTM NAD83 Zone 61")

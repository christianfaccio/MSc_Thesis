import json
from pathlib import Path
from typing import TypedDict

import xarray as xr


class Source(TypedDict):
    name:  str
    lat:   float
    lon:   float
    depth: float
    Q:     float


def load_sources(path: str | Path) -> list[Source]:
    """Load and validate the source catalog from a JSON file."""
    path = Path(path)
    with path.open() as f:
        sources: list[Source] = json.load(f)

    required = {"name", "lat", "lon", "depth", "Q"}
    for i, s in enumerate(sources):
        missing = required - set(s)
        if missing:
            raise ValueError(f"source[{i}] missing fields: {missing}")
        if s["depth"] < 0:
            raise ValueError(
                f"source[{i}] depth must be positive (down), got {s['depth']}"
            )
        if s["Q"] <= 0:
            raise ValueError(f"source[{i}] Q must be positive, got {s['Q']}")

    return sources


def validate_in_domain(sources: list[Source], cmems: xr.Dataset) -> None:
    """Check that every source lies inside the CMEMS bounding box."""
    lat_min, lat_max = float(cmems.latitude.min()),  float(cmems.latitude.max())
    lon_min, lon_max = float(cmems.longitude.min()), float(cmems.longitude.max())
    depth_max = float(cmems.depth.max())

    for s in sources:
        if not (lat_min <= s["lat"] <= lat_max):
            raise ValueError(
                f"source '{s['name']}' lat={s['lat']} outside "
                f"[{lat_min}, {lat_max}]"
            )
        if not (lon_min <= s["lon"] <= lon_max):
            raise ValueError(
                f"source '{s['name']}' lon={s['lon']} outside "
                f"[{lon_min}, {lon_max}]"
            )
        if s["depth"] > depth_max:
            raise ValueError(
                f"source '{s['name']}' depth={s['depth']} below "
                f"CMEMS max depth {depth_max}"
            )
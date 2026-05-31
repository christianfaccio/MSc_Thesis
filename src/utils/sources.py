import json
from pathlib import Path
from typing import TypedDict
import numpy as np

# xarray imported lazily inside validate_in_domain() so training doesn't need it.


class Source(TypedDict):
    name:  str
    y:   float
    x:   float
    depth: float
    Q:     float

def random_sources(rng: np.random.Generator | None = None, n_sources: int = 1,
                   min_x: float = 0.0, max_x: float = 100.0,
                   min_y: float = 0.0, max_y: float = 100.0,
                   min_depth: float = 0.0, max_depth: float = 100.0,
                   min_q: float = 0.0, max_q: float = 10.0,
                   ) -> list[Source]:
    """Generate n_sources, each anchored to an x- or y-domain border (50/50)
    with random depth and emission rate Q.

    Accepts a numpy Generator so the caller (env.np_random) controls
    determinism without poisoning the global RNG.
    """
    if rng is None:
        rng = np.random.default_rng()
    sources = []
    for i in range(n_sources):
        on_x_border = bool(rng.integers(0, 2))  # source on the y=0 edge if True, else x=0
        x = float(rng.uniform(min_x, max_x)) if on_x_border else 0.0
        y = 0.0 if on_x_border else float(rng.uniform(min_y, max_y))
        depth = float(rng.uniform(min_depth, max_depth))
        Q = float(rng.uniform(min_q, max_q))
        sources.append(Source(name=str(i), x=x, y=y, depth=depth, Q=Q))
    return sources

def load_sources(path: str | Path) -> list[Source]:
    """Load and validate the source catalog from a JSON file."""
    path = Path(path)
    with path.open() as f:
        sources: list[Source] = json.load(f)

    required = {"name", "y", "x", "depth", "Q"}
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

# TODO: check
def validate_in_domain(sources: list[Source], cmems) -> None:
    """Check that every source lies inside the CMEMS bounding box (cmems: xarray.Dataset)."""
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
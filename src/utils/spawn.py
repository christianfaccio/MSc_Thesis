import numpy as np


def spawn_positions_along_land_strip(rng: np.random.Generator, n: int,
                                     domain: tuple[float, float, float],
                                     clearance: float = 500.0) -> np.ndarray:
    """Sample n surface spawn positions on an L-shaped strip a fixed `clearance`
    (m) off the west (x=0) and south (y=0) land borders.

    Land is the west (x=0) and south (y=0) borders (where the salinity sources
    live). Each point sits on one of two arms whose corner is at
    (clearance, clearance):

      - west-parallel arm:  x = clearance, y ∈ [clearance, domain_y]
      - south-parallel arm: y = clearance, x ∈ [clearance, domain_x]

    so every point's distance to the nearest land border is exactly `clearance`.
    Depth z = 0 (agents spawn at the surface). Points are distinct with
    probability 1 (continuous sampling along each arm).

    Accepts a numpy Generator (env.np_random) so the caller controls
    determinism. Returns an (n, 3) float array of (x, y, z) positions.
    """
    X, Y, _ = domain
    pts = np.empty((n, 3), dtype=float)
    for i in range(n):
        if bool(rng.integers(0, 2)):  # west-parallel arm (x fixed at clearance)
            pts[i] = [clearance, float(rng.uniform(clearance, Y)), 0.0]
        else:                          # south-parallel arm (y fixed at clearance)
            pts[i] = [float(rng.uniform(clearance, X)), clearance, 0.0]
    return pts

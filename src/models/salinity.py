import numpy as np

SECONDS_PER_DAY = 86400.0

def _gaussian_raw(x, y, z, centers, weights, sigma_h, sigma_v):
    '''Un-normalized sum of 3D Gaussians. Vectorized: scalars in -> scalar out,
    arrays in -> array out (the exp terms broadcast against x, y, z).'''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    S = np.zeros(np.broadcast(x, y, z).shape)
    for (cx, cy, cz), w in zip(centers, weights):
        S = S + w * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma_h ** 2)
            - (z - cz) ** 2 / (2 * sigma_v ** 2)
        )
    return S


def gaussian_field_norm(centers, weights, sigma_h, sigma_v, domain, n: int = 32):
    '''Return (raw_min, raw_max) of the un-normalized field over the domain.

    Compute this ONCE per field (e.g. per episode, when the blobs are drawn) and
    pass it to compute_salinity_gaussian, so per-point queries stay cheap instead
    of rebuilding the normalization grid on every call.'''
    xs = np.linspace(0.0, domain[0], n)
    ys = np.linspace(0.0, domain[1], n)
    zs = np.linspace(0.0, domain[2], max(8, n // 4))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    raw = _gaussian_raw(X, Y, Z, centers, weights, sigma_h, sigma_v)
    return float(raw.min()), float(raw.max())


def compute_salinity_gaussian(x, y, z, centers, weights, sigma_h, sigma_v, span,
                              raw_min, raw_max):
    '''
    Synthetic salinity [PSU] at (x, y, z): a sum of 3D Gaussians affine-scaled to
    `span` using precomputed (raw_min, raw_max) from `gaussian_field_norm`.

    Scalar in -> float out; array in -> ndarray out (vectorized), so it serves
    both per-agent queries and grid sampling (target selection, plotting).
    '''
    raw = _gaussian_raw(x, y, z, centers, weights, sigma_h, sigma_v)
    S = span * (raw - raw_min) / (raw_max - raw_min)
    return float(S) if S.ndim == 0 else S

def compute_salinity_gradient_gaussian(x, y, z, centers, weights, sigma_h, sigma_v,
                                       span, raw_min, raw_max):
    '''
    Analytic gradient (dS/dx, dS/dy, dS/dz) [PSU/m] of compute_salinity_gaussian.
    The affine span-normalization is a constant scale, so the gradient is just
    the raw-field gradient times span / (raw_max - raw_min).
    '''
    dSdx = dSdy = dSdz = 0.0
    for (cx, cy, cz), w in zip(centers, weights):
        dx, dy, dz = x - cx, y - cy, z - cz
        S_i = w * np.exp(-(dx * dx + dy * dy) / (2 * sigma_h ** 2)
                         - dz * dz / (2 * sigma_v ** 2))
        dSdx += -(dx / sigma_h ** 2) * S_i
        dSdy += -(dy / sigma_h ** 2) * S_i
        dSdz += -(dz / sigma_v ** 2) * S_i
    scale = span / (raw_max - raw_min)
    return scale * dSdx, scale * dSdy, scale * dSdz

def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Convert cell-center coordinates to cell-edge boundaries."""
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges

def compute_salinity(
    currents,                            # xarray.Dataset; type-hint avoided for lazy import
    sources: list,
    spinup_days: float = 60.0,          # For how many days running the model
    release_interval_s: float = 600.0,  # Seconds for each release
    Kh: float = 1.0,                    # noise param
    Kv: float = 1.0e-3,                 # nose param
    dt_s: float = 300.0,
    advection_tail_days: float = 7.0,
) -> np.ndarray:
    """Run an OceanParcels Lagrangian simulation to steady state.

    Returns a (nz, ny, nx) array of salinity excess [PSU] on the bundle's
    grid, with each source contributing in proportion to its emission rate Q.

    Parcels (and xarray) are imported lazily so callers that only use
    `compute_salinity_analytical` don't need them installed.
    """
    from parcels import (
        AdvectionRK4_3D,
        FieldSet,
        JITParticle,
        ParcelsRandom,
        ParticleSet,
        StatusCode,
    )

    def _diffuse(particle, fieldset, time):
        """Turbulent diffusion as Gaussian random walk."""
        dx = ParcelsRandom.normalvariate(0., 1.) * (2.0 * fieldset.Kh * particle.dt) ** 0.5
        dy = ParcelsRandom.normalvariate(0., 1.) * (2.0 * fieldset.Kh * particle.dt) ** 0.5
        dz = ParcelsRandom.normalvariate(0., 1.) * (2.0 * fieldset.Kv * particle.dt) ** 0.5
        particle_dlon += dx   # noqa: F821
        particle_dlat += dy               # noqa: F821
        particle_ddepth += dz                       # noqa: F821

    def _delete_out_of_bounds(particle, fieldset, time):
        """Remove particles that escape the domain (surface, sides, or below)."""
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()
        if particle.state == StatusCode.ErrorThroughSurface:
            particle.delete()

    # --- 1. Build the FieldSet (Parcels expects positive-down depth) -------
    fieldset = FieldSet.from_xarray_dataset(
        currents,
        variables={"U": "u", "V": "v", "W": "w"},
        dimensions={
            "lat": "y",
            "lon": "x",
            "depth": "depth",
        },
        mesh="flat",   # interpret the horizontal coordinates meters
        allow_time_extrapolation=True,
    )
    fieldset.add_constant("Kh", Kh)
    fieldset.add_constant("Kv", Kv)

    # --- 2. Continuous release, weighted by Q ------------------------------
    # Each source releases a number of particles ∝ Q_i, so each particle
    # carries the same salt mass — the binned counts are then directly
    # proportional to local concentration.
    total_seconds = spinup_days * SECONDS_PER_DAY
    Q_max = max(s["Q"] for s in sources)
    base_n = max(1, int(total_seconds // release_interval_s))   # how many releases at minimum

    x, y, depths, times = [], [], [], []
    for src in sources:
        n_i = max(1, int(round(base_n * src["Q"] / Q_max)))
        ts = np.linspace(0.0, total_seconds, n_i, endpoint=False)
        for t in ts:
            x.append(src["x"])
            y.append(src["y"])
            depths.append(src["depth"])
            times.append(float(t))

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=JITParticle,
        lon=x,
        lat=y,
        depth=depths,
        time=times,
    )

    # --- 3. Run forward (last-released particles still need to advect) -----
    runtime_s = total_seconds + advection_tail_days * SECONDS_PER_DAY
    kernel = (
        pset.Kernel(AdvectionRK4_3D)
        + pset.Kernel(_diffuse)
        + pset.Kernel(_delete_out_of_bounds)
    )
    pset.execute(kernel, runtime=runtime_s, dt=dt_s)

    # --- 4. Bin onto the bundle's grid -------------------------------------
    z_grid = currents.z.values                            # positive depth
    y_grid = currents.y.values
    x_grid = currents.x.values

    z_edges = _cell_edges(z_grid)
    y_edges = _cell_edges(y_grid)
    x_edges = _cell_edges(x_grid)

    p_depth = np.array([p.depth for p in pset])
    p_y = np.array([p.lat for p in pset])
    p_x = np.array([p.lon for p in pset])

    counts, _ = np.histogramdd(
        np.column_stack([p_depth, p_y, p_x]),
        bins=[z_edges, y_edges, x_edges],
    )

    # --- 5. Counts -> PSU concentration ------------------------------------
    # Per-particle mass: chosen so peak emitter (Q_max) at base_n particles
    # carries Q_max * total_seconds total salt.
    mass_per_particle = Q_max * total_seconds / base_n   # [PSU * m^3]

    dz_m = float(np.diff(z_edges).mean())
    dy_m = float(np.diff(y_edges).mean())
    dx_m = float(np.diff(x_edges).mean())
    cell_volume = dz_m * dy_m * dx_m

    salinity_excess = counts * mass_per_particle / cell_volume   # [PSU]
    return salinity_excess

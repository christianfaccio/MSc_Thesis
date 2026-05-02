"""
Pre-compute the environment bundles used during training and testing.

Each bundle is a NetCDF file containing:
    - currents     (u, v, w)        -> CMEMS surface + analytical Ekman extension
    - salinity                       -> Lagrangian particles from industrial sources
    - turbidity                      -> Beer-Lambert depth attenuation

A `manifest.json` is written alongside the bundles as the index used by the
Gym env at reset().

Sign convention: depth is stored on the `z` coordinate as **negative-down**
(z = 0 at surface, z < 0 underwater). All downstream code uses np.abs(z) when
a positive depth is needed (Parcels, turbidity model, lat/lon conversions).

Usage:
    python scripts/precompute_envs.py --n-envs 100
"""

import argparse
import math
from pathlib import Path

import numpy as np
import xarray as xr
from parcels import (
    AdvectionRK4_3D,
    FieldSet,
    JITParticle,
    ParcelsRandom,
    ParticleSet,
    StatusCode,
)

from src.models.turbidity import turbidity_model  # noqa: E402
from src.utils.manifest import write_manifest  # noqa: E402
from src.utils.sources import load_sources, validate_in_domain  # noqa: E402


# --------------------------------------------------------------------------
# Constants (Ekman + grid)
# --------------------------------------------------------------------------
LATITUDE = 24.5            # Abu Dhabi reference latitude [deg]
EDDY_VISCOSITY = 0.05      # turbulent mixing coefficient [m^2/s]
DEPTH_LEVELS = np.array([0, 5, 10, 20, 30, 50, 75, 100], dtype=float)  # meters

OMEGA = 7.2921e-5
sin_lat = np.sin(np.deg2rad(LATITUDE))
F_CORIOLIS = 2.0 * OMEGA * sin_lat
D_E = np.sqrt(2.0 * EDDY_VISCOSITY / abs(F_CORIOLIS))   # Ekman depth [m]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Convert cell-center coordinates to cell-edge boundaries."""
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def diffuse_3d(particle, fieldset, time):
    """Turbulent diffusion as Gaussian random walk.

    Parcels v3 API: accumulate into particle_dlon/dlat/ddepth, never
    modify particle.lon/lat/depth directly. Uses `math` (not numpy)
    because kernels are JIT-compiled to C.
    """
    dx = ParcelsRandom.normalvariate(0., 1.) * (2.0 * fieldset.Kh * particle.dt) ** 0.5
    dy = ParcelsRandom.normalvariate(0., 1.) * (2.0 * fieldset.Kh * particle.dt) ** 0.5
    dz = ParcelsRandom.normalvariate(0., 1.) * (2.0 * fieldset.Kv * particle.dt) ** 0.5
    cos_lat = math.cos(particle.lat * math.pi / 180.0)
    # NOTE: 111320.0 is used for spherical correction (see mesh: "spherical" on `run_particles()`)
    particle_dlon += dx / (111320.0 * cos_lat)  # noqa: F821
    particle_dlat += dy / 111320.0              # noqa: F821
    particle_ddepth += dz                       # noqa: F821


def delete_out_of_bounds(particle, fieldset, time):
    """Remove particles that escape the domain (surface, sides, or below)."""
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()
    if particle.state == StatusCode.ErrorThroughSurface:
        particle.delete()


# --------------------------------------------------------------------------
# Pipeline pieces
# --------------------------------------------------------------------------
def compose_currents(ds: xr.Dataset, counter: int) -> xr.Dataset:
    """Build a 3D current bundle (CMEMS surface + Ekman extension).

    Wraps around CMEMS time if `counter` exceeds the dataset's time axis.
    """
    time_idx = counter % len(ds.time)
    snap = ds.isel(time=time_idx)
    surface = snap.isel(depth=0)

    lons = surface.longitude.values
    lats = surface.latitude.values

    u_surf = np.nan_to_num(surface.uo.values, nan=0.0)
    v_surf = np.nan_to_num(surface.vo.values, nan=0.0)

    speed_surf = np.hypot(u_surf, v_surf)
    angle_surf = np.arctan2(v_surf, u_surf)

    z_pos = DEPTH_LEVELS                                 # positive depth [m]
    zn = np.pi * z_pos / D_E
    decay = np.exp(-zn)

    phase = angle_surf[None, :, :] + np.pi / 4 - zn[:, None, None]
    u3d = decay[:, None, None] * speed_surf[None, :, :] * np.cos(phase)
    v3d = decay[:, None, None] * speed_surf[None, :, :] * np.sin(phase)
    w3d = np.zeros_like(u3d)

    bundle = xr.Dataset(
        data_vars={
            "u": (("z", "latitude", "longitude"), u3d),
            "v": (("z", "latitude", "longitude"), v3d),
            "w": (("z", "latitude", "longitude"), w3d),
        },
        coords={
            "z": ("z", -z_pos, {"units": "m", "positive": "up"}),
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
        },
        attrs={
            "cmems_time_index": int(time_idx),
            "cmems_time": str(snap.time.values),
            "ekman_depth_m": float(D_E),
            "ekman_Az": float(EDDY_VISCOSITY),
            "latitude_ref": float(LATITUDE),
            "depth_levels_m": z_pos.tolist(),
        },
    )
    return bundle


def run_particles(
    bundle: xr.Dataset,
    sources: list,
    spinup_days: float = 60.0,
    release_interval_s: float = 600.0,
    Kh: float = 1.0,
    Kv: float = 1.0e-3,
    dt_s: float = 300.0,
    advection_tail_days: float = 7.0,
) -> np.ndarray:
    """Run an OceanParcels Lagrangian simulation to steady state.

    Returns a (nz, ny, nx) array of salinity excess [PSU] on the bundle's
    grid, with each source contributing in proportion to its emission rate Q.
    """
    # --- 1. Build the FieldSet (Parcels expects positive-down depth) -------
    z_pos_coord = -bundle.z.values                       # negative-down -> positive
    fs_ds = bundle.assign_coords(z=z_pos_coord)
    fs_ds = fs_ds.expand_dims(time=[np.datetime64("2020-01-01")]) # Parcels expects a time dimension

    fieldset = FieldSet.from_xarray_dataset(
        fs_ds,
        variables={"U": "u", "V": "v", "W": "w"},
        dimensions={
            "lat": "latitude",
            "lon": "longitude",
            "depth": "z",
            "time": "time",
        },
        mesh="spherical",   # interpret the horizontal coordinates as degrees and not meters
        allow_time_extrapolation=True,
    )
    fieldset.add_constant("Kh", Kh)
    fieldset.add_constant("Kv", Kv)

    # --- 2. Continuous release, weighted by Q ------------------------------
    # Each source releases a number of particles ∝ Q_i, so each particle
    # carries the same salt mass — the binned counts are then directly
    # proportional to local concentration.
    total_seconds = spinup_days * 86400.0
    Q_max = max(s["Q"] for s in sources)
    base_n = max(1, int(total_seconds // release_interval_s))

    lons, lats, depths, times = [], [], [], []
    for src in sources:
        n_i = max(1, int(round(base_n * src["Q"] / Q_max)))
        ts = np.linspace(0.0, total_seconds, n_i, endpoint=False)
        for t in ts:
            lons.append(src["lon"])
            lats.append(src["lat"])
            depths.append(src["depth"])
            times.append(float(t))

    pset = ParticleSet(
        fieldset=fieldset,
        pclass=JITParticle,
        lon=lons,
        lat=lats,
        depth=depths,
        time=times,
    )

    # --- 3. Run forward (last-released particles still need to advect) -----
    runtime_s = total_seconds + advection_tail_days * 86400.0
    kernel = (
        pset.Kernel(AdvectionRK4_3D)
        + pset.Kernel(diffuse_3d)
        + pset.Kernel(delete_out_of_bounds)
    )
    pset.execute(kernel, runtime=runtime_s, dt=dt_s)

    # --- 4. Bin onto the bundle's grid -------------------------------------
    z_grid = -bundle.z.values                            # positive depth
    lat_grid = bundle.latitude.values
    lon_grid = bundle.longitude.values

    z_edges = _cell_edges(z_grid)
    lat_edges = _cell_edges(lat_grid)
    lon_edges = _cell_edges(lon_grid)

    p_depth = np.array([p.depth for p in pset])
    p_lat = np.array([p.lat for p in pset])
    p_lon = np.array([p.lon for p in pset])

    counts, _ = np.histogramdd(
        np.column_stack([p_depth, p_lat, p_lon]),
        bins=[z_edges, lat_edges, lon_edges],
    )

    # --- 5. Counts -> PSU concentration ------------------------------------
    # Per-particle mass: choose so peak emitter (Q_max) at base_n particles
    # carries Q_max * total_seconds total salt.
    mass_per_particle = Q_max * total_seconds / base_n   # [PSU * m^3]

    dz_m = float(np.diff(z_edges).mean())
    mean_lat = float(np.mean(lat_grid))
    dlat_m = float(np.diff(lat_edges).mean()) * 111320.0
    dlon_m = float(np.diff(lon_edges).mean()) * 111320.0 * np.cos(np.deg2rad(mean_lat))
    cell_volume = dz_m * dlat_m * dlon_m

    salinity_excess = counts * mass_per_particle / cell_volume   # [PSU]
    return salinity_excess


def bundle_environment(
    currents: xr.Dataset,
    salinity: np.ndarray,
    k_turbidity: float = 0.3,
) -> xr.Dataset:
    """Combine currents + salinity + turbidity into one Dataset."""
    expected = (
        currents.sizes["z"],
        currents.sizes["latitude"],
        currents.sizes["longitude"],
    )
    if salinity.shape != expected:
        raise ValueError(
            f"salinity shape {salinity.shape} != grid {expected}"
        )

    depth = np.abs(currents.z.values)
    tau_1d = turbidity_model(depth, k=k_turbidity)
    tau_3d = np.broadcast_to(tau_1d[:, None, None], expected).copy()

    bundle = currents.assign(
        salinity=(("z", "latitude", "longitude"), salinity),
        turbidity=(("z", "latitude", "longitude"), tau_3d),
    )
    bundle["salinity"].attrs = {
        "units": "PSU",
        "long_name": "salinity field from Lagrangian source simulation",
    }
    bundle["turbidity"].attrs = {
        "units": "1",
        "long_name": "Beer-Lambert depth attenuation",
        "k_per_m": float(k_turbidity),
    }
    bundle.attrs["k_turbidity"] = float(k_turbidity)
    return bundle


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="data/envs/")
    parser.add_argument(
        "--cmems-file", type=str, default="data/abu_dhabi_ocean_data.nc"
    )
    parser.add_argument(
        "--sources-file", type=str, default="config/sources.json"
    )
    parser.add_argument("--k-turbidity", type=float, default=0.3)
    parser.add_argument("--spinup-days", type=float, default=60.0)
    parser.add_argument("--release-interval-s", type=float, default=600.0)
    parser.add_argument("--Kh", type=float, default=1.0)
    parser.add_argument("--Kv", type=float, default=1.0e-3)
    parser.add_argument("--dt-s", type=float, default=300.0)
    parser.add_argument("--advection-tail-days", type=float, default=7.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cmems = xr.open_dataset(args.cmems_file)
    sources = load_sources(args.sources_file)
    validate_in_domain(sources, cmems)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_params = {
        "ekman_eddy_viscosity": EDDY_VISCOSITY,
        "ekman_latitude": LATITUDE,
        "ekman_depth_m": float(D_E),
        "depth_levels_m": DEPTH_LEVELS.tolist(),
        "k_turbidity": args.k_turbidity,
        "particles": {
            "spinup_days": args.spinup_days,
            "advection_tail_days": args.advection_tail_days,
            "release_interval_s": args.release_interval_s,
            "Kh": args.Kh,
            "Kv": args.Kv,
            "dt_s": args.dt_s,
        },
        "cmems_file": str(args.cmems_file),
    }

    environments: list[dict] = []

    for env_id in range(args.n_envs):
        path = out_dir / f"env_{env_id:03d}.nc"

        if path.exists() and not args.overwrite:
            print(f"[{env_id}] {path.name} exists, skipping")
        else:
            if path.exists():
                path.unlink()

            currents = compose_currents(cmems, env_id)
            salinity = run_particles(
                currents,
                sources,
                spinup_days=args.spinup_days,
                release_interval_s=args.release_interval_s,
                Kh=args.Kh,
                Kv=args.Kv,
                dt_s=args.dt_s,
                advection_tail_days=args.advection_tail_days,
            )
            bundle = bundle_environment(
                currents, salinity, k_turbidity=args.k_turbidity
            )
            bundle.to_netcdf(path)
            bundle.close()
            print(f"[{env_id}] wrote {path.name}")

        # Always record, including for skipped bundles
        with xr.open_dataset(path) as b:
            environments.append(
                {
                    "env_id": env_id,
                    "path": path.name,
                    "cmems_time": str(b.attrs.get("cmems_time", "")),
                    "size_bytes": path.stat().st_size,
                }
            )

    manifest_path = write_manifest(
        out_dir,
        environments=environments,
        sources=sources,
        global_params=global_params,
    )
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

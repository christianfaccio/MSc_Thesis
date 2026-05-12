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
from pathlib import Path

import numpy as np
import xarray as xr

from src.models.salinity import compute_salinity #noqa: E402
from src.models.turbidity import compute_turbidity  # noqa: E402
from src.utils.manifest import write_manifest  # noqa: E402
from src.utils.sources import load_sources, validate_in_domain  # noqa: E402
from src.utils.pos_latlon import latlon_to_pos

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

def compose_currents(counter: int, 
                     xml_file: Path, 
                     ds: xr.Dataset = None, 
                     grid_n: int = 25, 
                     extent: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)) -> xr.Dataset:
    """Build a 3D current bundle (real/synthetic surface + Ekman extension).

    Wraps around CMEMS time if `counter` exceeds the dataset's time axis.
    """
    if ds is not None:
        time_idx = counter % len(ds.time)
        snap = ds.isel(time=time_idx)
        surface = snap.isel(depth=0)

        lons = surface.longitude.values
        lats = surface.latitude.values
        xyz = np.array([latlon_to_pos(lat=la, lon=0.0, depth=0.0) for la in lats])
        y_pos = xyz[:, 1]
        xyz = np.array([latlon_to_pos(lat=0.0, lon=lo, depth=0.0) for lo in lons])
        x_pos = xyz[:, 0]

        u_surf = np.nan_to_num(surface.uo.values, nan=0.0)
        v_surf = np.nan_to_num(surface.vo.values, nan=0.0)
    else:
        from scripts.plot_currents import parse_envrioment_parameters, select_components, compute_fields_2d
        data = parse_envrioment_parameters(xml_file)
        comps = select_components(data, {"uniform","noise","vortex","global_waves","local_waves"}, t0=counter)  # TODO: insert more variability, better if
                                                                                                                # by changing the currents parameters directly.
                                                                                                                # Now only the phase is changing
        x_pos, y_pos, fields = compute_fields_2d(comps, 0.0, 0.0, grid_n, extent)
        x_axis = x_pos[0, :]   # x varies along axis 1
        y_axis = y_pos[:, 0]   # y varies along axis 0

        u_surf = sum(uv[0] for uv in fields.values())
        v_surf = sum(uv[1] for uv in fields.values())

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
            "u": (("z", "y", "x"), u3d),
            "v": (("z", "y", "x"), v3d),
            "w": (("z", "y", "x"), w3d),
        },
        coords={    
            "depth": ("z", z_pos, {"units": "m", "positive": "up"}),
            "y": ("y", y_axis, {"units": "m"}),
            "x": ("x", x_axis, {"units": "m"}),
            "time": ("time", [np.datetime64("2020-01-01")]),
        },
        attrs={
            "ekman_depth": float(D_E),
            "eddy_viscosity": float(EDDY_VISCOSITY),
            "latitude_ref": float(LATITUDE),
            "depth_levels": z_pos.tolist(),
        },
    )
    return bundle

def bundle_environment(
    currents: xr.Dataset,
    salinity: np.ndarray,
    k_turbidity: float = 0.3,
) -> xr.Dataset:
    """Combine currents + salinity + turbidity into one Dataset."""
    expected = (
        currents.sizes["z"],
        currents.sizes["y"],
        currents.sizes["x"],
    )
    if salinity.shape != expected:
        raise ValueError(
            f"salinity shape {salinity.shape} != grid {expected}"
        )

    depth = np.abs(currents.z.values)
    tau_1d = compute_turbidity(depth, k=k_turbidity)
    tau_3d = np.broadcast_to(tau_1d[:, None, None], expected).copy()

    bundle = currents.assign(
        salinity=(("z", "y", "x"), salinity),
        turbidity=(("z", "y", "x"), tau_3d),
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-type", type=str, default="synthetic")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="data/envs/")
    parser.add_argument(
        "--cmems-file", type=str, default="data/abu_dhabi_ocean_data.nc"
    )
    parser.add_argument(
        "--xml-file", type=str, default="config/simulation.xml"
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

    sources = load_sources(args.sources_file)
    if args.data_type == "real":
        cmems = xr.open_dataset(args.cmems_file) 
        validate_in_domain(sources, cmems)
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
    else:
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
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    environments: list[dict] = []

    for env_id in range(args.n_envs):
        path = out_dir / f"env_{args.data_type}_{env_id:03d}.nc"

        if path.exists() and not args.overwrite:
            print(f"[{env_id}] {path.name} exists, skipping")
        else:
            if path.exists():
                path.unlink()

            currents = compose_currents(counter=env_id, xml_file=Path(args.xml_file), ds=cmems) if args.data_type == "real" else compose_currents(counter=env_id, xml_file=Path(args.xml_file))
            salinity = compute_salinity(
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

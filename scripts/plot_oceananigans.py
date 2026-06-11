"""
Visualize Oceananigans output from a NetCDF produced by oceananigans/hydrostatic.jl
(hydrostatic, 5 km → hydrostatic_<season>.nc) or non_hydrostatic.jl (LES, 100 m).
Both write the same variables (u, v, w, T, S) on an Arakawa C-grid, so the same
script handles either file.

Produces, under data/oceananigans/plots/, a static view of one snapshot:
  - currents_<season>.png          (3D matplotlib quiver, colored by speed)
  - currents_surface_<season>.png  (2D top-down surface streamlines, colored by speed)
  - salinity_<season>.html     (plotly volume, red X source markers)
  - temperature_<season>.html  (plotly volume)
  - turbidity_<season>.html    (plotly volume, analytical Beer-Lambert from depth)
and, with --animate, top-down GIFs of the time-evolution:
  - salinity_<season>.gif      (plan-view, column-max, source markers)
  - temperature_<season>.gif   (plan-view, depth-mean)

Usage:
    python scripts/plot_oceananigans.py                       # last snapshot of hydrostatic_winter.nc
    python scripts/plot_oceananigans.py --season winter --time-idx 100
    python scripts/plot_oceananigans.py --season winter --time-hours 48
    python scripts/plot_oceananigans.py --season winter --animate --no-show
    python scripts/plot_oceananigans.py --file data/oceananigans/hydrostatic_winter.nc \
        --sources-file config/sources.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.models.turbidity import compute_turbidity  # noqa: E402
from src.utils.plotting import (  # noqa: E402
    _coord, _fmt_time, _to_center_zyx, plot_volume_netcdf, plot_currents_netcdf,
    plot_surface_currents_netcdf, animate_field_netcdf,
)

DATA_DIR = REPO_ROOT / "data" / "oceananigans"
PLOT_DIR = DATA_DIR / "plots"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", choices=["winter", "summer"], default="winter")
    p.add_argument("--file", type=Path, default=None,
                   help="path to the NetCDF (default: data/oceananigans/hydrostatic<season>.nc)")
    p.add_argument("--sources-file", type=Path, default=None,
                   help="source catalog JSON for markers (default: config/sources.json)")
    p.add_argument("--time-idx", type=int, default=-1,
                   help="time snapshot index for the static plots (default: -1 = last). "
                        "Negative indexes from the end, as in numpy.")
    p.add_argument("--time-hours", type=float, default=None,
                   help="pick the static snapshot nearest this simulation time [hours]; "
                        "overrides --time-idx when given.")
    p.add_argument("--animate", action="store_true",
                   help="also write top-down evolution GIFs of T and S over all frames")
    p.add_argument("--anim-fps", type=int, default=12,
                   help="GIF frames per second (default: 12)")
    p.add_argument("--anim-stride", type=int, default=1,
                   help="temporal subsampling for the GIF: keep every Nth frame (default: 1)")
    p.add_argument("--anim-reduce", choices=["auto", "max", "mean", "slice"], default="auto",
                   help="vertical collapse for the GIF (default auto: S=max, T=mean). "
                        "'slice' uses --anim-depth.")
    p.add_argument("--anim-depth", type=float, default=2.0,
                   help="depth [m, positive-down] for --anim-reduce slice (default: 2)")
    p.add_argument("--stride", type=int, default=10,
                   help="quiver subsampling stride per axis (3D currents plot)")
    p.add_argument("--stream-density", type=float, default=1.4,
                   help="streamline density for the 2D surface-currents plot (default: 1.4)")
    p.add_argument("--vol-grid", type=int, default=40,
                   help="downsampled volume grid per axis for Plotly")
    p.add_argument("--z-aspect", type=float, default=0.2,
                   help="on-screen height of the z-axis as a fraction of the horizontal "
                        "extent (domain-independent: 1.0 = cube, which over-inflates "
                        "vertical currents on a wide/shallow domain). Default 0.2 keeps "
                        "depth visible without distorting the flow, for any domain size.")
    p.add_argument("--k-turbidity", type=float, default=0.3,
                   help="Beer-Lambert k [1/m]; 0.3 = Arabian Gulf coastal default")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    nc_path = args.file if args.file is not None else DATA_DIR / f"hydrostatic_{args.season}.nc"
    if not nc_path.exists():
        sys.exit(f"NetCDF not found: {nc_path}\n"
                 f"Run `cd oceananigans && OCEAN_ARCH=CPU julia --project=. abu_dhabi_coastal.jl` "
                 f"first (with SEASON = :{args.season}), or pass --file.")

    # Prefer the run's metadata sidecar (written by hydrostatic.jl per run) for
    # sources and labeling; otherwise fall back to config/sources.json.
    sidecar = nc_path.with_suffix(".json")
    if args.sources_file is not None:
        sources_file = args.sources_file
    elif sidecar.exists():
        sources_file = sidecar
    else:
        sources_file = REPO_ROOT / "config" / "sources.json"

    ds = xr.open_dataset(nc_path, decode_timedelta=True)
    print(f"loaded {nc_path}")
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  vars: {list(ds.data_vars)}")

    # Reference cell-center grid from T.
    _, xc = _coord(ds.T, "x")
    _, yc = _coord(ds.T, "y")
    _, zc = _coord(ds.T, "z")

    sources = json.loads(Path(sources_file).read_text()) if Path(sources_file).exists() else None
    label = args.season
    if isinstance(sources, dict):
        # Run-metadata sidecar: {"season": ..., "run_index": ..., "sources": [...]}
        label = f"{sources.get('season', args.season)}_run{sources.get('run_index', 0):03d}"
        sources = sources.get("sources")
        print(f"  using run metadata sidecar: {sources_file} (label: {label})")
    if sources is None:
        print(f"  (no source markers: {sources_file} not found)")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve which snapshot the static plots draw. --time-hours wins if given.
    n_times = ds.sizes["time"]
    if args.time_hours is not None:
        target = np.timedelta64(int(round(args.time_hours * 3600)), "s")
        elapsed = ds.time.values - ds.time.values[0]
        t_idx = int(np.abs(elapsed - target).argmin())
    else:
        t_idx = args.time_idx
    t_idx_norm = t_idx % n_times  # normalize negatives for reporting
    elapsed_t = ds.time.values[t_idx] - ds.time.values[0]
    print(f"  static snapshot: index {t_idx_norm}/{n_times - 1}  "
          f"(t = {_fmt_time(elapsed_t)} into recording)")

    # 1) Currents — 3D quiver + 2D top-down surface streamlines
    plot_currents_netcdf(ds, t_idx, xc, yc, zc, args.stride, args.z_aspect, sources,
                  PLOT_DIR / f"currents_{label}.png")
    plot_surface_currents_netcdf(ds, t_idx, xc, yc, zc, args.stream_density, sources,
                  PLOT_DIR / f"currents_surface_{label}.png")

    # 2) Salinity
    S = _to_center_zyx(ds.S.isel(time=t_idx), xc, yc, zc)
    plot_volume_netcdf(S, xc, yc, zc, args.vol_grid, "Viridis",
                f"Salinity [PSU] — {label}", "S", args.z_aspect,
                sources, PLOT_DIR / f"salinity_{label}.html")

    # 3) Temperature
    T_field = _to_center_zyx(ds.T.isel(time=t_idx), xc, yc, zc)
    plot_volume_netcdf(T_field, xc, yc, zc, args.vol_grid, "Plasma",
                f"Temperature [°C] — {label}", "T", args.z_aspect,
                None, PLOT_DIR / f"temperature_{label}.html")

    # 4) Turbidity (analytical Beer-Lambert, depth-only, broadcast to 3D)
    depth_pos = -zc                         # positive-down depth values
    tau_1d = compute_turbidity(depth_pos, k=args.k_turbidity)
    tau_3d = np.broadcast_to(
        tau_1d[:, None, None], (len(zc), len(yc), len(xc))
    ).copy()
    plot_volume_netcdf(tau_3d, xc, yc, zc, args.vol_grid, "Greys",
                f"Turbidity τ — k={args.k_turbidity} [1/m]", "τ", args.z_aspect,
                None, PLOT_DIR / f"turbidity_{label}.html")

    # 5) Evolution GIFs (top-down, whole time series)
    if args.animate:
        s_reduce = "max" if args.anim_reduce == "auto" else args.anim_reduce
        t_reduce = "mean" if args.anim_reduce == "auto" else args.anim_reduce
        animate_field_netcdf(
            ds, "S", sources, PLOT_DIR / f"salinity_{label}.gif",
            reduce=s_reduce, depth=args.anim_depth, fps=args.anim_fps,
            frame_stride=args.anim_stride, cmap="viridis",
            label="S [PSU]", title=f"Salinity — {label}")
        animate_field_netcdf(
            ds, "T", None, PLOT_DIR / f"temperature_{label}.gif",
            reduce=t_reduce, depth=args.anim_depth, fps=args.anim_fps,
            frame_stride=args.anim_stride, cmap="plasma",
            label="T [°C]", title=f"Temperature — {label}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

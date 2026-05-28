"""
Visualize a single Oceananigans LES snapshot from
data/oceananigans/scenario_01_<season>.nc.

Produces four 3D plots under data/oceananigans/plots/:
  - currents_<season>.png      (matplotlib quiver, colored by speed)
  - salinity_<season>.html     (plotly volume, red X source markers)
  - temperature_<season>.html  (plotly volume)
  - turbidity_<season>.html    (plotly volume, analytical Beer-Lambert from depth)

Usage:
    python scripts/plot_oceananigans.py
    python scripts/plot_oceananigans.py --season summer --time-idx 5
    python scripts/plot_oceananigans.py --stride 8 --vol-grid 50
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.models.turbidity import compute_turbidity  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "oceananigans"
PLOT_DIR = DATA_DIR / "plots"
SOURCES_FILE = REPO_ROOT / "config" / "sources.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", choices=["winter", "summer"], default="winter")
    p.add_argument("--time-idx", type=int, default=-1,
                   help="time snapshot index (default: last)")
    p.add_argument("--stride", type=int, default=10,
                   help="quiver subsampling stride per axis")
    p.add_argument("--vol-grid", type=int, default=40,
                   help="downsampled volume grid per axis for Plotly")
    p.add_argument("--k-turbidity", type=float, default=0.3,
                   help="Beer-Lambert k [1/m]; 0.3 = Arabian Gulf coastal default")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def _coord(da: xr.DataArray, axis: str) -> tuple[str, np.ndarray]:
    """Return (dim name, values) for a spatial axis ('x'/'y'/'z') of `da`."""
    for d in da.dims:
        if d.lower().startswith(axis):
            return d, da.coords[d].values
    raise KeyError(f"no '{axis}' coord on {da.name} (dims={da.dims})")


def _to_center_zyx(da: xr.DataArray, xc, yc, zc) -> np.ndarray:
    """Interpolate `da` to cell centers (xc, yc, zc) and return (nz, ny, nx)."""
    xd, xv = _coord(da, "x")
    yd, yv = _coord(da, "y")
    zd, zv = _coord(da, "z")
    interp = {}
    if not np.array_equal(xv, xc):
        interp[xd] = xc
    if not np.array_equal(yv, yc):
        interp[yd] = yc
    if not np.array_equal(zv, zc):
        interp[zd] = zc
    arr = da.interp(interp) if interp else da
    return arr.transpose(zd, yd, xd).values


def plot_currents(ds: xr.Dataset, time_idx: int, xc, yc, zc,
                  stride: int, sources: list[dict], out_path: Path) -> None:
    u = _to_center_zyx(ds.u.isel(time=time_idx), xc, yc, zc)
    v = _to_center_zyx(ds.v.isel(time=time_idx), xc, yc, zc)
    w = _to_center_zyx(ds.w.isel(time=time_idx), xc, yc, zc)

    s = stride
    Z, Y, X = np.meshgrid(zc[::s], yc[::s], xc[::s], indexing="ij")
    ud, vd, wd = u[::s, ::s, ::s], v[::s, ::s, ::s], w[::s, ::s, ::s]
    speed = np.sqrt(ud ** 2 + vd ** 2 + wd ** 2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    smin, smax = float(speed.min()), float(speed.max())
    norm = plt.Normalize(smin, smax if smax > smin else smin + 1e-9)
    colors = plt.cm.viridis(norm(speed.ravel()))

    domain = max(float(xc.max() - xc.min()), float(yc.max() - yc.min()))
    arrow = 0.05 * domain if domain > 0 else 0.05

    # Z values are negative (Oceananigans convention: surface at z=0, depths at z<0).
    # matplotlib will naturally place larger (less negative) values toward the top.
    ax.quiver(X, Y, Z, ud, vd, wd,
              length=arrow, normalize=True, colors=colors, linewidth=0.6)

    if sources:
        for src in sources:
            ax.scatter(src["x"], src["y"], -float(src["depth"]),
                       c="red", s=80, marker="X",
                       edgecolors="black", linewidths=1.0, label=src["name"])
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]  (surface at 0)")
    t_value = ds.time.values[time_idx]
    ax.set_title(f"Currents — Oceananigans LES  (t = {t_value})")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="speed |u| [m/s]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_volume(field_zyx: np.ndarray, xc, yc, zc,
                vol_grid: int, colorscale: str,
                title: str, value_label: str,
                sources: list[dict] | None, out_path: Path) -> None:
    """3D Plotly volume rendering. Downsamples field + coords to ~vol_grid per axis."""
    sx = max(len(xc) // vol_grid, 1)
    sy = max(len(yc) // vol_grid, 1)
    sz = max(len(zc) // vol_grid, 1)
    xd, yd, zd = xc[::sx], yc[::sy], zc[::sz]
    Z, Y, X = np.meshgrid(zd, yd, xd, indexing="ij")
    F = field_zyx[::sz, ::sy, ::sx]
    fmin, fmax = float(F.min()), float(F.max())
    if fmax <= fmin:
        fmax = fmin + 1e-9

    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),                     # already negative → surface naturally at top
        value=F.flatten(),
        isomin=fmin + 0.05 * (fmax - fmin),
        isomax=fmax,
        opacity=0.1,
        opacityscale=[[0.0, 0.0], [0.2, 0.05], [0.5, 0.2], [1.0, 0.8]],
        surface_count=20,
        colorscale=colorscale,
        colorbar=dict(title=value_label),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name=value_label,
    ))

    if sources:
        fig.add_trace(go.Scatter3d(
            x=[s["x"] for s in sources],
            y=[s["y"] for s in sources],
            z=[-float(s["depth"]) for s in sources],
            mode="markers+text",
            text=[s["name"] for s in sources],
            textposition="top center",
            marker=dict(size=6, color="red", symbol="x"),
            name="sources",
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"saved {out_path}")


def main() -> None:
    args = parse_args()
    nc_path = DATA_DIR / f"scenario_01_{args.season}.nc"
    if not nc_path.exists():
        sys.exit(f"NetCDF not found: {nc_path}\n"
                 f"Run `cd oceananigans && julia --project=. langmuir.jl` first "
                 f"(with SEASON = :{args.season}).")

    ds = xr.open_dataset(nc_path)
    print(f"loaded {nc_path}")
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  vars: {list(ds.data_vars)}")

    # Reference cell-center grid from T.
    _, xc = _coord(ds.T, "x")
    _, yc = _coord(ds.T, "y")
    _, zc = _coord(ds.T, "z")

    sources = json.loads(SOURCES_FILE.read_text())
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    t_idx = args.time_idx

    # 1) Currents
    plot_currents(ds, t_idx, xc, yc, zc, args.stride, sources,
                  PLOT_DIR / f"currents_{args.season}.png")

    # 2) Salinity
    S = _to_center_zyx(ds.S.isel(time=t_idx), xc, yc, zc)
    plot_volume(S, xc, yc, zc, args.vol_grid, "Viridis",
                f"Salinity [PSU] — {args.season}", "S",
                sources, PLOT_DIR / f"salinity_{args.season}.html")

    # 3) Temperature
    T_field = _to_center_zyx(ds.T.isel(time=t_idx), xc, yc, zc)
    plot_volume(T_field, xc, yc, zc, args.vol_grid, "Plasma",
                f"Temperature [°C] — {args.season}", "T",
                None, PLOT_DIR / f"temperature_{args.season}.html")

    # 4) Turbidity (analytical Beer-Lambert, depth-only, broadcast to 3D)
    depth_pos = -zc                         # positive-down depth values
    tau_1d = compute_turbidity(depth_pos, k=args.k_turbidity)
    tau_3d = np.broadcast_to(
        tau_1d[:, None, None], (len(zc), len(yc), len(xc))
    ).copy()
    plot_volume(tau_3d, xc, yc, zc, args.vol_grid, "Greys",
                f"Turbidity τ — k={args.k_turbidity} [1/m]", "τ",
                None, PLOT_DIR / f"turbidity_{args.season}.html")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

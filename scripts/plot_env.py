"""
Visualize a precomputed environment bundle (env_*.nc).

Produces two figures:
  1. 3D quiver of the current field (u, v, w) with industrial sources marked.
  2. 3D scatter of the reward function R(s, tau) so local/global optima are visible.

The reward used here is a Gaussian over (salinity, turbidity) centered at user-
chosen targets (s*, tau*) — same shape the agent will optimize. Tweak --s-target,
--tau-target, --sigma-s, --sigma-tau on the CLI.

Usage:
    uv run scripts/plot_env.py data/envs/env_000.nc
    uv run scripts/plot_env.py data/envs/env_000.nc --s-target 1e-5 --tau-target 0.7
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from src.models.turbidity import turbidity_model  # noqa: E402


def reward_field(salinity: np.ndarray, turbidity: np.ndarray,
                 s_target: float, tau_target: float,
                 sigma_s: float, sigma_tau: float) -> np.ndarray:
    """Gaussian reward centered at (s_target, tau_target)."""
    ds = (salinity - s_target) / sigma_s
    dt = (turbidity - tau_target) / sigma_tau
    return np.exp(-(ds**2 + dt**2))


def plot_currents(ds: xr.Dataset, sources: list[dict] | None,
                  stride: int = 2, fig_path: Path | None = None) -> None:
    z = ds.z.values
    lat = ds.latitude.values
    lon = ds.longitude.values

    # Subsample for legibility
    s = stride
    Z, LAT, LON = np.meshgrid(z, lat[::s], lon[::s], indexing="ij")
    U = ds.u.values[:, ::s, ::s]
    V = ds.v.values[:, ::s, ::s]
    W = ds.w.values[:, ::s, ::s]
    speed = np.sqrt(U**2 + V**2 + W**2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color arrows by speed
    smin, smax = float(speed.min()), float(speed.max())
    norm = plt.Normalize(smin, smax if smax > smin else smin + 1e-9)
    colors = plt.cm.viridis(norm(speed.ravel()))

    ax.quiver(
        LON, LAT, Z, U, V, W,
        length=0.05, normalize=True, colors=colors, linewidth=0.6,
    )

    if sources:
        for src in sources:
            ax.scatter(
                src["lon"], src["lat"], -src["depth"],
                c="red", s=80, marker="X",
                edgecolors="black", linewidths=1.0,
                label=src["name"],
            )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("z [m]  (negative = depth)")
    ax.set_title(f"Currents — {ds.attrs.get('cmems_time', 'unknown')}")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="speed [m/s]")

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")


def plot_reward(ds: xr.Dataset, s_target: float, tau_target: float,
                sigma_s: float, sigma_tau: float,
                k_turbidity: float = 0.3,
                resolution: tuple[int, int, int] = (40, 60, 60),
                sources: list[dict] | None = None,
                fig_path: Path | None = None,
                show: bool = True) -> None:
    """Render R(lat, lon, z) as a continuous 3D volume.

    Salinity is bilinearly interpolated to a denser grid; turbidity is
    evaluated analytically at every dense depth (Beer-Lambert is continuous
    in z by construction). Reward is then computed cell-wise on the dense
    grid and rendered as a semi-transparent volume in plotly.
    """
    lat = ds.latitude.values
    lon = ds.longitude.values
    z = ds.z.values

    nz, nlat, nlon = resolution
    z_dense = np.linspace(z.min(), z.max(), nz)            # negative-down
    lat_dense = np.linspace(lat.min(), lat.max(), nlat)
    lon_dense = np.linspace(lon.min(), lon.max(), nlon)

    sal_dense = ds.salinity.interp(
        z=z_dense, latitude=lat_dense, longitude=lon_dense,
    ).values

    # Turbidity is analytic and continuous in depth — recompute, don't interp.
    tau_dense = turbidity_model(z_dense, k=k_turbidity)
    tau_3d = np.broadcast_to(tau_dense[:, None, None], sal_dense.shape)

    R = reward_field(sal_dense, tau_3d, s_target, tau_target, sigma_s, sigma_tau)
    R_max = float(R.max())

    Z, LAT, LON = np.meshgrid(z_dense, lat_dense, lon_dense, indexing="ij")

    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=LON.flatten(),
        y=LAT.flatten(),
        z=Z.flatten(),
        value=R.flatten(),
        isomin=0.05 * R_max,
        isomax=R_max,
        opacity=0.1,                                      # base transparency
        opacityscale=[                                    # ramp: low R = invisible
            [0.0, 0.0],
            [0.2, 0.05],
            [0.5, 0.2],
            [1.0, 0.8],
        ],
        surface_count=20,
        colorscale="Plasma",
        colorbar=dict(title="reward"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="reward",
    ))

    # Global maximum
    imax = int(np.argmax(R))
    iz, ilat, ilon = np.unravel_index(imax, R.shape)
    fig.add_trace(go.Scatter3d(
        x=[lon_dense[ilon]], y=[lat_dense[ilat]], z=[z_dense[iz]],
        mode="markers",
        marker=dict(size=8, color="lime",
                    line=dict(color="black", width=1), symbol="diamond"),
        name=f"global max R={R.flat[imax]:.3f}",
    ))

    if sources:
        fig.add_trace(go.Scatter3d(
            x=[s["lon"] for s in sources],
            y=[s["lat"] for s in sources],
            z=[-s["depth"] for s in sources],
            mode="markers+text",
            text=[s["name"] for s in sources],
            textposition="top center",
            marker=dict(size=6, color="red", symbol="x"),
            name="sources",
        ))

    fig.update_layout(
        title=(f"Reward landscape — s*={s_target:g}, τ*={tau_target:g}, "
               f"σ=({sigma_s:g}, {sigma_tau:g})"),
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="z [m]",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if fig_path:
        fig.write_html(str(fig_path))
        print(f"saved {fig_path}")
    if show:
        fig.show()


def load_sources_for_env(env_path: Path) -> list[dict] | None:
    """Try to read the source catalog from the bundle's manifest.json."""
    manifest_path = env_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open() as f:
        manifest = json.load(f)
    return manifest.get("sources")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("env_file", type=Path, help="path to env_*.nc")
    p.add_argument("--s-target", type=float, default=None,
                   help="optimal salinity excess [PSU]; default = 80%% of field max")
    p.add_argument("--tau-target", type=float, default=0.6,
                   help="optimal turbidity in [0,1]")
    p.add_argument("--sigma-s", type=float, default=None,
                   help="salinity bandwidth; default = 0.25 * field max")
    p.add_argument("--sigma-tau", type=float, default=0.15)
    p.add_argument("--stride", type=int, default=2,
                   help="quiver subsampling stride in lat/lon")
    p.add_argument("--save-dir", type=Path, default="data/plots",
                   help="if set, save PNGs here instead of showing interactively")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = xr.open_dataset(args.env_file)

    s_max = float(ds.salinity.max())
    s_target = args.s_target if args.s_target is not None else 0.8 * max(s_max, 1e-12)
    sigma_s = args.sigma_s if args.sigma_s is not None else 0.25 * max(s_max, 1e-12)

    sources = load_sources_for_env(args.env_file)

    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    stem = args.env_file.stem
    fig_curr = save_dir / f"{stem}_currents.png" if save_dir else None
    fig_rew = save_dir / f"{stem}_reward.html" if save_dir else None

    plot_currents(ds, sources, stride=args.stride, fig_path=fig_curr)
    k_turb = float(ds.attrs.get("k_turbidity", 0.3))
    plot_reward(
        ds,
        s_target=s_target,
        tau_target=args.tau_target,
        sigma_s=sigma_s,
        sigma_tau=args.sigma_tau,
        k_turbidity=k_turb,
        sources=sources,
        fig_path=fig_rew,
        show=not args.no_show,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

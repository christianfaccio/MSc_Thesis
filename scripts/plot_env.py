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
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


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
                size_min: float = 2.0, size_max: float = 250.0,
                fig_path: Path | None = None) -> None:
    z = ds.z.values
    lat = ds.latitude.values
    lon = ds.longitude.values
    salinity = ds.salinity.values
    turbidity = ds.turbidity.values

    R = reward_field(salinity, turbidity, s_target, tau_target, sigma_s, sigma_tau)
    R_max = float(R.max())
    R_norm = R / R_max if R_max > 0 else R

    Z, LAT, LON = np.meshgrid(z, lat, lon, indexing="ij")

    xs = LON.ravel()
    ys = LAT.ravel()
    zs = Z.ravel()
    rs = R.ravel()
    rs_n = R_norm.ravel()

    sizes = size_min + (size_max - size_min) * rs_n
    alphas = 0.05 + 0.95 * rs_n   # faint for low reward, solid at the optima

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0.0, vmax=max(R_max, 1e-12))
    rgba = cmap(norm(rs))
    rgba[:, 3] = alphas                            # per-point alpha

    ax.scatter(xs, ys, zs, s=sizes, c=rgba, edgecolors="none")

    # Global maximum
    imax = int(np.argmax(R))
    iz, ilat, ilon = np.unravel_index(imax, R.shape)
    ax.scatter(
        lon[ilon], lat[ilat], z[iz],
        marker="*", c="lime", s=350,
        edgecolors="black", linewidths=1.2,
        label=f"global max R={R.flat[imax]:.3f}",
    )
    ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("z [m]")
    ax.set_title(
        f"Reward landscape — target (s*={s_target:g}, τ*={tau_target:g}), "
        f"σ=({sigma_s:g}, {sigma_tau:g})"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="reward")

    if fig_path:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"saved {fig_path}")


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
    fig_rew = save_dir / f"{stem}_reward.png" if save_dir else None

    plot_currents(ds, sources, stride=args.stride, fig_path=fig_curr)
    plot_reward(
        ds,
        s_target=s_target,
        tau_target=args.tau_target,
        sigma_s=sigma_s,
        sigma_tau=args.sigma_tau,
        fig_path=fig_rew,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

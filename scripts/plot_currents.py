"""Top-down quiver plot of SwarmSwIM currents driven by an XML config (2D or 3D)."""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from SwarmSwIM.sim_functions import (
    DepthDecayCurrent,
    EkmanSpiral,
    LayeredCurrent,
    TimeNoise,
    VortexField,
    global_waves as global_waves_fn,
    local_waves as local_waves_fn,
    parse_envrioment_parameters,
)

COMPONENT_NAMES = [
    "uniform",
    "noise",
    "vortex",
    "global_waves",
    "local_waves",
    "current_3d",
]


class AgentStub:
    """Minimal Agent stand-in: current models only read `pos` and `name`."""

    def __init__(self, x, y, z=0.0, name="probe"):
        self.pos = np.array([x, y, z], dtype=float)
        self.name = name


def build_current_3d(data):
    model = data.get("current_3d_model", "none")
    params = data.get("current_3d_params", {})
    if model == "depth_decay":
        return DepthDecayCurrent(**params)
    if model == "ekman":
        return EkmanSpiral(**params)
    if model == "layered":
        return LayeredCurrent(layers=params["layers"])
    if model in ("data", "precomputed"):
        raise ValueError(
            f"current_3d model '{model}' needs a NetCDF; this viz only handles synthetic models."
        )
    raise ValueError(f"current_3d model '{model}' is not enabled in the XML.")


def select_components(data, requested, t0):
    rng = np.random.default_rng(data.get("seed"))
    if "all" in requested:
        requested = set(COMPONENT_NAMES)

    comps = {}
    for name in requested:
        if name == "uniform" and data.get("is_uniform_current"):
            comps["uniform"] = np.asarray(data["uniform_current"], dtype=float)
        elif name == "noise" and data.get("is_noise_currents"):
            comps["noise"] = TimeNoise(
                time=t0,
                freq=data["noise_currents_freq"],
                intensity=data["noise_currents_intensity"],
                rng=rng,
            )
        elif name == "vortex" and data.get("is_vortex_currents"):
            comps["vortex"] = VortexField(
                density=data["vortex_currents_density"],
                intensity=data["vortex_currents_intensity"],
                rng=rng,
            )
        elif name == "global_waves" and data.get("is_global_waves"):
            comps["global_waves"] = data["global_waves"]
        elif name == "local_waves" and data.get("is_local_waves"):
            comps["local_waves"] = data["local_waves"]
        elif name == "current_3d" and data.get("is_current_3d"):
            comps["current_3d"] = build_current_3d(data)
    return comps


def sample_single(name, x, y, z, t, comps):
    agent = AgentStub(x, y, z)
    v = np.zeros(3)
    if name == "uniform":
        v += comps["uniform"]
    elif name == "noise":
        v += comps["noise"].calculate_noises(t, agent)
    elif name == "vortex":
        v += comps["vortex"].current_vortex_calculate(agent)
    elif name == "global_waves":
        for wp in comps["global_waves"]:
            v += global_waves_fn(t, **wp)
    elif name == "local_waves":
        for wp in comps["local_waves"]:
            wp2 = dict(wp)
            # XML uses 'wavelength'; upstream function signature has the typo 'wavelenght'.
            if "wavelength" in wp2:
                wp2["wavelenght"] = wp2.pop("wavelength")
            v += local_waves_fn(t, agent, **wp2)
    elif name == "current_3d":
        v += comps["current_3d"].calculate(agent)
    return v


def compute_fields_2d(comps, t, depth, grid_n, extent):
    xs = np.linspace(extent[0], extent[1], grid_n)
    ys = np.linspace(extent[2], extent[3], grid_n)
    X, Y = np.meshgrid(xs, ys)
    fields = {}
    for name in comps:
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(grid_n):
            for j in range(grid_n):
                vec = sample_single(name, X[i, j], Y[i, j], depth, t, comps)
                U[i, j], V[i, j] = vec[0], vec[1]
        fields[name] = (U, V)
    return X, Y, fields


def compute_fields_3d(comps, t, grid_n, depth_n, extent, depth_range):
    xs = np.linspace(extent[0], extent[1], grid_n)
    ys = np.linspace(extent[2], extent[3], grid_n)
    zs = np.linspace(depth_range[0], depth_range[1], depth_n)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    fields = {}
    for name in comps:
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        W = np.zeros_like(Z)
        for k in range(depth_n):
            for i in range(grid_n):
                for j in range(grid_n):
                    vec = sample_single(name, X[k, i, j], Y[k, i, j], Z[k, i, j], t, comps)
                    U[k, i, j], V[k, i, j], W[k, i, j] = vec[0], vec[1], vec[2]
        fields[name] = (U, V, W)
    return X, Y, Z, fields


def _quiver2d_one(ax, X, Y, U, V, title, vmax):
    mag = np.sqrt(U ** 2 + V ** 2)
    q = ax.quiver(X, Y, U, V, mag, cmap="viridis", pivot="mid", clim=(0, vmax))
    ax.set_aspect("equal")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    return q


def plot_combined_2d(X, Y, fields, title, savepath):
    U = sum(f[0] for f in fields.values())
    V = sum(f[1] for f in fields.values())
    vmax = float(np.sqrt(U ** 2 + V ** 2).max()) or 1.0
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    q = _quiver2d_one(ax, X, Y, U, V, title, vmax)
    plt.colorbar(q, ax=ax, label="speed |u,v| [m/s]")
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()


def plot_breakdown_2d(X, Y, fields, title, savepath):
    Usum = sum(f[0] for f in fields.values())
    Vsum = sum(f[1] for f in fields.values())
    panels = list(fields.items()) + [("sum", (Usum, Vsum))]

    vmax = 0.0
    for _, (U, V) in panels:
        vmax = max(vmax, float(np.sqrt(U ** 2 + V ** 2).max()))
    vmax = vmax or 1.0

    n = len(panels)
    ncols = 3 if n > 4 else 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    q = None
    flat_axes = list(axes.flat)
    for ax, (name, (U, V)) in zip(flat_axes, panels):
        q = _quiver2d_one(ax, X, Y, U, V, name, vmax)
    for ax in flat_axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.colorbar(q, ax=flat_axes, label="speed |u,v| [m/s]", shrink=0.8)
    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()


def plot_combined_3d(X, Y, Z, fields, title, savepath, extent, depth_range):
    Usum = sum(f[0] for f in fields.values())
    Vsum = sum(f[1] for f in fields.values())
    Wsum = sum(f[2] for f in fields.values())

    max_mag = float(np.sqrt(Usum ** 2 + Vsum ** 2 + Wsum ** 2).max()) or 1.0
    domain_size = max(extent[1] - extent[0], extent[3] - extent[2])
    arrow_scale = (domain_size * 0.10) / max_mag

    fig = plt.figure(figsize=(10.0, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("viridis_r")
    zmin, zmax = depth_range
    zspan = max(zmax - zmin, 1e-9)
    depth_n = X.shape[0]
    for k in range(depth_n):
        depth_frac = (Z[k, 0, 0] - zmin) / zspan
        color = cmap(depth_frac)
        ax.quiver(
            X[k].flatten(), Y[k].flatten(), Z[k].flatten(),
            Usum[k].flatten(), Vsum[k].flatten(), Wsum[k].flatten(),
            length=arrow_scale, normalize=False, color=color, linewidth=0.9,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("depth [m]")
    ax.invert_zaxis()
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(zmin, zmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="depth [m]", shrink=0.7, pad=0.1)

    if savepath:
        plt.savefig(savepath, dpi=140, bbox_inches="tight")
    plt.show()


def print_summary_2d(fields):
    print("Per-component horizontal speed on grid [m/s]:")
    for name, (U, V) in fields.items():
        mag = np.sqrt(U ** 2 + V ** 2)
        print(f"  {name:14s}  max={mag.max():.4f}  mean={mag.mean():.4f}")
    Usum = sum(f[0] for f in fields.values())
    Vsum = sum(f[1] for f in fields.values())
    msum = np.sqrt(Usum ** 2 + Vsum ** 2)
    print(f"  {'sum':14s}  max={msum.max():.4f}  mean={msum.mean():.4f}")


def print_summary_3d(fields):
    print("Per-component speed on 3D grid [m/s]:")
    for name, (U, V, W) in fields.items():
        h = np.sqrt(U ** 2 + V ** 2)
        m = np.sqrt(U ** 2 + V ** 2 + W ** 2)
        print(f"  {name:14s}  |u,v| max={h.max():.4f}  |u,v,w| max={m.max():.4f}")
    Usum = sum(f[0] for f in fields.values())
    Vsum = sum(f[1] for f in fields.values())
    Wsum = sum(f[2] for f in fields.values())
    m = np.sqrt(Usum ** 2 + Vsum ** 2 + Wsum ** 2)
    print(f"  {'sum':14s}  |u,v,w| max={m.max():.4f}  mean={m.mean():.4f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--xml",
        default="config/simulation.xml",
        help="path to simulation XML (default: config/simulation.xml)",
    )
    p.add_argument(
        "--components",
        nargs="+",
        required=True,
        choices=COMPONENT_NAMES + ["all"],
        metavar="NAME",
        help="components to show: any of " + ", ".join(COMPONENT_NAMES) + ", or 'all'",
    )
    p.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="2d",
        help="2d: horizontal slice at --depth; 3d: stack of horizontal layers across --depth-range",
    )
    p.add_argument("--time", type=float, default=0.0, help="simulation time t [s]")
    p.add_argument(
        "--grid",
        type=int,
        default=None,
        help="horizontal grid per side (default: 25 in 2d, 10 in 3d)",
    )
    p.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[0.0, 100.0, 0.0, 100.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    p.add_argument(
        "--depth",
        type=float,
        default=0.0,
        help="2d mode: z slice depth [m]; ignored in 3d mode",
    )
    p.add_argument(
        "--depth-range",
        type=float,
        nargs=3,
        default=[0.0, 50.0, 8.0],
        metavar=("ZMIN", "ZMAX", "N"),
        help="3d mode: depth range [m] and number of layers",
    )
    p.add_argument(
        "--breakdown",
        action="store_true",
        help="2d only: each component in its own subplot on a shared color scale",
    )
    p.add_argument("--save", default=None, help="optional output png path")
    return p.parse_args()


def main():
    args = parse_args()
    data = parse_envrioment_parameters(args.xml)
    requested = set(args.components)
    comps = select_components(data, requested, t0=args.time)

    if not comps:
        print("Nothing to plot: no requested component is enabled in the XML.")
        sys.exit(1)

    skipped = (requested - {"all"}) - set(comps.keys())
    if skipped:
        print(f"Skipped (disabled in XML): {', '.join(sorted(skipped))}")
    print(f"Plotting: {', '.join(sorted(comps.keys()))}")

    grid_n = args.grid if args.grid is not None else (25 if args.mode == "2d" else 10)
    label = "+".join(sorted(comps.keys()))

    if args.mode == "2d":
        X, Y, fields = compute_fields_2d(comps, args.time, args.depth, grid_n, args.extent)
        print_summary_2d(fields)
        title = (
            f"Currents [{label}]  z={args.depth:g} m  t={args.time:g} s  "
            f"({os.path.basename(args.xml)})"
        )
        if args.breakdown:
            plot_breakdown_2d(X, Y, fields, title, args.save)
        else:
            plot_combined_2d(X, Y, fields, title, args.save)
    else:
        if args.breakdown:
            print("--breakdown is ignored in 3d mode.")
        zmin, zmax, n_f = args.depth_range
        depth_n = max(int(n_f), 2)
        X, Y, Z, fields = compute_fields_3d(
            comps, args.time, grid_n, depth_n, args.extent, (zmin, zmax)
        )
        print_summary_3d(fields)
        title = (
            f"Currents [{label}]  3D  z∈[{zmin:g},{zmax:g}] m  "
            f"t={args.time:g} s  ({os.path.basename(args.xml)})"
        )
        plot_combined_3d(X, Y, Z, fields, title, args.save, args.extent, (zmin, zmax))


if __name__ == "__main__":
    main()

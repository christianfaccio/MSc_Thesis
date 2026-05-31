"""
Render the Beer-Lambert turbidity field τ(d) over a 3D synthetic domain.

τ(d) = 1 - exp(-k · |d|),  so the field is depth-only — the 3D volume shows
horizontal bands that darken with depth. A 1D profile is also rendered as a
quick sanity check on the chosen k value.

Usage:
    uv run -m scripts.plot_turbidity
    uv run -m scripts.plot_turbidity --k 0.1
    uv run -m scripts.plot_turbidity --save data/plots/turbidity.html
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.models.turbidity import compute_turbidity  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=float, default=0.05,
                   help="Beer-Lambert attenuation coefficient [1/m]")
    p.add_argument("--extent", type=float, nargs=2, default=[0.0, 100.0],
                   help="x and y domain [min max] in meters")
    p.add_argument("--depth-range", type=float, nargs=2, default=[0.0, 100.0],
                   help="depth domain [min max] in meters (positive down)")
    p.add_argument("--grid", type=int, default=40,
                   help="cells per axis")
    p.add_argument("--save", type=Path, default=None,
                   help="output HTML path for the 3D volume")
    p.add_argument("--save-profile", type=Path, default=None,
                   help="output PNG path for the 1D τ(d) profile")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    xs = np.linspace(args.extent[0], args.extent[1], args.grid)
    ys = np.linspace(args.extent[0], args.extent[1], args.grid)
    zs = np.linspace(args.depth_range[0], args.depth_range[1], args.grid)

    # 1D profile (turbidity is depth-only — line plot is the most honest view)
    fig1, ax = plt.subplots(figsize=(5, 6))
    tau_1d = compute_turbidity(zs, k=args.k)
    ax.plot(tau_1d, zs)
    ax.invert_yaxis()                                      # surface on top
    ax.set_xlabel("turbidity τ ∈ [0, 1]")
    ax.set_ylabel("depth [m, positive down]")
    ax.set_title(f"Beer-Lambert τ(d) — k = {args.k} [1/m]")
    ax.grid(True, alpha=0.3)
    if args.save_profile:
        args.save_profile.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(args.save_profile, dpi=150, bbox_inches="tight")
        print(f"saved {args.save_profile}")

    # 3D volume (just depth bands, but useful for visual coherence with salinity render)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    tau_3d = compute_turbidity(Z, k=args.k)

    fig2 = go.Figure()
    fig2.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=-Z.flatten(),                                    # negate so surface is on top
        value=tau_3d.flatten(),
        isomin=0.0,
        isomax=1.0,
        opacity=0.1,
        surface_count=15,
        colorscale="Greys",
        colorbar=dict(title="τ"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="turbidity",
    ))
    fig2.update_layout(
        title=f"Turbidity volume — k = {args.k} [1/m]  (depth-only field)",
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig2.write_html(str(args.save))
        print(f"saved {args.save}")

    if not args.no_show:
        fig2.show()
        plt.show()


if __name__ == "__main__":
    main()

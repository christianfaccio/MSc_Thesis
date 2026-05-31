"""
Render the analytical salinity field S(x, y, z) over a 3D synthetic domain.

S is a sum of 3D Gaussian blobs centered at each source (see
`compute_salinity_analytical` in src/models/salinity.py). The plot is a
plotly Volume rendering with red X markers at the source positions.

Usage:
    uv run -m scripts.plot_salinity
    uv run -m scripts.plot_salinity --sigma-h 20 --sigma-v 8
    uv run -m scripts.plot_salinity --sources-file config/sources_synthetic.json --save data/plots/salinity.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.models.salinity import compute_salinity_analytical  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources-file", type=Path,
                   default=REPO_ROOT / "config" / "sources.json")
    p.add_argument("--sigma-h", type=float, default=15.0,
                   help="horizontal patch sigma [m]")
    p.add_argument("--sigma-v", type=float, default=10.0,
                   help="vertical patch sigma [m]")
    p.add_argument("--extent", type=float, nargs=2, default=[0.0, 100.0],
                   help="x and y domain [min max] in meters")
    p.add_argument("--depth-range", type=float, nargs=2, default=[0.0, 100.0],
                   help="depth domain [min max] in meters (positive down)")
    p.add_argument("--grid", type=int, default=40,
                   help="cells per axis (cubed for total grid size)")
    p.add_argument("--save", type=Path, default=None,
                   help="output HTML path (omit to skip saving)")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sources = json.loads(args.sources_file.read_text())

    xs = np.linspace(args.extent[0], args.extent[1], args.grid)
    ys = np.linspace(args.extent[0], args.extent[1], args.grid)
    zs = np.linspace(args.depth_range[0], args.depth_range[1], args.grid)

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    S = compute_salinity_analytical(X, Y, Z, sources, args.sigma_h, args.sigma_v)
    S_max = float(S.max())

    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=-Z.flatten(),                                     # negate so surface is on top
        value=S.flatten(),
        isomin=0.05 * S_max,
        isomax=S_max,
        opacity=0.1,
        opacityscale=[
            [0.0, 0.0],
            [0.2, 0.05],
            [0.5, 0.2],
            [1.0, 0.8],
        ],
        surface_count=20,
        colorscale="Viridis",
        colorbar=dict(title="salinity"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="salinity",
    ))

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
        title=(f"Analytical salinity — σ_h={args.sigma_h} m, σ_v={args.sigma_v} m, "
               f"{len(sources)} sources"),
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(args.save))
        print(f"saved {args.save}")

    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()

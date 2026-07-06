"""
Functions needed to plot env fields.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def _coord(da: xr.DataArray, axis: str) -> tuple[str, np.ndarray]:
    """Return (dim name, values) for a spatial axis ('x'/'y'/'z') of `da`."""
    for d in da.dims:
        if d.lower().startswith(axis):
            return d, da.coords[d].values
    raise KeyError(f"no '{axis}' coord on {da.name} (dims={da.dims})")

def _domain_edges(c) -> tuple[float, float]:
    """True domain extent [edge_lo, edge_hi] from cell-CENTER coords `c`.

    Oceananigans cell centers sit half a cell in from the boundary, so the center
    range (e.g. 3.9 … 996.1) understates the real domain (0 … 1000). Sources are
    anchored on the boundary (x=0 / y=0), so filtering or ranging against the
    center min/max wrongly drops them — use the half-cell-padded edges instead.
    """
    c = np.asarray(c, dtype=float)
    if c.size < 2:
        return float(c.min()), float(c.max())
    d = (float(c.max()) - float(c.min())) / (c.size - 1)
    return float(c.min()) - d / 2, float(c.max()) + d / 2

def _fmt_time(t) -> str:
    """Human-readable label for an Oceananigans time coord value.

    Oceananigans writes elapsed time as a timedelta64; show it in hours. Falls
    back to str() for datetime64 or anything unexpected.
    """
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.timedelta64):
        hours = t.astype("timedelta64[s]").astype(float) / 3600.0
        return f"{hours:.1f} h"
    return str(t)

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

def plot_currents_netcdf(ds: xr.Dataset, time_idx: int, xc, yc, zc,
                  stride: int, z_aspect: float, sources: list[dict], out_path: Path) -> None:
    u = _to_center_zyx(ds.u.isel(time=time_idx), xc, yc, zc)
    v = _to_center_zyx(ds.v.isel(time=time_idx), xc, yc, zc)
    w = _to_center_zyx(ds.w.isel(time=time_idx), xc, yc, zc)

    s = stride
    Z, Y, X = np.meshgrid(zc[::s], yc[::s], xc[::s], indexing="ij")
    ud, vd, wd = u[::s, ::s, ::s], v[::s, ::s, ::s], w[::s, ::s, ::s]
    speed = np.sqrt(ud ** 2 + vd ** 2 + wd ** 2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_range = float(xc.max() - xc.min()) or 1.0
    y_range = float(yc.max() - yc.min()) or 1.0
    z_range = float(zc.max() - zc.min()) or 1.0

    # The domain is wide and shallow (e.g. 1 km x 1 km x 40 m). The default cube
    # aspect ratio stretches the z-axis and visually inflates every vertical
    # velocity component, drowning the horizontal flow. Draw the z-box at a fixed
    # fraction (z_aspect) of the horizontal extent instead, independent of domain.
    z_box = z_aspect * max(x_range, y_range)
    ax.set_box_aspect((x_range, y_range, z_box))
    # Effective vertical exaggeration that set_box_aspect applies to a unit of z.
    z_exag = z_box / z_range

    smin, smax = float(speed.min()), float(speed.max())
    norm = plt.Normalize(smin, smax if smax > smin else smin + 1e-9)
    colors = plt.cm.viridis(norm(speed.ravel()))

    # Normalize by the display-space magnitude (w scaled by z_exag) so every arrow
    # has the same on-screen length. z_exag cancels in the data-space components
    # passed to quiver because set_box_aspect re-applies it during rendering.
    disp_mag = np.sqrt(ud ** 2 + vd ** 2 + (wd * z_exag) ** 2)
    disp_mag[disp_mag == 0] = 1.0
    spacing = float(xc[stride] - xc[0]) if len(xc) > stride else x_range
    arrow = 0.6 * abs(spacing)
    U = ud / disp_mag * arrow
    V = vd / disp_mag * arrow
    W = wd / disp_mag * arrow

    # Z values are negative (Oceananigans convention: surface at z=0, depths at z<0).
    # matplotlib will naturally place larger (less negative) values toward the top.
    ax.quiver(X, Y, Z, U, V, W, normalize=False, colors=colors, linewidth=0.6)

    # Only draw markers inside this file's domain (edges, not cell centers, so
    # boundary sources at x=0 / y=0 are kept). A mismatched catalog (e.g. 5 km
    # coords on a 1 km file) would otherwise blow the 3D axes out to its extent.
    x_lo, x_hi = _domain_edges(xc)
    y_lo, y_hi = _domain_edges(yc)
    z_lo, _ = _domain_edges(zc)
    n_drawn = 0
    if sources:
        for src in sources:
            if not (x_lo <= src["x"] <= x_hi and y_lo <= src["y"] <= y_hi):
                continue
            ax.scatter(src["x"], src["y"], -float(src["depth"]),
                       c="red", s=80, marker="X",
                       edgecolors="black", linewidths=1.0, label=src["name"])
            n_drawn += 1
        if n_drawn:
            ax.legend(loc="upper right", fontsize=8)
        if n_drawn < len(sources):
            print(f"  (skipped {len(sources) - n_drawn} source marker(s) outside "
                  f"the file domain — pass the matching sidecar via --sources-file)")

    # Pin the axes to the domain edges so nothing can stretch them past the domain.
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, 0.0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]  (surface at 0)")
    elapsed = ds.time.values[time_idx] - ds.time.values[0]
    ax.set_title(f"Currents — Oceananigans  (t = {_fmt_time(elapsed)})")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="speed |u| [m/s]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")

def plot_surface_currents_netcdf(ds: xr.Dataset, time_idx: int, xc, yc, zc,
                  density: float, sources: list[dict] | None, out_path: Path) -> None:
    """Top-down (plan-view) map of the SURFACE horizontal currents at one snapshot.

    Streamlines colored by horizontal speed |u, v| at the topmost level (z nearest 0),
    so the meandering jet and eddies read clearly. `density` controls streamline
    spacing (matplotlib streamplot density).
    """
    # u, v live on the C-grid; interpolate to cell centers, then take the surface.
    u = _to_center_zyx(ds.u.isel(time=time_idx), xc, yc, zc)
    v = _to_center_zyx(ds.v.isel(time=time_idx), xc, yc, zc)
    k = int(np.argmax(zc))          # z closest to 0 = surface
    us, vs = u[k], v[k]             # (ny, nx)
    speed = np.sqrt(us ** 2 + vs ** 2)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")

    smax = float(np.nanmax(speed))
    strm = ax.streamplot(xc, yc, us, vs, color=speed, cmap="viridis",
                         density=density, linewidth=1.0,
                         norm=plt.Normalize(0.0, smax if smax > 0 else 1e-9))
    fig.colorbar(strm.lines, ax=ax, shrink=0.8, label="surface speed |u, v| [m/s]")

    if sources:
        ax.scatter([s["x"] for s in sources], [s["y"] for s in sources],
                   c="red", s=80, marker="X", edgecolors="black", linewidths=1.0,
                   zorder=5)

    ax.set_xlim(float(xc.min()), float(xc.max()))
    ax.set_ylim(float(yc.min()), float(yc.max()))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    elapsed = ds.time.values[time_idx] - ds.time.values[0]
    ax.set_title(f"Surface currents — Oceananigans  (t = {_fmt_time(elapsed)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

def plot_volume_netcdf(field_zyx: np.ndarray, xc, yc, zc,
                vol_grid: int, colorscale: str,
                title: str, value_label: str, z_aspect: float,
                sources: list[dict] | None, out_path: Path,
                target_pts: np.ndarray | None = None,
                target_marker: dict | None = None) -> None:
    """3D Plotly volume rendering. Downsamples field + coords to ~vol_grid per axis.

    Optional target overlay:
      - target_pts: (N, 3) array of (x, y, z) points in the domain whose (S, τ)
        match a sampled target within tolerance — drawn as a small marker cloud.
      - target_marker: dict {x, y, z, text} for the sampled target point itself.
    """
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

    # Target overlay: matching-point cloud + the sampled target marker.
    if target_pts is not None and len(target_pts):
        target_pts = np.asarray(target_pts)
        fig.add_trace(go.Scatter3d(
            x=target_pts[:, 0], y=target_pts[:, 1], z=target_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="crimson", opacity=0.45),
            name="target zone (S*±ε, τ*±ε)",
        ))
    if target_marker is not None:
        fig.add_trace(go.Scatter3d(
            x=[target_marker["x"]], y=[target_marker["y"]], z=[target_marker["z"]],
            mode="markers+text", text=[target_marker.get("text", "target")],
            textposition="top center",
            marker=dict(size=8, color="gold", symbol="diamond",
                        line=dict(color="black", width=1)),
            name="target",
        ))

    # Drop markers outside this file's domain (edges, not cell centers, so
    # boundary sources at x=0 / y=0 are kept; also filters a stale 5 km catalog).
    x_lo, x_hi = _domain_edges(xc)
    y_lo, y_hi = _domain_edges(yc)
    z_lo, _ = _domain_edges(zc)
    if sources:
        in_dom = [s for s in sources
                  if x_lo <= s["x"] <= x_hi and y_lo <= s["y"] <= y_hi]
        if len(in_dom) < len(sources):
            print(f"  (skipped {len(sources) - len(in_dom)} source marker(s) outside "
                  f"the file domain — pass the matching sidecar via --sources-file)")
        if in_dom:
            fig.add_trace(go.Scatter3d(
                x=[s["x"] for s in in_dom],
                y=[s["y"] for s in in_dom],
                z=[-float(s["depth"]) for s in in_dom],
                mode="markers+text",
                text=[s["name"] for s in in_dom],
                textposition="top center",
                marker=dict(size=6, color="red", symbol="x"),
                name="sources",
            ))

    x_range = float(xc.max() - xc.min()) or 1.0
    y_range = float(yc.max() - yc.min()) or 1.0
    max_h = max(x_range, y_range)
    fig.update_layout(
        title=title,
        scene=dict(
            # Pin ranges to the file's domain edges so the axes can never auto-range
            # past it (a stale catalog, or a boundary target/source marker).
            xaxis=dict(title="x [m]", range=[x_lo, x_hi]),
            yaxis=dict(title="y [m]", range=[y_lo, y_hi]),
            zaxis=dict(title="z [m]", range=[z_lo, 0.0]),
            # Match the matplotlib quiver: z-box at a fixed fraction of the
            # horizontal extent, not the default cube that distorts a shallow domain.
            aspectmode="manual",
            aspectratio=dict(x=x_range / max_h, y=y_range / max_h, z=z_aspect),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"saved {out_path}")

def plot_target_zone_netcdf(target_pts: np.ndarray, target_marker: dict | None,
                xc, yc, zc, z_aspect: float, title: str, out_path: Path) -> None:
    """Standalone 3D Plotly view of a target SUCCESS ZONE — every domain point whose
    (S, τ) matches the sampled target within (ε_S, ε_τ), i.e. where an agent counts
    as having reached the target. τ is depth-only, so the zone reads as a layer (or
    a few) at the target depth band. Points are colored by depth; the gold ◆ is the
    sampled target itself. No field volume — just the reachable region.
    """
    x_lo, x_hi = _domain_edges(xc)
    y_lo, y_hi = _domain_edges(yc)
    z_lo, _ = _domain_edges(zc)

    fig = go.Figure()
    if target_pts is not None and len(target_pts):
        pts = np.asarray(target_pts)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2.5, color=pts[:, 2], colorscale="Viridis",
                        colorbar=dict(title="z [m]"), opacity=0.55),
            name="success zone",
        ))
    if target_marker is not None:
        fig.add_trace(go.Scatter3d(
            x=[target_marker["x"]], y=[target_marker["y"]], z=[target_marker["z"]],
            mode="markers+text", text=[target_marker.get("text", "target")],
            textposition="top center",
            marker=dict(size=8, color="gold", symbol="diamond",
                        line=dict(color="black", width=1)),
            name="target",
        ))

    x_range = (x_hi - x_lo) or 1.0
    y_range = (y_hi - y_lo) or 1.0
    max_h = max(x_range, y_range)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x [m]", range=[x_lo, x_hi]),
            yaxis=dict(title="y [m]", range=[y_lo, y_hi]),
            zaxis=dict(title="z [m]", range=[z_lo, 0.0]),
            aspectmode="manual",
            aspectratio=dict(x=x_range / max_h, y=y_range / max_h, z=z_aspect),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"saved {out_path}")

def animate_field_netcdf(ds: xr.Dataset, field_name: str, sources: list[dict] | None,
                  out_path: Path, *, reduce: str = "max", depth: float | None = None,
                  fps: int = 12, frame_stride: int = 1, cmap: str = "viridis",
                  label: str = "", title: str = "") -> None:
    """Animate the time-evolution of a 3D tracer as a top-down (plan-view) GIF.

    The vertical is collapsed to a 2D (time, y, x) map per `reduce`:
      - "max"   → column maximum (shows plumes at any depth; best for salinity),
      - "mean"  → depth average (representative view; best for the T front),
      - "slice" → nearest level to depth `depth` [m, positive-down].
    Frames are temporally subsampled by `frame_stride`. The color scale is fixed
    across all frames (global vmin/vmax) so the evolution is comparable.
    """
    da = ds[field_name]
    xd, xv = _coord(da, "x")
    yd, yv = _coord(da, "y")
    zd, zv = _coord(da, "z")

    if reduce == "max":
        field2d = da.max(dim=zd)
    elif reduce == "mean":
        field2d = da.mean(dim=zd)
    elif reduce == "slice":
        target = -abs(depth if depth is not None else 0.0)
        k = int(np.abs(zv - target).argmin())
        field2d = da.isel({zd: k})
    else:
        raise ValueError(f"reduce must be 'max', 'mean' or 'slice', got {reduce!r}")

    # (time, y, x) numpy stack, temporally subsampled.
    field2d = field2d.transpose("time", yd, xd)
    frames = field2d.values[::frame_stride]
    times = ds.time.values[::frame_stride]
    n_frames = frames.shape[0]

    vmin, vmax = float(np.nanmin(frames)), float(np.nanmax(frames))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    mesh = ax.pcolormesh(xv, yv, frames[0], cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto")
    fig.colorbar(mesh, ax=ax, shrink=0.8,
                 label=label or field_name)

    reduce_tag = (f"slice @ {abs(depth):.0f} m" if reduce == "slice"
                  else f"{reduce} over depth")
    if sources:
        ax.scatter([s["x"] for s in sources], [s["y"] for s in sources],
                   c="red", s=80, marker="X", edgecolors="black", linewidths=1.0,
                   zorder=5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title_base = title or field_name
    txt = ax.set_title(f"{title_base}  ({reduce_tag})\nt = {_fmt_time(times[0])}")

    def update(i):
        # pcolormesh wants the ravelled array for the quadmesh facecolors.
        mesh.set_array(frames[i].ravel())
        txt.set_text(f"{title_base}  ({reduce_tag})\nt = {_fmt_time(times[i])}")
        return mesh, txt

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out_path), writer=animation.PillowWriter(fps=fps))
    except Exception as e:  # Pillow missing or write failure
        plt.close(fig)
        raise RuntimeError(
            f"could not write GIF {out_path} ({e}); ensure Pillow is installed "
            f"(`uv pip install pillow`)."
        ) from e
    plt.close(fig)
    print(f"saved {out_path}  ({n_frames} frames @ {fps} fps)")
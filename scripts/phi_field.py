"""
Visualize the reward-potential field Φ(x,y,z) for a USER-CHOSEN target pair
(S*, τ*), with marginalizations onto each axis and the global-optimal points.

Φ(x,y,z) = exp(-((S(x,y,z)-S*)/σ_s)² - ((τ(z)-τ*)/σ_τ)²),  σ_s=0.5, σ_τ=0.8.

Unlike landscape.py (which uses the episode's own randomly-chosen target), here you
pass S* and τ* on the command line, so you can probe "what does the field look like,
and where are the optima, if the target were this salinity/turbidity?". The field
itself (salinity sources / NetCDF window) is fixed by --seed.

Because turbidity is depth-only, τ* fixes a matching depth z* = -ln(1-τ*)/0.01; the
Φ-max over depth sits on that plane and the horizontal structure comes from salinity.

Shows (saved under debug/out/, and on screen unless --no-show):
  - a single 3D scatter of the global-optimal points (Φ ≥ max-Φ − tol, clustered)
    plus the argmax, with the SUCCESS ZONE drawn as a gold slice at depth z*
    (|ΔS|<ε_S, using the env's training epsilons) — rendered exactly like
    scripts/plot_trajectories.py;
  - an interactive 3D scatter of the same (Plotly HTML).

By default (no --salinity/--turbidity) the target is auto-picked from the value
tail — whichever extreme (<5% or >95%) is farther from the median — so the success
zone is small; pass --salinity/--turbidity to probe a specific pair instead.

Usage:
    python -m debug.phi_field                              # auto tail target, small zone
    python -m debug.phi_field --percentile 2 --seed 2      # rarer target, smaller zone
    python -m debug.phi_field --salinity 41.2 --turbidity 0.29   # explicit target
    python -m debug.phi_field --animate --steps 720 --every 15 --grid 48,48,24
"""
import argparse
import os

import numpy as np

from debug import rollout as R
from debug.landscape import eval_phi_grid, cluster_maxima


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=R.DEFAULT_CHECKPOINT)
    p.add_argument("--seed", type=int, default=0, help="fixes the field (sources / NetCDF window)")
    p.add_argument("--salinity", type=float, default=None,
                   help="target salinity S* (default: auto-pick from the value tail)")
    p.add_argument("--turbidity", type=float, default=None,
                   help="target turbidity τ* in [0,1) (default: auto-pick from the value tail)")
    p.add_argument("--percentile", type=float, default=5.0,
                   help="tail width for auto-target: target lands in the <P%% (or >100-P%%) "
                        "tail of the salinity distribution, whichever is farther from the "
                        "median — giving a small success zone (default 5)")
    p.add_argument("--target-points", type=int, default=4000,
                   help="random field samples used to estimate the value distribution")
    p.add_argument("--synthetic", action="store_true",
                   help="use the synthetic analytical field instead of the run's NetCDF")
    p.add_argument("--grid", default="80,80,40", help="nx,ny,nz grid resolution")
    p.add_argument("--global-tol", type=float, default=1e-3,
                   help="points with Φ ≥ maxΦ − tol count as global optima (default 1e-3)")
    p.add_argument("--cluster-frac", type=float, default=0.04,
                   help="merge optima closer than this fraction of the domain diagonal")
    p.add_argument("--animate", action="store_true",
                   help="also render a GIF/MP4 of how Φ, the success zone and the optima "
                        "vary over an episode (the field is time-dependent)")
    p.add_argument("--steps", type=int, default=720,
                   help="episode length to traverse for --animate (default 720)")
    p.add_argument("--every", type=int, default=20,
                   help="sample/render a frame every N env steps (default 20)")
    p.add_argument("--fps", type=int, default=8, help="animation frame rate")
    p.add_argument("--video-format", choices=["gif", "mp4"], default="gif")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def auto_target(env, salinity_fn, percentile, npts, rng):
    """Pick a rare (S*, τ*) target from the value tail, so the success zone is small.

    Samples salinity at `npts` random points, then chooses whichever tail (lower
    <P% or upper >100-P%) is farther from the median ('either extreme'), and picks
    an actual point from that tail — so the (S*, τ*) pair is guaranteed reachable.
    Returns (S*, τ*, info)."""
    X, Y, Z = env.domain
    P = rng.uniform([0, 0, 0], [X, Y, Z], size=(npts, 3))
    try:
        S = np.asarray(salinity_fn(P[:, 0], P[:, 1], P[:, 2]), dtype=float)
        if S.shape != (npts,):
            raise ValueError
    except Exception:
        S = np.array([float(salinity_fn(*p)) for p in P])
    med = float(np.median(S))
    lo, hi = np.percentile(S, percentile), np.percentile(S, 100 - percentile)
    if (S.max() - med) >= (med - S.min()):
        tail, mask = "upper", S >= hi
    else:
        tail, mask = "lower", S <= lo
    idx = np.flatnonzero(mask)
    j = int(idx[rng.integers(len(idx))])
    pt = P[j]
    return float(S[j]), float(R.compute_turbidity(pt[2])), dict(
        tail=tail, median=med, lo=float(lo), hi=float(hi), point=pt)


def global_optima(Phi, xs, ys, zs, tol, cluster_frac, domain):
    """Clustered points with Φ within `tol` of the max (sorted by Φ desc)."""
    gmax = float(Phi.max())
    mask = Phi >= (gmax - tol)
    ci = np.argwhere(mask)
    coords = np.stack([xs[ci[:, 0]], ys[ci[:, 1]], zs[ci[:, 2]]], axis=1)
    phivals = Phi[mask]
    diag = np.linalg.norm(domain)
    keep = cluster_maxima(coords, phivals, cluster_frac * diag)
    coords = coords[keep]
    return coords[np.argsort(phivals[keep])[::-1]]


def main():
    cli = parse_args()
    nx, ny, nz = (int(v) for v in cli.grid.split(","))
    run = R.load_run(cli.checkpoint, synthetic=cli.synthetic)
    env = R.build_env(run.args)
    env.reset(seed=cli.seed)

    X, Y, Z = env.domain
    xs = np.linspace(0, X, nx)
    ys = np.linspace(0, Y, ny)
    zs = np.linspace(0, Z, nz)
    salinity_fn = R.make_salinity_fn(env)

    # Target: explicit if both given, else auto-pick from the value tail (small zone).
    if cli.salinity is not None and cli.turbidity is not None:
        Sstar, Tstar = cli.salinity, cli.turbidity
    else:
        rng = np.random.default_rng(cli.seed)
        Sstar, Tstar, ti = auto_target(env, salinity_fn, cli.percentile,
                                       cli.target_points, rng)
        print(f"[auto-target] {ti['tail']} {cli.percentile:g}% tail "
              f"(median={ti['median']:.3f}, P{cli.percentile:g}={ti['lo']:.3f}, "
              f"P{100-cli.percentile:g}={ti['hi']:.3f}) → S*={Sstar:.3f}, τ*={Tstar:.3f} "
              f"at depth {ti['point'][2]:.1f} m")

    z_star = float(np.clip(-np.log(max(1e-9, 1.0 - Tstar)) / R.K_TURBIDITY, 0.0, Z))
    src = "synthetic" if cli.synthetic else f"NetCDF: {env.active_netcdf_path}"
    print(f"Evaluating Φ on {nx}×{ny}×{nz} grid ({src})")
    print(f"Target (S*, τ*) = ({Sstar}, {Tstar})   matching depth z* = {z_star:.1f} m")
    Phi, S = eval_phi_grid(salinity_fn, xs, ys, zs, Sstar, Tstar)

    gmax = float(Phi.max())
    gidx = np.unravel_index(np.argmax(Phi), Phi.shape)
    g_pt = np.array([xs[gidx[0]], ys[gidx[1]], zs[gidx[2]]])

    iz = int(np.argmin(np.abs(zs - z_star)))
    Sz = S[:, :, iz]
    pct = {p: np.percentile(Sz, p) for p in (50, 90, 95, 99)}
    print(f"\nField salinity range (whole grid): [{S.min():.3f}, {S.max():.3f}]")
    print(f"Field salinity range @ z*={z_star:.1f} m: [{Sz.min():.3f}, {Sz.max():.3f}]"
          f"  (S*={Sstar})")
    print(f"  percentiles @ z*:  p50={pct[50]:.3f}  p90={pct[90]:.3f}  "
          f"p95={pct[95]:.3f}  p99={pct[99]:.3f}")
    print(f"Max Φ = {gmax:.4f} at (x={g_pt[0]:.0f}, y={g_pt[1]:.0f}, z={g_pt[2]:.1f})")
    if gmax < 0.9:
        print(f"  ⚠ target pair is hard to satisfy in this field (max Φ < 0.9); the "
              f"'optima' below are merely the closest the field gets.")

    # global-optimal points: Φ within global-tol of the max, clustered
    coords = global_optima(Phi, xs, ys, zs, cli.global_tol, cli.cluster_frac, env.domain)
    print(f"\nGlobal-optimal points (Φ ≥ {gmax:.4f}−{cli.global_tol}, clustered): {len(coords)}")
    print(f"{'#':>3} {'x':>7} {'y':>7} {'z':>7}")
    for n, p in enumerate(coords):
        print(f"{n:>3} {p[0]:>7.0f} {p[1]:>7.0f} {p[2]:>7.1f}")

    # success zone: exactly the env's in-zone test, with the training epsilons
    epsS, epsT = env.epsilon_salinity, env.epsilon_turbidity
    tau_z = R.compute_turbidity(zs)                      # (nz,) — depth-only
    zone = (np.abs(S - Sstar) < epsS) & (np.abs(tau_z - Tstar)[None, None, :] < epsT)
    from scipy.ndimage import label
    _, ncomp = label(zone)
    print(f"\nSuccess zone (|ΔS|<{epsS} and |Δτ|<{epsT}, training epsilons): "
          f"{zone.sum()} cells = {100*zone.mean():.2f}% of domain, {ncomp} connected region(s)")
    if not zone.any():
        print("  ⚠ empty success zone for this target at this grid resolution.")

    # 2D mask at the z* plane (τ(z*)=τ*, so only |ΔS|<ε_S remains) — this is what
    # scripts/plot_trajectories.py draws as the gold success zone.
    zone_slice = np.abs(Sz - Sstar) < epsS

    # Φ at z*: with τ(z*)=τ* the depth term is 1, so Φ here is the salinity Gaussian
    # exp(-((S-S*)/σ_s)²) — it PEAKS (=1) at the target, so local optima are visible
    # as peaks. The success threshold Φ_thresh corresponds to |ΔS|=ε_S (Φ above it
    # ⇔ in-zone), evaluated through the same reward_func (no hardcoded σ_s).
    Phiz = Phi[:, :, iz]
    phi_thresh = float(R.reward_func(Sstar + epsS, Tstar, Sstar, Tstar))

    out = R.ensure_out_dir()
    _plot_field(gmax, z_star, Sstar, Tstar, env.domain, cli, out,
                xs, ys, Phiz, zone_slice=zone_slice, eps=(epsS, epsT), phi_thresh=phi_thresh)
    _plot_3d(z_star, Sstar, Tstar, env.domain, out, cli, xs, ys, zone_slice)

    if cli.animate:
        _animate(run, cli, xs, ys, zs, z_star, Sstar, Tstar, epsS, epsT, out)

    if not cli.no_show:
        import matplotlib.pyplot as plt
        plt.show()


def _animate(run, cli, xs, ys, zs, z_star, Sstar, Tstar, epsS, epsT, out):
    """Render how Φ, the success zone and the optima evolve over an episode.

    The NetCDF field is time-dependent: advancing the sim with no-op steps moves
    the loader's interpolation time exactly as a real episode would, so each frame
    is the field at that step's actual time. Animates the three 2D max-projections.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    env = R.build_env(run.args)
    env.reset(seed=cli.seed)
    X, Y, Z = env.domain
    tau_z = R.compute_turbidity(zs)
    salinity_fn = R.make_salinity_fn(env)
    noop = np.full(env.n_agents, 13, dtype=np.int64)
    dt_step = env.dt * env.frame_skip  # seconds per env step

    # precompute frames (advancing time between samples)
    frames = []  # each: dict(step, projections, zone projections, coords)
    step = 0
    next_sample = 0
    while step <= cli.steps:
        if step >= next_sample:
            Phi, S = eval_phi_grid(salinity_fn, xs, ys, zs, Sstar, Tstar)
            zone = (np.abs(S - Sstar) < epsS) & (np.abs(tau_z - Tstar)[None, None, :] < epsT)
            frames.append(dict(
                step=step, gmax=float(Phi.max()),
                xy=Phi.max(axis=2), xz=Phi.max(axis=1), yz=Phi.max(axis=0),
                zxy=zone.any(axis=2), zxz=zone.any(axis=1), zyz=zone.any(axis=0),
                nzone=int(zone.sum())))
            next_sample += cli.every
        if step >= cli.steps:
            break
        env.step(noop)  # advance sim time by one env step (no-op agents)
        step += 1
    print(f"\n[animate] {len(frames)} frames over {cli.steps} steps "
          f"({cli.steps*dt_step:.0f}s); rendering ...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def draw(fr):
        for ax in axes:
            ax.clear()
        axy, axz, ayz = axes
        axy.imshow(fr["xy"].T, origin="lower", extent=[0, X, 0, Y], aspect="equal",
                   cmap="viridis", vmin=0, vmax=1)
        axz.imshow(fr["xz"].T, origin="upper", extent=[0, X, Z, 0], aspect="auto",
                   cmap="viridis", vmin=0, vmax=1)
        ayz.imshow(fr["yz"].T, origin="upper", extent=[0, Y, Z, 0], aspect="auto",
                   cmap="viridis", vmin=0, vmax=1)
        for ax, m, ax_x, ax_y in ((axy, fr["zxy"], xs, ys), (axz, fr["zxz"], xs, zs),
                                  (ayz, fr["zyz"], ys, zs)):
            if m.any() and not m.all():
                ax.contour(ax_x, ax_y, m.T.astype(float), levels=[0.5],
                           colors="red", linewidths=1.6)
        for ax in (axz, ayz):
            ax.axhline(z_star, color="white", ls="--", lw=1, alpha=0.7)
        axy.set_title("x-y (max over z)"); axy.set_xlabel("x [m]"); axy.set_ylabel("y [m]")
        axz.set_title("x-depth (max over y)"); axz.set_xlabel("x [m]"); axz.set_ylabel("depth [m]")
        ayz.set_title("y-depth (max over x)"); ayz.set_xlabel("y [m]"); ayz.set_ylabel("depth [m]")
        t = fr["step"] * dt_step
        fig.suptitle(f"Φ & success zone over time — target (S*={Sstar}, τ*={Tstar}), "
                     f"seed {cli.seed}\nstep {fr['step']}/{cli.steps}  (t={t:.0f}s)  "
                     f"maxΦ={fr['gmax']:.3f}  zone={fr['nzone']} cells  "
                     f"red=success zone", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / cli.fps,
                         save_count=len(frames))
    base = os.path.join(out, f"phi_S{Sstar}_T{Tstar}_seed{cli.seed}")
    if cli.video_format == "mp4":
        try:
            from matplotlib.animation import FFMpegWriter
            path = base + ".mp4"
            anim.save(path, writer=FFMpegWriter(fps=cli.fps))
            print(f"[animate] saved → {path}")
            return
        except Exception as e:
            print(f"[animate] mp4 failed ({e}); falling back to gif")
    path = base + ".gif"
    anim.save(path, writer=PillowWriter(fps=cli.fps))
    print(f"[animate] saved → {path}")


def _plot_field(gmax, z_star, Sstar, Tstar, domain, cli, out,
                xs, ys, Phiz, zone_slice=None, eps=None, phi_thresh=None):
    """Two side-by-side 3D plots at the fixed depth z*:

    Left  — the success zone, a gold scatter at the z* plane (cells where |ΔS|<ε_S,
            with τ(z*)=τ* fixing the depth), exactly as scripts/plot_trajectories.py.
    Right — the Φ landscape at z*: x/y are space, the vertical axis is the reward
            potential Φ(x, y) (= exp(-((S-S*)/σ_s)²) since τ(z*)=τ*), so the TARGET is
            the optimum (Φ=1) and local optima show up as peaks. A translucent plane
            marks the success threshold Φ_thresh (|ΔS|=ε_S; Φ above it ⇔ in-zone) and
            the in-zone cells are highlighted in gold."""
    import matplotlib.pyplot as plt
    X, Y, Z = domain
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    fig = plt.figure(figsize=(16, 7))

    # ---- left: success zone (gold slice at z*) ----
    ax3 = fig.add_subplot(121, projection="3d")
    if zone_slice is not None and zone_slice.any():
        ax3.scatter(Xg[zone_slice], Yg[zone_slice],
                    np.full(int(zone_slice.sum()), z_star),
                    color="gold", alpha=0.18, s=10, depthshade=False,
                    label=(f"success zone (z*, ε={eps[0]},{eps[1]})" if eps
                           else "success zone (z*)"))
    ax3.set_xlim(0, X); ax3.set_ylim(0, Y); ax3.set_zlim(Z, 0)
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("depth [m]")
    ax3.set_title(f"success zone @ z*={z_star:.0f} m")
    ax3.legend(loc="upper left", fontsize=8)

    # ---- right: Φ landscape at z* (vertical axis = Φ, target = optimum) ----
    ax4 = fig.add_subplot(122, projection="3d")
    surf = ax4.plot_surface(Xg, Yg, Phiz, cmap="viridis", vmin=0, vmax=1,
                            linewidth=0, antialiased=True, alpha=0.9)
    # Success threshold: a translucent plane at Φ_thresh (|ΔS|=ε_S); the surface
    # rising above it marks the in-zone basins.
    if phi_thresh is not None:
        ax4.plot_surface(Xg, Yg, np.full_like(Phiz, phi_thresh), color="orange", alpha=0.2)
    # In-zone cells (|ΔS|<ε_S) highlighted on the surface at their Φ height.
    if zone_slice is not None and zone_slice.any():
        ax4.scatter(Xg[zone_slice], Yg[zone_slice], Phiz[zone_slice],
                    color="gold", edgecolor="k", s=18, depthshade=False,
                    label="in-zone cells")
        ax4.legend(loc="upper left", fontsize=8)
    fig.colorbar(surf, ax=ax4, fraction=0.03, pad=0.12, label="Φ")
    ax4.set_zlim(0, 1)
    ax4.set_xlabel("x [m]"); ax4.set_ylabel("y [m]"); ax4.set_zlabel("Φ (potential)")
    thr = f", threshold Φ={phi_thresh:.2f}" if phi_thresh is not None else ""
    ax4.set_title(f"Φ landscape @ z*={z_star:.0f} m  (target S*={Sstar:.3f} → Φ=1{thr})")

    fig.suptitle(
        f"Φ field at fixed depth — target (S*={Sstar}, τ*={Tstar})  —  "
        f"seed {cli.seed}, {'synthetic' if cli.synthetic else 'NetCDF'}, maxΦ={gmax:.3f}",
        fontsize=11)
    fig.tight_layout()
    png = os.path.join(out, f"phi_S{Sstar}_T{Tstar}_seed{cli.seed}.png")
    fig.savefig(png, dpi=150)
    print(f"\nSaved figure → {png}")


def _plot_3d(z_star, Sstar, Tstar, domain, out, cli, xs, ys, zone_slice):
    """Interactive 3D scatter of the success zone (gold slice at z*, as in
    scripts/plot_trajectories.py)."""
    try:
        import plotly.graph_objects as go
        X, Y, Z = domain
        fig = go.Figure()
        if zone_slice is not None and zone_slice.any():
            Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
            fig.add_trace(go.Scatter3d(
                x=Xg[zone_slice], y=Yg[zone_slice],
                z=np.full(int(zone_slice.sum()), -z_star), mode="markers",
                marker=dict(size=3, color="gold", opacity=0.25), name="success zone (z*)"))
        fig.update_layout(title=f"Success zone — target (S*={Sstar}, τ*={Tstar})",
                          scene=dict(xaxis_title="x [m]", yaxis_title="y [m]",
                                     zaxis_title="-depth [m]",
                                     xaxis=dict(range=[0, X]), yaxis=dict(range=[0, Y]),
                                     zaxis=dict(range=[-Z, 0])))
        html = os.path.join(out, f"phi_S{Sstar}_T{Tstar}_seed{cli.seed}_3d.html")
        fig.write_html(html)
        print(f"Saved 3D success zone → {html}")
    except Exception as e:
        print(f"(skipped 3D Plotly scatter: {e})")


if __name__ == "__main__":
    main()

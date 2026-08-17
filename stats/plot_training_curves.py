#!/usr/bin/env python
"""Training curves for a set of runs, as a thesis-ready figure.

Reads `charts/episodic_length`, `charts/episodic_return` and
`charts/episode_success` straight out of the TensorBoard event files under
`runs/` and plots them side by side, one line per policy, with a variability
band. Emits the vector PDF, the smoothed curves as CSV, and a ready-made LaTeX
float, in the same conventions as stats/analyze_sweeps.py -- palette and arm
colours are imported from it, so a policy is the same colour in every figure of
the thesis.

    python stats/plot_training_curves.py                       # the four 2N runs
    python stats/plot_training_curves.py --band ci --window 400
    python stats/plot_training_curves.py \\
        --run ippo=runs/ippo_buoyancy_history_4N_new \\
        --run mappo=runs/mappo_buoyancy_history_4N_new \\
        --out stats/out/training_curves_4N

WHAT THE BAND IS, AND IS NOT
----------------------------
These trainers log the three charts ONCE PER EPISODE, so a run contributes
thousands of samples and the band is computed over a rolling window of episodes:

    --band std   +/- 1 standard deviation of the episodes in the window
                 = how much individual episodes differ from each other
    --band ci    +/- 1.96 * s/sqrt(n) on the window mean
                 = how precisely the mean is pinned down at that point
    --band none  the smoothed mean alone

It is NOT across-seed variance. Each configuration was trained once, so nothing
here speaks to run-to-run reproducibility, and the band must not be read as an
error bar on "how good is this algorithm" -- only on "how variable are this
run's episodes". Reporting seed variance would need the same configuration
retrained under several --seed values.

Bounded metrics are clipped to their own range (success to [0,1], length to
[1, max_steps]) so the band cannot imply impossible values.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from analyze_sweeps import (ARM_COLOR, ARM_LABEL, AXIS, INK, INK_2, MUTED,
                            SURFACE, _fig, _save, _style)

# (tag, panel title, y-axis label, (lo, hi) clip or None)
METRICS = [
    ("charts/episodic_length", "Episode length", "steps", (1.0, None)),
    ("charts/episodic_return", "Episode return", "return", None),
    ("charts/episode_success", "Episode success", "success rate", (0.0, 1.0)),
]

def default_runs(n: int) -> dict[str, Path]:
    """The 2x2 ablation for a swarm size, under the `_new` (post-fix) naming."""
    return {f"{algo}{suffix}": Path(f"runs/{algo}_buoyancy_history_{n}N{tag}_new")
            for algo in ("ippo", "mappo")
            for suffix, tag in (("", ""), ("_comm", "_comm"))}


def load_run(run_dir: Path, tags: list[str]) -> dict[str, pd.DataFrame]:
    """Scalar series for `tags`, concatenated over every event file in the dir.

    A resumed run leaves several event files behind; taking only the first would
    silently truncate the curve, so all of them are read and merged on step.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"no TensorBoard event file under {run_dir}")
    out: dict[str, list] = {t: [] for t in tags}
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags()["scalars"])
        for t in tags:
            if t in have:
                out[t] += [(s.step, s.value) for s in ea.Scalars(t)]
    frames = {}
    for t, rows in out.items():
        if not rows:
            print(f"[warn] {run_dir.name}: tag '{t}' absent")
            continue
        d = pd.DataFrame(rows, columns=["step", "value"]).drop_duplicates("step")
        frames[t] = d.sort_values("step").reset_index(drop=True)
    return frames


def smooth(d: pd.DataFrame, window: int, band: str, clip):
    """Rolling mean and band over a window of EPISODES (not of gradient steps)."""
    minp = max(5, window // 4)
    r = d["value"].rolling(window, min_periods=minp)
    mean, std, cnt = r.mean(), r.std(), r.count()
    x = d["step"].rolling(window, min_periods=minp).mean()
    if band == "ci":
        half = 1.96 * std / np.sqrt(cnt.clip(lower=1))
    elif band == "std":
        half = std
    else:
        half = pd.Series(np.zeros(len(d)))
    lo, hi = mean - half, mean + half
    if clip is not None:
        c_lo, c_hi = clip
        if c_lo is not None:
            lo = lo.clip(lower=c_lo)
            mean = mean.clip(lower=c_lo)
        if c_hi is not None:
            hi = hi.clip(upper=c_hi)
            mean = mean.clip(upper=c_hi)
    ok = mean.notna()
    return x[ok].values, mean[ok].values, lo[ok].values, hi[ok].values


BAND_NOTE = {
    "std": "shaded: $\\pm1$ s.d. of the episodes in the window",
    "ci": "shaded: 95\\% CI of the window mean",
    "none": "no variability band",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=2, metavar="N",
                    help="swarm size whose 2x2 ablation to plot (default 2); ignored if "
                         "--run is given")
    ap.add_argument("--run", action="append", metavar="LABEL=DIR", default=[],
                    help="a run to plot (repeatable), overriding --agents")
    ap.add_argument("--window", type=int, default=200,
                    help="rolling window, in EPISODES (default 200)")
    ap.add_argument("--band", choices=["std", "ci", "none"], default="std",
                    help="what the shaded region shows (default std). Never across-seed "
                         "variance -- these runs have one seed each.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default stats/out/training_curves_N<agents>)")
    ap.add_argument("--assets-dir", type=Path, default=None,
                    help="also copy the PDF and .tex here, e.g. thesis/assets")
    ap.add_argument("--name", type=str, default=None,
                    help="basename for the emitted figure "
                         "(default fig7_training_curves_N<agents>)")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects LABEL=DIR, got '{spec}'")
        k, v = spec.split("=", 1)
        runs[k] = Path(v)
    if not runs:
        runs = default_runs(args.agents)
    if args.out is None:
        args.out = Path(f"stats/out/training_curves_N{args.agents}")
    if args.name is None:
        args.name = f"fig7_training_curves_N{args.agents}"
    missing = [f"{k} -> {v}" for k, v in runs.items() if not v.exists()]
    if missing:
        raise SystemExit("run directory not found:\n  " + "\n  ".join(missing))
    args.out.mkdir(parents=True, exist_ok=True)

    tags = [m[0] for m in METRICS]
    print(f"reading {len(runs)} run(s), window={args.window} episodes, band={args.band}")
    data, rows = {}, []
    for label, d in runs.items():
        data[label] = load_run(d, tags)
        n = {t: len(f) for t, f in data[label].items()}
        print(f"  {label:12} <- {d.name}   episodes: {n.get(tags[0], 0)}")

    fig, axes = _fig(1, len(METRICS), figsize=(4.6 * len(METRICS), 3.7), squeeze=False)
    axes = axes[0]
    for ax, (tag, title, ylab, clip) in zip(axes, METRICS):
        for label in runs:
            d = data[label].get(tag)
            if d is None or d.empty:
                continue
            color = ARM_COLOR.get(label, MUTED)
            x, m, lo, hi = smooth(d, args.window, args.band, clip)
            xm = x / 1e6
            if args.band != "none":
                ax.fill_between(xm, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(xm, m, color=color, linewidth=1.9, zorder=3,
                    label=ARM_LABEL.get(label, label) if ax is axes[0] else None)
            for xi, mi, li, hi_ in zip(x, m, lo, hi):
                rows.append(dict(run=label, metric=tag.split("/")[1], step=xi,
                                 mean=mi, lo=li, hi=hi_))
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", fontweight="semibold")
        ax.set_xlabel("environment steps (millions)", fontsize=9, color=INK_2)
        ax.set_ylabel(ylab, fontsize=9, color=INK_2)
        if clip == (0.0, 1.0):
            ax.set_ylim(0, 1)
        _style(ax, grid_axis="both")
    fig.legend(loc="lower center", ncol=len(runs), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pdf = _save(fig, args.out, args.name)

    pd.DataFrame(rows).to_csv(args.out / f"{args.name}.csv", index=False)

    n_agents = args.agents if not args.run else len(runs)
    tex = args.out / f"{args.name}.tex"
    tex.write_text(
        "% generated by stats/plot_training_curves.py -- do not edit by hand\n"
        "\\begin{figure}[t]\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{assets/{pdf.name}}}\n"
        f"\\caption[Training curves ($N = {n_agents}$)]{{Training curves for the four "
        f"policies at $N = {n_agents}$. "
        f"Curves are a rolling mean over {args.window} episodes; "
        f"{BAND_NOTE[args.band]}. Because the three charts are logged once per episode, "
        "the band measures how much individual episodes differ from one another at that "
        "point in training; each configuration was trained with a single seed, so it is "
        "\\emph{not} an across-seed error bar and says nothing about run-to-run "
        f"reproducibility.\\label{{fig:{args.name}}}}}\n\\end{{figure}}\n")

    made = [pdf, tex]
    if args.assets_dir:
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for p in made:
            shutil.copy2(p, args.assets_dir / p.name)
        print(f"copied {len(made)} artefact(s) -> {args.assets_dir}")
    print(f"\nwrote -> {args.out}")
    for p in made + [args.out / f'{args.name}.csv']:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

"""
Head-to-head comparison of ANY NUMBER of policies on the Oceananigans swarm
task, all deployed with the SAME number of agents (default N=2):

  * each --policy arm is a trained PPO (single-agent) or MAPPO/IPPO (multi-agent)
    checkpoint. Single- and multi-agent checkpoints are AUTO-DETECTED and
    deployed identically as N agents in an N-agent env — a PPO policy runs as N
    INDEPENDENT copies (batched forward, no shared obs, no coordination); a
    MAPPO/IPPO policy runs its jointly-trained shared policy. So you can line up,
    in ONE run, e.g. ppo vs mappo vs mappo+comm without re-rolling ppo each time.
  * random — N agents choosing uniformly-random actions (the luck floor), added
    by default (drop with --no-random).

The thesis claim is "a jointly-trained MARL policy with N agents beats N
independent single-agent RL policies", judged on success_any (the metric the
committee cares about). This script makes that comparison HONEST:

  * every arm is evaluated on the SAME episode seeds, so each seed is the SAME
    frozen field + target + spawn for all arms -> the episodes are PAIRED;
  * the primary metric (episode success_any, first-reach lens) is reported with
    a Wilson 95% CI per arm AND a paired McNemar exact test for EVERY pair of
    arms, so "MAPPO > single-agent" is a statistical statement, not an eyeball;
  * success_all / per-agent / SPL / success@T are reported too (use
    --no-end-on-any-success for the swarm-truth lens where success_all is
    meaningful — under the default first-reach lens the episode ends at the
    first arrival, so success_all is censored).

The FIRST policy supplies the common task config (field, domain, target mode,
epsilon, ...); every other arm is FORCED onto it, so a seed reproduces the same
instance in every arm. The obs-affecting params (k, communication,
dead_reckoning, sigma_*) stay per-arm — each policy sees exactly the observation
layout it was trained on.

Outputs (under --out-dir, default stats/out/compare__<labels>__N<N>_<mode>/):
  * summary.json          — every metric + the paired McNemar tests
  * per_episode.csv       — one row per agent-episode, arm in the 'mode' column
  * paired_episodes.csv    — one row per seed: success_any/all for all arms
  * figures/success_at_budget.png  — success vs step budget (any solid, all dashed)
  * figures/approach_curve.png     — mean distance-to-zone vs normalized time
  * figures/success_bars.png       — success_any / success_all / per-agent / SPL

Usage:
    # new multi-policy form (label optional; derived from the run name if omitted)
    python stats/compare_single_vs_multi.py \
        --policy ppo=runs/ppo_buoyancy_history/checkpoints/latest.pt \
        --policy mappo=runs/mappo_buoyancy_history_success_all/checkpoints/iter_0720.pt \
        --policy mappo_comm=runs/mappo_comms_success_all/checkpoints/iter_0720.pt \
        --netcdf-file data/oceananigans/buoyancy_active/test \
        --episodes 200 --seed 0 --no-end-on-any-success

    # legacy two-arm form still works
    python stats/compare_single_vs_multi.py \
        --sa-checkpoint runs/ppo_buoyancy_history/checkpoints/latest.pt \
        --ma-checkpoint runs/mappo_buoyancy_history/checkpoints/latest.pt \
        --netcdf-file data/oceananigans/buoyancy_active/test --episodes 200 --seed 0
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import eval_oceananigans_stats as ev  # reuse env/policy/rollout/metric machinery

BUDGETS = (100, 250, 500, 700, 1000, 1500, 2000, 3600)

# Task-defining parameters copied from the FIRST policy's checkpoint onto every
# other arm so a seed reproduces the SAME field/target/spawn everywhere. These
# affect the episode INSTANCE, not the observation layout — the obs-affecting
# params (k, communication, dead_reckoning, sigma_*) stay per-arm.
TASK_KEYS = ("netcdf_file", "domain", "target_mode", "target_percentile",
             "static_frame", "epsilon_salinity", "epsilon_turbidity",
             "v_agent", "dt", "frame_skip", "max_steps", "n_agents",
             "end_on_any_success", "eval_success_steps",
             "min_spawn_distance", "spawn_max_tries")

# Keys whose mismatch vs the task-source policy is worth surfacing (a policy
# optimized on a different task is still evaluated, but the warning flags it).
WARN_KEYS = ("netcdf_file", "target_mode", "epsilon_salinity", "domain",
             "target_percentile", "static_frame")

# Qualitative palette for the policy arms (random is always gray).
_PALETTE = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange",
            "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", default=[], metavar="[LABEL=]CKPT",
                   help="a policy arm as LABEL=path/to/checkpoint.pt (repeatable). "
                        "LABEL is optional (derived from the run name if omitted). "
                        "PPO / MAPPO / IPPO checkpoints are auto-detected and all "
                        "deployed as N agents. The FIRST policy supplies the common "
                        "task config.")
    # legacy two-arm aliases (kept for backward compatibility)
    p.add_argument("--sa-checkpoint", default=None,
                   help="(legacy) shorthand for --policy single-agent=CKPT")
    p.add_argument("--ma-checkpoint", default=None,
                   help="(legacy) shorthand for --policy multi-agent=CKPT")
    p.add_argument("--no-random", dest="include_random", action="store_false",
                   default=True, help="omit the random baseline arm (on by default)")
    p.add_argument("--n-agents", type=int, default=2,
                   help="agents per arm (default 2). Must equal a policy's trained "
                        "n_agents if that policy uses communication (its neighbor-block "
                        "observation is N-specific); free otherwise.")
    p.add_argument("--netcdf-file", type=str, default=None,
                   help="field spec for ALL arms — a folder (globs *.nc), a single "
                        ".nc, or a glob. Default: the first policy's field. Use a "
                        "held-out split, e.g. data/oceananigans/buoyancy_active/test.")
    p.add_argument("--episodes", type=int, default=100,
                   help="paired episodes (same seeds across arms; default 100)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; episode i uses seed+i (default: random base)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override episode length for all arms (default: first policy's)")
    p.add_argument("--target-mode", type=str, default=None,
                   choices=["random", "tail"],
                   help="override the target sampling for ALL arms (default: the first "
                        "policy's). 'tail' draws S* from a rare salinity tail on the "
                        "target's depth plane (LOW/HIGH 50/50), shrinking the success "
                        "zone so success needs real navigation — the harder, "
                        "meeting-scenario regime. Applied identically to every arm, so "
                        "the episodes stay paired.")
    p.add_argument("--target-percentile", type=float, default=None,
                   help="tail width in percent (tail mode only; default: first "
                        "policy's, typically 5.0)")
    p.add_argument("--success-steps", type=int, default=1,
                   help="consecutive in-zone steps counted as success (default 1)")
    p.add_argument("--min-spawn-distance", type=float, default=0.0,
                   help="distant-start difficulty knob: reject-sample spawns until "
                        "every agent starts at least this many METRES from the nearest "
                        "success-zone cell, applied identically to all arms (paired). "
                        "0 (default) = original uniform spawn. Most meaningful with "
                        "--target-mode tail, where the zone is small and localized.")
    p.add_argument("--spawn-max-tries", type=int, default=200,
                   help="rejection budget per agent for --min-spawn-distance (default "
                        "200); the farthest candidate is used if none clears it.")
    p.add_argument("--mode", default="greedy", choices=["greedy", "stochastic"],
                   help="decode for the trained policies (default greedy = deployment)")
    p.add_argument("--end-on-any-success", dest="end_on_any_success",
                   default=True, action="store_true",
                   help="first-reach lens (episode ends at the first arrival); this is "
                        "the deployment / success_any regime and the DEFAULT.")
    p.add_argument("--no-end-on-any-success", dest="end_on_any_success",
                   action="store_false",
                   help="swarm-truth lens: episode runs until EVERY agent arrives, so "
                        "success_all and per-agent success are de-censored.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output dir (default: stats/out/compare__<labels>__N<N>_<mode>/)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def collect_policy_specs(cli):
    """Ordered list of (label_or_None, checkpoint_path) from the legacy aliases
    (first) followed by every --policy entry. The first spec is the task source."""
    specs = []
    if cli.sa_checkpoint:
        specs.append(("single-agent", cli.sa_checkpoint))
    if cli.ma_checkpoint:
        specs.append(("multi-agent", cli.ma_checkpoint))
    for item in cli.policy:
        if "=" in item:
            label, path = item.split("=", 1)
            specs.append((label.strip() or None, path.strip()))
        else:
            specs.append((None, item.strip()))
    if not specs:
        raise SystemExit(
            "no policies given. Pass at least one --policy LABEL=checkpoint.pt "
            "(or the legacy --sa-checkpoint / --ma-checkpoint).")
    return specs


def derive_label(ckpt, path, kind):
    """A short arm label: the run's exp_name if available, else the run dir name."""
    rn = ckpt.get("run_name")
    if rn and "__" in rn:
        parts = rn.split("__")
        if len(parts) >= 2 and parts[1]:
            return parts[1]                       # exp_name slot of the run name
    # fallback: the directory two levels above the checkpoint (the run dir)
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return run_dir or kind


def uniquify(label, existing):
    """Disambiguate a duplicate label with a #2, #3, ... suffix."""
    if label not in existing:
        return label
    i = 2
    while f"{label}#{i}" in existing:
        i += 1
    return f"{label}#{i}"


def build_colors(labels):
    colors, k = {}, 0
    for l in labels:
        if l == "random":
            colors[l] = "tab:gray"
        else:
            colors[l] = _PALETTE[k % len(_PALETTE)]
            k += 1
    return colors


# --------------------------------------------------------------------------- #
def mcnemar_exact(a_wins, b_wins):
    """Two-sided exact McNemar test on paired binary outcomes. a_wins = #seeds
    where A succeeded and B failed; b_wins = the reverse. Returns (p, statement)."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def load_arm(ckpt_path, task_args, device, mode):
    """Build the env (task params forced from `task_args`) + the action selector
    for one trained-policy arm. Returns (ckpt, env, selector, kind, obs_dim)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    if "netcdf_file" not in a:
        raise SystemExit(f"'{ckpt_path}' is not an Oceananigans checkpoint.")
    args = SimpleNamespace(**a)
    # obs-affecting params (k, communication, dead_reckoning, sigma_*, reward_potential)
    # stay as trained; only the task instance params are forced to be common.
    for key in TASK_KEYS:
        setattr(args, key, getattr(task_args, key))

    env = ev.build_env(args)
    try:
        policy, _ = ev.load_policy(ckpt, env, device)
    except SystemExit as e:
        raise SystemExit(
            f"{os.path.basename(ckpt_path)}: {e}\n"
            f"  (a communication policy is locked to its trained n_agents; pass "
            f"--n-agents equal to it, or compare a no-communication checkpoint.)")
    obs_mean, obs_var = ev.combine_obs_rms(ckpt["obs_rms"])

    def normalize(raw):
        norm = (raw - obs_mean) / np.sqrt(obs_var + 1e-8)
        return np.clip(norm, -10.0, 10.0).astype(np.float32)

    selector = ev.make_checkpoint_selector(policy, normalize, greedy=(mode == "greedy"))
    kind = "mappo" if isinstance(policy, ev.MappoPolicy) else (
        "ippo" if isinstance(policy, ev.IppoPolicy) else "ppo")
    obs_dim = int(np.prod(env.observation_space.shape))
    return ckpt, env, selector, kind, obs_dim


def run_arm(env, selector, base_seed, episodes, max_steps, multi_agent, dt_s):
    """Roll out `episodes` paired episodes; return (per_agent, traces_all,
    episode_success) exactly in the shape ev.aggregate expects."""
    per_agent, traces_all, episode_success = [], [], []
    for ep in range(episodes):
        traces, meta = ev.rollout_episode(
            env, selector, base_seed + ep, max_steps, multi_agent)
        ep_succ, ep_ts = [], []
        for i, tr in enumerate(traces):
            m = ev.agent_metrics(tr, meta, dt_s)
            m["_episode"], m["_agent"] = ep, i
            per_agent.append(m)
            traces_all.append(tr)
            ep_succ.append(m["success"])
            if m["success"]:
                ep_ts.append(m["steps_to_success"])
        episode_success.append(dict(
            any=any(ep_succ), all=all(ep_succ),
            t_any=(min(ep_ts) if ep_ts else None),
            t_all=(max(ep_ts) if (ep_ts and all(ep_succ)) else None)))
    return per_agent, traces_all, episode_success


# --------------------------------------------------------------------------- #
def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _stat(agg, key, field="median", pct=False):
    s = agg[key]
    v = s[field] if isinstance(s, dict) else s
    return _fmt(v, pct)


def print_table(labels, results, ep_stats, header):
    col_w = max(18, max(len(l) for l in labels) + 2)
    lab_w = 30
    total = lab_w + col_w * len(labels)
    print("\n" + "=" * total)
    print(header)
    print("=" * total)
    rowfmt = "{:<%d}" % lab_w + ("{:>%d}" % col_w) * len(labels)
    print(rowfmt.format("metric", *labels))
    print("-" * total)

    def row(name, fn):
        print(rowfmt.format(name, *[fn(l) for l in labels]))

    # primary: episode-level success_any with Wilson CI (the committee metric)
    row("success_any (episode)", lambda l: (
        f"{ep_stats[l]['any_rate'] * 100:.1f}% "
        f"[{ep_stats[l]['any_ci'][0] * 100:.0f},{ep_stats[l]['any_ci'][1] * 100:.0f}]"))
    row("success_all (episode)", lambda l: _fmt(ep_stats[l]["all_rate"], pct=True))
    row("per-agent success", lambda l: (
        f"{results[l]['success_rate'] * 100:.1f}% "
        f"[{results[l]['success_ci95'][0] * 100:.0f},{results[l]['success_ci95'][1] * 100:.0f}]"))
    row("SPL", lambda l: _fmt(results[l]["spl_mean"]))
    row("success_any@700", lambda l: _fmt(results[l]["success_at_budget"]["700"], pct=True))
    row("success_any@1500", lambda l: _fmt(results[l]["success_at_budget"]["1500"], pct=True))
    row("success_all@700", lambda l: _fmt(results[l]["success_all_at_budget"]["700"], pct=True))
    row("success_all@1500", lambda l: _fmt(results[l]["success_all_at_budget"]["1500"], pct=True))
    row("steps to success (med)", lambda l: _stat(results[l], "steps_to_success"))
    row("path eff. success (med)", lambda l: _stat(results[l], "path_efficiency_success"))
    row("closer-each-step (med)", lambda l: _stat(results[l], "monotonic_frac"))
    row("min dist | fail (med,m)", lambda l: _stat(results[l], "min_dist_failures"))
    row("final |dS| all (med)", lambda l: _stat(results[l], "final_dS_all"))
    row("no-op frac (med)", lambda l: _stat(results[l], "noop_frac"))
    row("z ping-pong (med)", lambda l: _stat(results[l], "zflip_frac"))
    print("=" * total)


def pairwise_mcnemar(labels, flags_by_label):
    """Exact McNemar for every unordered pair; returns list of dicts.

    `flags_by_label` maps label -> per-episode 0/1 outcome, so the same routine
    serves success_any and success_all."""
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            xa = np.asarray(flags_by_label[a], bool)
            xb = np.asarray(flags_by_label[b], bool)
            a_wins = int(np.sum(xa & ~xb))   # a succeeded, b failed
            b_wins = int(np.sum(xb & ~xa))
            p = mcnemar_exact(a_wins, b_wins)
            lead = a if a_wins > b_wins else (b if b_wins > a_wins else "tie")
            out.append(dict(a=a, b=b, a_only=a_wins, b_only=b_wins,
                            p_value=p, lead=lead))
    return out


def print_mcnemar(labels, flags_by_label, metric="success_any"):
    tests = pairwise_mcnemar(labels, flags_by_label)
    w = max(12, max(len(l) for l in labels))
    print(f"\nPaired McNemar exact test on episode {metric} (same seeds):")
    print("-" * (2 * w + 52))
    for t in tests:
        sig = "**" if t["p_value"] < 0.05 else ("*" if t["p_value"] < 0.10 else "ns")
        print(f"  {t['a']:>{w}} vs {t['b']:<{w}}  {t['a']}-only={t['a_only']:3d}  "
              f"{t['b']}-only={t['b_only']:3d}  p={t['p_value']:.3f} {sig}  "
              f"(lead: {t['lead']})")
    print("-" * (2 * w + 52))
    print("  ** p<0.05   * p<0.10   ns not significant")
    return tests


# --------------------------------------------------------------------------- #
def make_plots(fig_dir, labels, colors, results, curves, any_ts, all_ts, ep_stats,
               max_steps, n_agents, episodes):
    os.makedirs(fig_dir, exist_ok=True)

    # 1) success vs step budget: any (solid) + all (dashed), one color per arm.
    plt.figure(figsize=(7.5, 5.5))

    def cdf(ts, color, style, label):
        n = len(ts)
        hit = np.sort([t for t in ts if t is not None])
        if n and hit.size:
            x = np.concatenate([[0], hit, [max_steps]])
            y = np.concatenate([[0], np.arange(1, hit.size + 1), [hit.size]]) / n
            plt.step(x, y, where="post", lw=2, color=color, linestyle=style, label=label)

    for l in labels:
        c = colors[l]
        cdf(any_ts[l], c, "-", f"{l} (any)")
        cdf(all_ts[l], c, "--", f"{l} (all)")
    plt.xlabel("step budget T"); plt.ylabel("episode success within T")
    plt.title(f"Success vs step budget  (N={n_agents}, {episodes} eps)")
    plt.xlim(0, max_steps); plt.ylim(0, 1.02)
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "success_at_budget.png"), dpi=150)
    plt.close()

    # 2) mean approach to zone vs normalized episode time.
    plt.figure(figsize=(7, 5))
    for l in labels:
        grid, curve = curves[l]
        if curve is not None:
            plt.plot(grid, curve, lw=2, color=colors[l], label=l)
    plt.xlabel("normalized episode time"); plt.ylabel("distance to zone / spawn distance")
    plt.title(f"Mean approach to target zone  (N={n_agents})"); plt.ylim(0, 1.05)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "approach_curve.png"), dpi=150)
    plt.close()

    # 3) grouped bars: success_any / success_all / per-agent / SPL.
    metrics = ["success_any", "success_all", "per-agent", "SPL"]
    vals = {l: [ep_stats[l]["any_rate"], ep_stats[l]["all_rate"],
                results[l]["success_rate"], results[l]["spl_mean"]] for l in labels}
    x = np.arange(len(metrics))
    w = 0.8 / len(labels)
    plt.figure(figsize=(max(8.5, 2.0 * len(labels)), 5))
    for j, l in enumerate(labels):
        xs = x + (j - (len(labels) - 1) / 2) * w
        plt.bar(xs, vals[l], width=w, color=colors[l], label=l)
        for xi, v in zip(xs, vals[l]):
            if v is not None:
                plt.text(xi, v + 0.01, f"{v * 100:.0f}" if v <= 1 else f"{v:.2f}",
                         ha="center", va="bottom", fontsize=7)
    plt.xticks(x, metrics); plt.ylim(0, 1.05)
    plt.ylabel("rate  /  SPL"); plt.title(f"Outcome comparison  (N={n_agents}, {episodes} eps)")
    plt.grid(alpha=0.3, axis="y"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "success_bars.png"), dpi=150)
    plt.close()


def write_paired_csv(path, labels, any_by_label, all_by_label, episodes):
    with open(path, "w", newline="") as f:
        cols = ["episode"] + [f"{l}_any" for l in labels] + [f"{l}_all" for l in labels]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ep in range(episodes):
            row = {"episode": ep}
            for l in labels:
                row[f"{l}_any"] = int(any_by_label[l][ep])
                row[f"{l}_all"] = int(all_by_label[l][ep])
            w.writerow(row)


# --------------------------------------------------------------------------- #
def main():
    cli = parse_args()
    device = torch.device("cpu")
    N = cli.n_agents
    specs = collect_policy_specs(cli)

    # Resolve the common task config from the FIRST policy + CLI overrides. Every
    # arm (trained + random) is forced onto these, so a seed is the SAME episode
    # instance in every arm (paired comparison).
    src_raw = torch.load(specs[0][1], map_location=device, weights_only=False)
    if "netcdf_file" not in src_raw["args"]:
        raise SystemExit(f"'{specs[0][1]}' is not an Oceananigans checkpoint.")
    task = SimpleNamespace(**src_raw["args"])
    task.n_agents = N
    task.netcdf_file = cli.netcdf_file or task.netcdf_file
    task.max_steps = cli.max_steps or task.max_steps
    task.end_on_any_success = cli.end_on_any_success
    task.eval_success_steps = cli.success_steps
    task.min_spawn_distance = cli.min_spawn_distance
    task.spawn_max_tries = cli.spawn_max_tries
    if cli.target_mode is not None:
        task.target_mode = cli.target_mode
    if cli.target_percentile is not None:
        task.target_percentile = cli.target_percentile

    # Build every policy arm. Random (if requested) reuses the first arm's env
    # (identical task params -> paired instances; it ignores the observation).
    labels, arms, kinds, dims, run_names, paths = [], {}, {}, {}, {}, {}
    for label, path in specs:
        ckpt, env, selector, kind, dim = load_arm(path, task, device, cli.mode)
        # communication policies are locked to their trained n_agents (their
        # neighbor-block observation is N-specific).
        if ckpt["args"].get("communication", False) and \
                N != int(ckpt["args"].get("n_agents", 1)):
            raise SystemExit(
                f"'{path}' was trained with communication at n_agents="
                f"{ckpt['args'].get('n_agents')}, whose neighbor-block observation is "
                f"N-specific — rerun with --n-agents {ckpt['args'].get('n_agents')}.")
        if label is None:
            label = derive_label(ckpt, path, kind)
        label = uniquify(label, labels)
        # surface task mismatches vs the source policy (still evaluated, forced
        # onto the common task, but the policy was optimized on a different one).
        for key in WARN_KEYS:
            sv, mv = ckpt["args"].get(key), src_raw["args"].get(key)
            if sv is not None and mv is not None and \
                    tuple(np.atleast_1d(sv)) != tuple(np.atleast_1d(mv)):
                print(f"  WARNING: [{label}] trained with {key}={sv} but task source "
                      f"used {key}={mv} — forcing {mv} (episodes still paired, but "
                      f"this policy was optimized on a different task).")
        labels.append(label)
        arms[label] = (env, selector)
        kinds[label] = kind
        dims[label] = dim
        paths[label] = path
        run_names[label] = ckpt.get("run_name") or os.path.basename(
            os.path.dirname(os.path.dirname(path)))

    if cli.include_random:
        sel_random = ev.make_random_selector(cli.seed if cli.seed is not None else 0)
        arms["random"] = (arms[labels[0]][0], sel_random)  # reuse first env -> paired
        labels.append("random")
        kinds["random"] = "random"
        dims["random"] = dims[labels[0]]

    colors = build_colors(labels)
    dt_s = task.dt * task.frame_skip
    base_seed = cli.seed if cli.seed is not None else int(
        np.random.SeedSequence().entropy % (2 ** 31))
    multi_agent = N > 1

    pol_labels = [l for l in labels if l != "random"]
    tag = "compare__" + "__".join(l[:20] for l in pol_labels) + \
          f"__N{N}__{task.target_mode}"
    if cli.min_spawn_distance > 0.0:
        tag += f"__d{int(cli.min_spawn_distance)}"
    out_dir = cli.out_dir or os.path.join(ROOT, "stats", "out", tag)
    os.makedirs(out_dir, exist_ok=True)

    lens = "first-reach (success_any)" if cli.end_on_any_success else "all-success (swarm-truth)"
    print(f"Policies ({len(pol_labels)}):")
    for l in pol_labels:
        print(f"  [{l:>16}] {kinds[l]:>5}  obs {dims[l]:>3}  <- {run_names[l]}")
    print(f"Random arm   : {'yes' if cli.include_random else 'no'}")
    print(f"Agents/arm   : {N}   Episodes: {cli.episodes} paired, base seed {base_seed}")
    tgt = f"{task.target_mode}" + (f" (±{task.target_percentile}%)"
                                   if task.target_mode == "tail" else "")
    print(f"Field        : {task.netcdf_file}   max_steps {task.max_steps}   "
          f"success_steps {cli.success_steps}")
    print(f"Target       : {tgt}")
    if cli.min_spawn_distance > 0.0:
        print(f"Spawn        : >= {cli.min_spawn_distance:.0f} m from zone "
              f"(distant-start, {cli.spawn_max_tries} tries/agent)")
    print(f"Lens         : {lens}   decode: {cli.mode}")
    print(f"Out          : {out_dir}")

    results, curves = {}, {}
    per_agent_by_label, any_by_label, all_by_label, ep_stats = {}, {}, {}, {}
    any_ts, all_ts = {}, {}
    for l in labels:
        env, selector = arms[l]
        per_agent, traces_all, episode_success = run_arm(
            env, selector, base_seed, cli.episodes, task.max_steps, multi_agent, dt_s)
        agg = ev.aggregate(per_agent, episode_success)
        results[l] = agg
        curves[l] = ev.approach_curve(traces_all)
        per_agent_by_label[l] = per_agent
        any_by_label[l] = [bool(e["any"]) for e in episode_success]
        all_by_label[l] = [bool(e["all"]) for e in episode_success]
        any_ts[l] = [e["t_any"] for e in episode_success]
        all_ts[l] = [e["t_all"] for e in episode_success]
        k_any = int(sum(any_by_label[l]))
        k_all = int(sum(all_by_label[l]))
        n_ep = cli.episodes
        ep_stats[l] = dict(
            any_rate=k_any / n_ep, any_ci=list(ev.wilson_ci(k_any, n_ep)),
            all_rate=k_all / n_ep, all_ci=list(ev.wilson_ci(k_all, n_ep)), n=n_ep)
        print(f"  [{l:>16}] success_any {ep_stats[l]['any_rate']*100:5.1f}%  "
              f"success_all {ep_stats[l]['all_rate']*100:5.1f}%  "
              f"per-agent {agg['success_rate']*100:5.1f}%  SPL {agg['spl_mean']:.3f}")

    header = (f"{' vs '.join(labels)}   N={N}   "
              f"{cli.episodes} paired episodes   [{lens}]")
    print_table(labels, results, ep_stats, header)
    tests = print_mcnemar(labels, any_by_label, "success_any")
    # success_all is the swarm-truth lens and the one where cooperation (comms,
    # CTDE) can actually pay: success_any over N no-comms agents is best-of-N
    # independent draws, so it cannot reward coordination by construction.
    if cli.end_on_any_success:
        print("\n  [success_all McNemar skipped: --end-on-any-success stops the "
              "episode at first arrival, so success_all is censored. "
              "Re-run with --no-end-on-any-success.]")
        tests_all = []
    else:
        tests_all = print_mcnemar(labels, all_by_label, "success_all")

    make_plots(os.path.join(out_dir, "figures"), labels, colors, results, curves,
               any_ts, all_ts, ep_stats, task.max_steps, N, cli.episodes)

    def _mcnemar_block(ts):
        return {f"{t['a']}_vs_{t['b']}": dict(
            a_only=t["a_only"], b_only=t["b_only"],
            p_value=t["p_value"], lead=t["lead"]) for t in ts}

    mcnemar = _mcnemar_block(tests)
    mcnemar_all = _mcnemar_block(tests_all)

    summary = dict(
        policies={l: dict(checkpoint=paths[l],
                          run_name=run_names[l], kind=kinds[l], obs_dim=dims[l])
                  for l in pol_labels},
        labels=labels, include_random=cli.include_random,
        task_source=labels[0], n_agents=N,
        episodes=cli.episodes, base_seed=base_seed, max_steps=task.max_steps,
        success_steps=cli.success_steps, netcdf_file=task.netcdf_file,
        target_mode=task.target_mode, target_percentile=task.target_percentile,
        min_spawn_distance=cli.min_spawn_distance,
        end_on_any_success=cli.end_on_any_success, decode=cli.mode,
        generated=datetime.now().isoformat(timespec="seconds"),
        episode_stats=ep_stats, results=results, mcnemar_success_any=mcnemar,
        mcnemar_success_all=mcnemar_all,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ev.write_csv(os.path.join(out_dir, "per_episode.csv"), per_agent_by_label)
    write_paired_csv(os.path.join(out_dir, "paired_episodes.csv"),
                     labels, any_by_label, all_by_label, cli.episodes)

    print(f"\nWrote summary.json, per_episode.csv, paired_episodes.csv, figures/ -> {out_dir}")
    for env in {id(arms[l][0]): arms[l][0] for l in labels}.values():
        env.close()


if __name__ == "__main__":
    main()

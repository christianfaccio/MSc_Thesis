"""
Head-to-head comparison of THREE policies on the Oceananigans swarm task, all
deployed with the SAME number of agents (default N=2):

  1. single-agent  — one PPO policy trained with n_agents=1, deployed as N
                     INDEPENDENT copies (batched forward, no shared observation,
                     no coordination). This is the "N independent RL agents"
                     baseline the thesis must beat.
  2. multi-agent   — the MAPPO/IPPO shared policy trained jointly at N agents
                     (centralized critic in training; decentralized at exec).
  3. random        — N agents choosing uniformly-random actions (the luck floor).

The thesis claim is "a jointly-trained MARL policy with N agents beats N
independent single-agent RL policies", judged on success_any (the metric the
committee cares about). This script makes that comparison HONEST:

  * every arm is evaluated on the SAME episode seeds, so each seed is the SAME
    frozen field + target + spawn for all three arms -> the episodes are PAIRED;
  * the primary metric (episode success_any, first-reach lens) is reported with
    a Wilson 95% CI per arm AND a paired McNemar exact test between arms, so
    "MAPPO > single-agent" is a statistical statement, not an eyeball;
  * success_all / per-agent / SPL / success@T are reported too (use
    --no-end-on-any-success for the swarm-truth lens where success_all is
    meaningful — under the default first-reach lens the episode ends at the
    first arrival, so success_all is censored).

The single-agent policy runs in an N-agent env built WITHOUT the communication
neighbor block, so its per-agent observation is byte-identical to what it was
trained on (obs dims are N-independent when communication=False). The MAPPO
policy runs in its own N-agent env. Both envs are forced to identical
task-defining parameters (field, domain, target mode, epsilon, ...), so a seed
reproduces the same instance in every arm.

Outputs (under --out-dir, default stats/out/compare__<sa>__vs__<ma>/):
  * summary.json          — every metric + the paired McNemar tests
  * per_episode.csv       — one row per agent-episode, arm in the 'mode' column
  * paired_episodes.csv   — one row per seed: success_any/all for all three arms
  * figures/success_at_budget.png  — success vs step budget (any solid, all dashed)
  * figures/approach_curve.png     — mean distance-to-zone vs normalized time
  * figures/success_bars.png       — success_any / success_all / per-agent / SPL

Usage:
    python stats/compare_single_vs_multi.py \
        --sa-checkpoint runs/ppo_buoyancy_history/checkpoints/latest.pt \
        --ma-checkpoint runs/mappo_buoyancy_history/checkpoints/latest.pt \
        --netcdf-file data/oceananigans/buoyancy_active/test \
        --episodes 200 --seed 0
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

# Task-defining parameters copied from the multi-agent checkpoint onto the
# single-agent env so a seed reproduces the SAME field/target/spawn in every
# arm. These affect the episode INSTANCE, not the observation layout — the
# obs-affecting params (k, communication, dead_reckoning, sigma_*) stay per-arm.
TASK_KEYS = ("netcdf_file", "domain", "target_mode", "target_percentile",
             "static_frame", "epsilon_salinity", "epsilon_turbidity",
             "v_agent", "dt", "frame_skip", "max_steps", "n_agents",
             "end_on_any_success", "eval_success_steps")

COLORS = {"single-agent": "tab:blue", "multi-agent": "tab:red",
          "random": "tab:gray"}


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sa-checkpoint", required=True,
                   help="single-agent PPO checkpoint (.pt), deployed as N copies")
    p.add_argument("--ma-checkpoint", required=True,
                   help="multi-agent MAPPO/IPPO checkpoint (.pt)")
    p.add_argument("--n-agents", type=int, default=2,
                   help="agents per arm (default 2). Must equal the MA checkpoint's "
                        "trained n_agents if it uses communication (its neighbor-block "
                        "observation is N-specific); free otherwise.")
    p.add_argument("--netcdf-file", type=str, default=None,
                   help="field spec for ALL arms — a folder (globs *.nc), a single "
                        ".nc, or a glob. Default: the MA checkpoint's field. Use a "
                        "held-out split, e.g. data/oceananigans/buoyancy_active/test.")
    p.add_argument("--episodes", type=int, default=100,
                   help="paired episodes (same seeds across arms; default 100)")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; episode i uses seed+i (default: random base)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override episode length for all arms (default: MA checkpoint's)")
    p.add_argument("--target-mode", type=str, default=None,
                   choices=["random", "tail"],
                   help="override the target sampling for ALL arms (default: the MA "
                        "checkpoint's). 'tail' draws S* from a rare salinity tail on the "
                        "target's depth plane (LOW/HIGH 50/50), shrinking the success "
                        "zone so success needs real navigation — the harder, "
                        "meeting-scenario regime. Applied identically to every arm, so "
                        "the episodes stay paired.")
    p.add_argument("--target-percentile", type=float, default=None,
                   help="tail width in percent (tail mode only; default: MA "
                        "checkpoint's, typically 5.0)")
    p.add_argument("--success-steps", type=int, default=1,
                   help="consecutive in-zone steps counted as success (default 1)")
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
                   help="output dir (default: stats/out/compare__<sa>__vs__<ma>/)")
    return p.parse_args()


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


def load_arm(ckpt_path, task_args, device, mode, is_multi_hint):
    """Build the env (task params forced from `task_args`) + the action selector
    for one trained-policy arm. Returns (env, selector, agg_label_kind, obs_dim)."""
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


def print_mcnemar(labels, any_by_label):
    print("\nPaired McNemar exact test on episode success_any (same seeds):")
    print("-" * 74)
    for a, b in ((labels[0], labels[1]), (labels[1], labels[2]), (labels[0], labels[2])):
        xa = np.asarray(any_by_label[a], bool)
        xb = np.asarray(any_by_label[b], bool)
        a_wins = int(np.sum(xa & ~xb))   # a succeeded, b failed
        b_wins = int(np.sum(xb & ~xa))
        p = mcnemar_exact(a_wins, b_wins)
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
        lead = a if a_wins > b_wins else b
        print(f"  {a:>16} vs {b:<16}  {a}-only={a_wins:3d}  {b}-only={b_wins:3d}  "
              f"p={p:.3f} {sig}  (lead: {lead})")
    print("-" * 74)
    print("  ** p<0.05   * p<0.10   ns not significant")


# --------------------------------------------------------------------------- #
def make_plots(fig_dir, labels, results, curves, any_ts, all_ts, ep_stats,
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
        c = COLORS[l]
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
            plt.plot(grid, curve, lw=2, color=COLORS[l], label=l)
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
    plt.figure(figsize=(8.5, 5))
    for j, l in enumerate(labels):
        xs = x + (j - (len(labels) - 1) / 2) * w
        bars = plt.bar(xs, vals[l], width=w, color=COLORS[l], label=l)
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

    # Resolve the common task config from the MA checkpoint + CLI overrides. Both
    # trained-policy arms and the random arm are forced onto these, so a seed is
    # the SAME episode instance in every arm (paired comparison).
    ma_raw = torch.load(cli.ma_checkpoint, map_location=device, weights_only=False)
    if "netcdf_file" not in ma_raw["args"]:
        raise SystemExit(f"'{cli.ma_checkpoint}' is not an Oceananigans checkpoint.")
    task = SimpleNamespace(**ma_raw["args"])
    task.n_agents = N
    task.netcdf_file = cli.netcdf_file or task.netcdf_file
    task.max_steps = cli.max_steps or task.max_steps
    task.end_on_any_success = cli.end_on_any_success
    task.eval_success_steps = cli.success_steps
    if cli.target_mode is not None:
        task.target_mode = cli.target_mode
    if cli.target_percentile is not None:
        task.target_percentile = cli.target_percentile

    if getattr(task, "communication", False) and N != int(ma_raw["args"].get("n_agents", 1)):
        raise SystemExit(
            f"MA checkpoint was trained with communication at n_agents="
            f"{ma_raw['args'].get('n_agents')}, whose neighbor-block observation is "
            f"N-specific — rerun with --n-agents {ma_raw['args'].get('n_agents')}.")

    # Warn if the single-agent checkpoint was trained on a different task instance
    # (its task params get FORCED to the common set, but a mismatch means the two
    # policies were not trained on the same problem — worth surfacing).
    sa_raw = torch.load(cli.sa_checkpoint, map_location=device, weights_only=False)
    for key in ("netcdf_file", "target_mode", "epsilon_salinity", "domain",
                "target_percentile", "static_frame"):
        sv, mv = sa_raw["args"].get(key), ma_raw["args"].get(key)
        if key != "netcdf_file" and sv is not None and mv is not None and \
                tuple(np.atleast_1d(sv)) != tuple(np.atleast_1d(mv)):
            print(f"  WARNING: single-agent trained with {key}={sv} but multi-agent "
                  f"used {key}={mv} — forcing {mv} for the eval (episodes still paired, "
                  f"but the SA policy was optimized on a different task).")

    # Build the three arms.  Random reuses the single-agent env (identical task
    # params, communication off) so its episodes are paired with the others.
    _, env_sa, sel_sa, kind_sa, dim_sa = load_arm(
        cli.sa_checkpoint, task, device, cli.mode, is_multi_hint=False)
    _, env_ma, sel_ma, kind_ma, dim_ma = load_arm(
        cli.ma_checkpoint, task, device, cli.mode, is_multi_hint=True)
    sel_random = ev.make_random_selector(
        cli.seed if cli.seed is not None else 0)

    labels = ["single-agent", "multi-agent", "random"]
    arms = {
        "single-agent": (env_sa, sel_sa),
        "multi-agent": (env_ma, sel_ma),
        "random": (env_sa, sel_random),  # same env as SA -> paired instances
    }

    dt_s = task.dt * task.frame_skip
    base_seed = cli.seed if cli.seed is not None else int(
        np.random.SeedSequence().entropy % (2 ** 31))
    multi_agent = N > 1

    run_name_sa = sa_raw.get("run_name") or os.path.basename(
        os.path.dirname(os.path.dirname(cli.sa_checkpoint)))
    run_name_ma = ma_raw.get("run_name") or os.path.basename(
        os.path.dirname(os.path.dirname(cli.ma_checkpoint)))
    tag = f"compare__{run_name_sa}__vs__{run_name_ma}__N{N}__{task.target_mode}"
    out_dir = cli.out_dir or os.path.join(ROOT, "stats", "out", tag)
    os.makedirs(out_dir, exist_ok=True)

    lens = "first-reach (success_any)" if cli.end_on_any_success else "all-success (swarm-truth)"
    print(f"Single-agent : {cli.sa_checkpoint}  ({kind_sa}, obs {dim_sa})")
    print(f"Multi-agent  : {cli.ma_checkpoint}  ({kind_ma}, obs {dim_ma})")
    print(f"Agents/arm   : {N}   Episodes: {cli.episodes} paired, base seed {base_seed}")
    tgt = f"{task.target_mode}" + (f" (±{task.target_percentile}%)"
                                   if task.target_mode == "tail" else "")
    print(f"Field        : {task.netcdf_file}   max_steps {task.max_steps}   "
          f"success_steps {cli.success_steps}")
    print(f"Target       : {tgt}")
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
        print(f"  [{l:>12}] success_any {ep_stats[l]['any_rate']*100:5.1f}%  "
              f"success_all {ep_stats[l]['all_rate']*100:5.1f}%  "
              f"per-agent {agg['success_rate']*100:5.1f}%  SPL {agg['spl_mean']:.3f}")

    header = (f"single-agent vs multi-agent vs random   N={N}   "
              f"{cli.episodes} paired episodes   [{lens}]")
    print_table(labels, results, ep_stats, header)
    print_mcnemar(labels, any_by_label)

    make_plots(os.path.join(out_dir, "figures"), labels, results, curves,
               any_ts, all_ts, ep_stats, task.max_steps, N, cli.episodes)

    # McNemar tests for the JSON record.
    mcnemar = {}
    for a, b in (("multi-agent", "single-agent"), ("multi-agent", "random"),
                 ("single-agent", "random")):
        xa, xb = np.asarray(any_by_label[a], bool), np.asarray(any_by_label[b], bool)
        aw, bw = int(np.sum(xa & ~xb)), int(np.sum(xb & ~xa))
        mcnemar[f"{a}_vs_{b}"] = dict(a_only=aw, b_only=bw,
                                      p_value=mcnemar_exact(aw, bw))

    summary = dict(
        sa_checkpoint=cli.sa_checkpoint, ma_checkpoint=cli.ma_checkpoint,
        sa_kind=kind_sa, ma_kind=kind_ma, n_agents=N,
        episodes=cli.episodes, base_seed=base_seed, max_steps=task.max_steps,
        success_steps=cli.success_steps, netcdf_file=task.netcdf_file,
        end_on_any_success=cli.end_on_any_success, decode=cli.mode,
        generated=datetime.now().isoformat(timespec="seconds"),
        episode_stats=ep_stats, results=results, mcnemar_success_any=mcnemar,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ev.write_csv(os.path.join(out_dir, "per_episode.csv"), per_agent_by_label)
    write_paired_csv(os.path.join(out_dir, "paired_episodes.csv"),
                     labels, any_by_label, all_by_label, cli.episodes)

    print(f"\nWrote summary.json, per_episode.csv, paired_episodes.csv, figures/ -> {out_dir}")
    env_sa.close()
    env_ma.close()


if __name__ == "__main__":
    main()

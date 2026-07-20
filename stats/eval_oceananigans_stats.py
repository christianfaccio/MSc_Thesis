"""
Aggregate trajectory statistics for a policy trained on the Oceananigans env
(src/envs/oceananigans.py -> OceananigansEnv).

Unlike scripts/plot_oceananigans_trajectories.py (which rolls out ONE episode
for visual inspection), this rolls out MANY episodes and reduces them to a
handful of objective, literature-grounded numbers a thesis committee expects
for a navigation task:

  * success rate  (with a Wilson 95% CI, so the number is defensible)
  * SPL           (Success weighted by Path Length, Anderson et al. 2018 —
                   1.0 = went straight to the zone every time; SPL << success
                   means it arrives but wanders)
  * time-to-success, path tortuosity, distance-to-zone monotonicity,
    final measurement error (split success/failure), depth-band targeting,
    and action profile (no-op fraction, vertical ping-pong rate).

Both GREEDY (argmax) and STOCHASTIC (sample) policies are evaluated over the
SAME episode seeds, so the two columns are directly comparable.

The env is rebuilt EXACTLY as trained from the checkpoint's stored `args`, and
the checkpoint's obs normalization (obs_rms) is applied — identical to the plot
script — so these numbers match training-time behavior. Handles single-agent
PPO and multi-agent IPPO/MAPPO (policy class auto-detected from the checkpoint,
same logic as the plot script); each agent-episode is one sample, and episode-
level success_any / success_all are reported separately for the swarm case.

Outputs (under --out-dir, default stats/out/<run>__iter<NN>/):
  * summary.json      — all aggregate metrics (machine-readable)
  * per_episode.csv   — one row per agent-episode (raw, for your own plots)
  * figures/approach_curve.png     — mean distance-to-zone vs normalized time
  * figures/distributions.png      — SPL / error / time-to-success / actions

Usage:
    python stats/eval_oceananigans_stats.py \
        --checkpoint runs/<run>/checkpoints/latest.pt
    python stats/eval_oceananigans_stats.py --checkpoint <ckpt> \
        --episodes 200 --seed 0 --success-steps 1
"""
import argparse
import csv
import itertools
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.envs.oceananigans import OceananigansEnv
from src.single_agent.policy import PpoPolicy
from src.multi_agent.policy import IppoPolicy, MappoPolicy

K_TURBIDITY = 0.01  # Beer-Lambert coefficient, matches src/models/turbidity.py
# 27 discrete actions = product of {-1,0,1}^3 (x,y,z); index 13 = (0,0,0) no-op.
_ACTIONS = list(itertools.product([-1, 0, 1], repeat=3))
_NOOP = _ACTIONS.index((0, 0, 0))


# --------------------------------------------------------------------------- #
# checkpoint / env / policy setup  (mirrors plot_oceananigans_trajectories.py)
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--netcdf-file", type=str, default=None,
                   help="override the checkpoint's NetCDF spec — a folder (globs "
                        "*.nc), a single .nc file, or a glob. Use it to evaluate on "
                        "a held-out split, e.g. data/oceananigans/buoyancy_active/test.")
    p.add_argument("--episodes", type=int, default=100,
                   help="episodes per policy mode (default 100)")
    p.add_argument("--seed", type=int, default=None,
                   help="base episode seed; episode i uses seed+i (default: random base)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override max episode length (default: from checkpoint args)")
    p.add_argument("--success-steps", type=int, default=1,
                   help="consecutive in-zone steps counted as success for THIS eval "
                        "(default 1 = 'reached the zone'; training often used 3)")
    p.add_argument("--modes", nargs="+", default=["greedy", "stochastic"],
                   choices=["greedy", "stochastic"],
                   help="which policy modes to evaluate (default: both; "
                        "ignored for scripted --policy baselines)")
    p.add_argument("--policy", default="checkpoint",
                   choices=["checkpoint", "random", "gradient"],
                   help="what to roll out: the trained checkpoint (default), a "
                        "uniform-random baseline (the luck floor), or the scripted "
                        "reactive baseline (depth servo on the tau error + horizontal "
                        "step along the signed body-frame salinity gradient — the "
                        "ceiling of a memoryless policy). Scripted baselines still "
                        "need --checkpoint, but only to clone its env config/seeds.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output dir (default: stats/out/<run>__iter<NN>/)")
    return p.parse_args()


def combine_obs_rms(obs_rms):
    """Return (mean, var). PPO stores a LIST of per-env RMS dicts (merge with a
    count-weighted average); the multi-agent trainers store a single dict."""
    if isinstance(obs_rms, dict):
        return np.asarray(obs_rms["mean"], np.float64), np.asarray(obs_rms["var"], np.float64)
    counts = np.array([s["count"] for s in obs_rms], dtype=np.float64)
    w = counts / counts.sum()
    mean = sum(wi * np.asarray(s["mean"], np.float64) for wi, s in zip(w, obs_rms))
    var = sum(wi * np.asarray(s["var"], np.float64) for wi, s in zip(w, obs_rms))
    return mean, var


def build_env(args):
    return OceananigansEnv(
        xml_file=getattr(args, "xml_file", "config/simulation.xml"),
        netcdf_file=args.netcdf_file,
        k=args.k,
        n_agents=getattr(args, "n_agents", 1),
        v_agent=args.v_agent,
        max_steps=args.max_steps,
        dt=args.dt,
        domain=tuple(args.domain),
        frame_skip=args.frame_skip,
        gamma=args.gamma,
        success_bonus=getattr(args, "success_bonus", 10.0),
        static_frame=getattr(args, "static_frame", True),
        success_steps_required=getattr(args, "eval_success_steps", 1),
        max_cached_loaders=getattr(args, "max_cached_loaders", 8),
        end_on_any_success=getattr(args, "end_on_any_success", True),
        epsilon_salinity=getattr(args, "epsilon_salinity", 0.3),
        epsilon_turbidity=getattr(args, "epsilon_turbidity", 0.05),
        sigma_s=getattr(args, "sigma_s", 3.0),
        sigma_tau=getattr(args, "sigma_tau", 0.3),
        target_mode=getattr(args, "target_mode", "random"),
        target_percentile=getattr(args, "target_percentile", 5.0),
        reward_potential=getattr(args, "reward_potential", "distance"),
        dead_reckoning=getattr(args, "dead_reckoning", False),
    )


def load_policy(ckpt, env, device):
    multi_agent = int(ckpt["args"].get("n_agents", 1)) > 1
    local_dim = int(np.array(env.observation_space.shape).prod())
    ckpt_dim = ckpt["model_state_dict"]["actor.0.weight"].shape[1]
    if ckpt_dim != local_dim:
        raise SystemExit(
            f"Checkpoint expects a {ckpt_dim}-dim observation but the current env "
            f"builds {local_dim}-dim ones — trained on an older observation layout.")
    if multi_agent:
        critic_dim = ckpt["model_state_dict"]["critic.0.weight"].shape[1]
        if critic_dim == local_dim:
            policy = IppoPolicy(local_dim, env.action_space.n).to(device)
        else:
            policy = MappoPolicy(local_dim, critic_dim, env.action_space.n).to(device)
    else:
        shim = SimpleNamespace(single_observation_space=env.observation_space,
                               single_action_space=env.action_space)
        policy = PpoPolicy(shim).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    return policy, multi_agent


def ensure_zone_tree(env):
    """SPL/distance metrics need the zone KD-tree. It is built at reset only in
    'distance' reward mode; force-build it (from the frozen snapshot loader) so
    the metrics work for 'error'-mode checkpoints too."""
    if getattr(env, "_zone_tree", None) is None:
        env._build_zone_index(env._loaders[env.active_netcdf_path])


def dist_to_zone(env, pos):
    d, _ = env._zone_tree.query((pos[0], pos[1], pos[2]))
    return float(d)


# --------------------------------------------------------------------------- #
# action selectors: trained checkpoint or scripted baselines
# --------------------------------------------------------------------------- #
def make_checkpoint_selector(policy, normalize, greedy):
    """Trained policy: normalize the raw obs, then argmax (greedy) or sample."""
    def select(raw_obs):
        with torch.no_grad():
            logits = policy.actor(torch.tensor(normalize(raw_obs)))
            if greedy:
                a = logits.argmax(dim=-1)
            else:
                a = torch.distributions.Categorical(logits=logits).sample()
        return a.cpu().numpy().astype(np.int64)
    return select


def make_random_selector(seed):
    """Uniform-random action baseline — the luck floor for the success metrics."""
    rng = np.random.default_rng(seed)
    return lambda raw_obs: rng.integers(27, size=raw_obs.shape[0], dtype=np.int64)


def make_gradient_selector(env):
    """Scripted memoryless reactive baseline, built ONLY from what the agent
    observes (raw obs frame part: u v w | gu gv gw | S-S* | tau-tau* | depth):
      * vertical:   servo on the tau error — tau grows with depth, so too-high
                    tau means too deep -> heave up (and vice versa); deadband
                    at eps_tau/2 so it doesn't chatter inside the band.
      * horizontal: step along the body-frame salinity gradient (gu, gv),
                    signed by S-S* (S too high -> descend the gradient);
                    deadband at eps_S/2; the continuous direction is snapped
                    to the nearest of the 8 compass actions.
    Its score is the ceiling of ANY memoryless policy worth learning here —
    if it solves the task, the observation suffices and failures are a
    learning problem; if it doesn't, memory/odometry is genuinely needed."""
    eps_S, eps_tau = env.epsilon_salinity, env.epsilon_turbidity

    def select(raw_obs):
        a = np.empty(raw_obs.shape[0], dtype=np.int64)
        for i, o in enumerate(raw_obs):
            gu, gv = float(o[3]), float(o[4])
            dS, dT = float(o[6]), float(o[7])
            dx = dy = dz = 0
            if abs(dT) > eps_tau / 2:
                dz = -1 if dT > 0 else 1          # z positive down
            gnorm = float(np.hypot(gu, gv))
            if abs(dS) > eps_S / 2 and gnorm > 1e-12:
                s = -np.sign(dS)                   # move to change S toward S*
                dx = int(round(s * gu / gnorm))
                dy = int(round(s * gv / gnorm))
            a[i] = _ACTIONS.index((dx, dy, dz))
        return a
    return select


# --------------------------------------------------------------------------- #
# rollout: collect raw per-agent traces for one episode
# --------------------------------------------------------------------------- #
def rollout_episode(env, select_actions, seed, max_steps, multi_agent):
    """Return a list (one entry per agent) of dicts with the raw trace needed to
    compute every metric. A frozen (already-succeeded) agent stops recording."""
    obs, _ = env.reset(seed=seed)
    ensure_zone_tree(env)
    n = env.n_agents
    obs = np.asarray(obs)
    if not multi_agent:
        obs = obs[None, :]  # (1, dim) so the loop is uniform

    traces = [dict(pos=[env.sim.agents[i].pos.copy()],
                   dist=[dist_to_zone(env, env.sim.agents[i].pos)],
                   actions=[], S=[], tau=[],
                   success=False, success_step=None) for i in range(n)]
    done_agent = np.zeros(n, dtype=bool)

    for step in range(max_steps):
        a = select_actions(obs)                            # (n,) int64
        act_env = int(a[0]) if not multi_agent else a
        obs, _, term, trunc, _ = env.step(act_env)
        obs = np.asarray(obs)
        if not multi_agent:
            obs = obs[None, :]
        term = np.atleast_1d(term).astype(bool)
        trunc = np.atleast_1d(trunc).astype(bool)

        for i in range(n):
            if done_agent[i]:
                continue
            agent = env.sim.agents[i]
            tr = traces[i]
            tr["actions"].append(int(a[i]))
            tr["pos"].append(agent.pos.copy())
            tr["dist"].append(dist_to_zone(env, agent.pos))
            S, tau = env._measure(agent)[:2]
            tr["S"].append(float(S)); tr["tau"].append(float(tau))
            if term[i]:
                tr["success"] = True
                tr["success_step"] = step + 1
                done_agent[i] = True

        done = np.logical_or(term, trunc)
        if bool(done.all()):
            break

    for tr in traces:
        tr["pos"] = np.asarray(tr["pos"])
        tr["dist"] = np.asarray(tr["dist"])
    return traces, dict(target_salinity=float(env.target_salinity),
                        target_turbidity=float(env.target_turbidity),
                        eps_S=float(env.epsilon_salinity),
                        eps_tau=float(env.epsilon_turbidity),
                        nc_file=os.path.basename(str(env.active_netcdf_path)))


# --------------------------------------------------------------------------- #
# per-agent-episode metrics
# --------------------------------------------------------------------------- #
def agent_metrics(tr, meta, dt_s):
    pos = tr["pos"]                       # (T+1, 3)
    dist = tr["dist"]                     # (T+1,)
    steps = len(tr["actions"])
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_len = float(seg.sum())
    net_disp = float(np.linalg.norm(pos[-1] - pos[0]))
    spawn_dist = float(dist[0])          # ell_i: straight-line spawn -> zone
    reached = tr["success"]

    # SPL term: shortest path is the straight line to the zone (no obstacles).
    spl = (spawn_dist / max(path_len, spawn_dist)) if reached and path_len > 0 else 0.0
    # Path efficiency (successful only, else NaN): 1.0 = perfectly direct.
    eff = (spawn_dist / path_len) if (reached and path_len > 0) else np.nan

    # Distance-to-zone monotonicity: fraction of steps that got us closer.
    dd = np.diff(dist)
    monotonic = float((dd < 0).mean()) if dd.size else np.nan

    # Vertical ping-pong: sign flips in the z-action over nonzero-z steps.
    dz = np.array([_ACTIONS[a][2] for a in tr["actions"]])
    nz = dz[dz != 0]
    zflip = float((np.diff(np.sign(nz)) != 0).mean()) if nz.size > 1 else 0.0
    noop_frac = float(np.mean(np.array(tr["actions"]) == _NOOP)) if steps else np.nan

    # Depth targeting: fraction of steps within the turbidity (depth) band.
    tau = np.asarray(tr["tau"])
    depth_band = (float(np.mean(np.abs(tau - meta["target_turbidity"]) < meta["eps_tau"]))
                  if tau.size else np.nan)

    final_dS = abs(tr["S"][-1] - meta["target_salinity"]) if tr["S"] else np.nan
    final_dtau = abs(tr["tau"][-1] - meta["target_turbidity"]) if tr["tau"] else np.nan

    return dict(
        success=bool(reached),
        steps=int(steps),
        time_s=float(steps * dt_s),
        steps_to_success=(int(tr["success_step"]) if reached else None),
        spawn_dist=spawn_dist,
        path_len=path_len,
        net_disp=net_disp,
        tortuosity=(path_len / net_disp if net_disp > 1e-6 else np.nan),
        spl=spl,
        path_efficiency=eff,
        min_dist=float(dist.min()),
        monotonic_frac=monotonic,
        noop_frac=noop_frac,
        zflip_frac=zflip,
        depth_band_frac=depth_band,
        final_dS=final_dS,
        final_dtau=final_dtau,
        nc_file=meta["nc_file"],
    )


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion (robust at small n / p~0,1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, center - half), min(1.0, center + half))


def _stats(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return dict(n=0, mean=None, median=None, std=None, p25=None, p75=None)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()), p25=float(np.percentile(a, 25)),
                p75=float(np.percentile(a, 75)))


def approach_curve(traces_all, npts=100):
    """Mean of (distance-to-zone / spawn distance) resampled onto normalized
    episode time [0,1], averaged over all agent-episodes. 1.0 = no progress,
    0.0 = at the zone."""
    grid = np.linspace(0, 1, npts)
    curves = []
    for tr in traces_all:
        d = tr["dist"]
        if d.size < 2 or d[0] <= 1e-9:
            continue
        t = np.linspace(0, 1, d.size)
        curves.append(np.interp(grid, t, d / d[0]))
    if not curves:
        return grid, None
    return grid, np.mean(curves, axis=0)


def aggregate(per_agent, episode_success):
    n = len(per_agent)
    succ = [m["success"] for m in per_agent]
    k = int(sum(succ))
    lo, hi = wilson_ci(k, n)
    succ_only = [m for m in per_agent if m["success"]]
    return dict(
        n_agent_episodes=n,
        success_rate=(k / n if n else None),
        success_ci95=[lo, hi],
        n_success=k,
        spl_mean=float(np.mean([m["spl"] for m in per_agent])) if n else None,
        # episode-level swarm outcomes
        episode_success_any=float(np.mean([e["any"] for e in episode_success])) if episode_success else None,
        episode_success_all=float(np.mean([e["all"] for e in episode_success])) if episode_success else None,
        # success_any within a step budget T — separates genuine navigation from
        # slow diffusive luck (a random walk keeps climbing with budget; a real
        # navigator saturates early). t_any = earliest success in the episode.
        success_at_budget={str(T): float(np.mean(
            [e["t_any"] is not None and e["t_any"] <= T for e in episode_success]))
            for T in (100, 250, 500, 700, 1000, 1500, 2000, 3600)} if episode_success else None,
        # timing / geometry
        steps_to_success=_stats([m["steps_to_success"] for m in succ_only]),
        time_to_success_s=_stats([m["time_s"] for m in succ_only]),
        path_efficiency_success=_stats([m["path_efficiency"] for m in succ_only]),
        tortuosity=_stats([m["tortuosity"] for m in per_agent]),
        monotonic_frac=_stats([m["monotonic_frac"] for m in per_agent]),
        min_dist_failures=_stats([m["min_dist"] for m in per_agent if not m["success"]]),
        final_dS_all=_stats([m["final_dS"] for m in per_agent]),
        final_dS_failures=_stats([m["final_dS"] for m in per_agent if not m["success"]]),
        final_dtau_all=_stats([m["final_dtau"] for m in per_agent]),
        depth_band_frac=_stats([m["depth_band_frac"] for m in per_agent]),
        noop_frac=_stats([m["noop_frac"] for m in per_agent]),
        zflip_frac=_stats([m["zflip_frac"] for m in per_agent]),
    )


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_table(results, header):
    modes = list(results.keys())
    print("\n" + "=" * 74)
    print(header)
    print("=" * 74)
    rowfmt = "{:<32}" + "{:>20}" * len(modes)
    print(rowfmt.format("metric", *modes))
    print("-" * 74)

    def line(label, fn):
        print(rowfmt.format(label, *[fn(results[m]) for m in modes]))

    def stat(agg, key, field="median", pct=False):
        s = agg[key]
        v = s[field] if isinstance(s, dict) else s
        if v is None:
            return "-"
        return f"{v*100:.1f}%" if pct else f"{v:.3f}"

    line("success rate", lambda a: (f"{a['success_rate']*100:.1f}% "
         f"[{a['success_ci95'][0]*100:.0f},{a['success_ci95'][1]*100:.0f}]"
         if a['success_rate'] is not None else "-"))
    line("  (n agent-episodes)", lambda a: str(a["n_agent_episodes"]))
    line("SPL (success-wtd path len)", lambda a: f"{a['spl_mean']:.3f}" if a['spl_mean'] is not None else "-")
    line("episode success_any", lambda a: stat(a, "episode_success_any", None, pct=True))
    line("  success_any@700 steps", lambda a: (f"{a['success_at_budget']['700']*100:.1f}%"
         if a.get("success_at_budget") else "-"))
    line("episode success_all", lambda a: stat(a, "episode_success_all", None, pct=True))
    line("steps to success (med)", lambda a: stat(a, "steps_to_success"))
    line("time to success s (med)", lambda a: stat(a, "time_to_success_s"))
    line("path eff. success (med)", lambda a: stat(a, "path_efficiency_success"))
    line("tortuosity (med)", lambda a: stat(a, "tortuosity"))
    line("closer-each-step frac (med)", lambda a: stat(a, "monotonic_frac"))
    line("min dist | failures (med,m)", lambda a: stat(a, "min_dist_failures"))
    line("final |dS| all (med)", lambda a: stat(a, "final_dS_all"))
    line("final |dS| failures (med)", lambda a: stat(a, "final_dS_failures"))
    line("depth-band frac (med)", lambda a: stat(a, "depth_band_frac"))
    line("no-op frac (med)", lambda a: stat(a, "noop_frac"))
    line("z ping-pong frac (med)", lambda a: stat(a, "zflip_frac"))
    print("=" * 74)


def save_plots(fig_dir, curves, per_agent_by_mode, t_any_by_mode=None, max_steps=None):
    os.makedirs(fig_dir, exist_ok=True)
    colors = {"greedy": "tab:blue", "stochastic": "tab:orange",
              "random": "tab:gray", "gradient": "tab:green"}

    # 0) success_any within a step budget (empirical CDF of the episode's
    #    earliest success time) — a navigator saturates early, diffusion climbs
    #    forever, so the SHAPE separates skill from luck.
    if t_any_by_mode:
        plt.figure(figsize=(7, 5))
        for mode, ts in t_any_by_mode.items():
            n = len(ts)
            hit = np.sort([t for t in ts if t is not None])
            if n and hit.size:
                x = np.concatenate([[0], hit, [max_steps or hit[-1]]])
                y = np.concatenate([[0], np.arange(1, hit.size + 1), [hit.size]]) / n
                plt.step(x, y, where="post", lw=2, label=mode, color=colors.get(mode))
        plt.xlabel("step budget T"); plt.ylabel("episode success_any within T")
        plt.title("Success vs step budget"); plt.ylim(0, 1.02)
        if max_steps:
            plt.xlim(0, max_steps)
        plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "success_at_budget.png"), dpi=150)
        plt.close()

    # 1) approach curve
    plt.figure(figsize=(7, 5))
    for mode, (grid, curve) in curves.items():
        if curve is not None:
            plt.plot(grid, curve, lw=2, label=mode, color=colors.get(mode))
    plt.xlabel("normalized episode time"); plt.ylabel("distance to zone / spawn distance")
    plt.title("Mean approach to target zone"); plt.ylim(0, 1.05)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "approach_curve.png"), dpi=150)
    plt.close()

    # 2) distributions: SPL/path-eff, final |dS|, time-to-success, no-op
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for mode, per in per_agent_by_mode.items():
        c = colors.get(mode)
        eff = [m["path_efficiency"] for m in per if m["success"] and np.isfinite(m["path_efficiency"])]
        axes[0, 0].hist(eff, bins=20, range=(0, 1), alpha=0.5, label=mode, color=c)
        dS = [m["final_dS"] for m in per if np.isfinite(m["final_dS"])]
        axes[0, 1].hist(dS, bins=30, alpha=0.5, label=mode, color=c)
        tts = [m["time_s"] for m in per if m["success"]]
        axes[1, 0].hist(tts, bins=25, alpha=0.5, label=mode, color=c)
        noop = [m["noop_frac"] for m in per if np.isfinite(m["noop_frac"])]
        axes[1, 1].hist(noop, bins=20, range=(0, 1), alpha=0.5, label=mode, color=c)
    axes[0, 0].set_title("path efficiency (successful)"); axes[0, 0].set_xlabel("ell / path_len")
    axes[0, 1].set_title("final |S - S*| (all episodes)"); axes[0, 1].set_xlabel("PSU")
    axes[1, 0].set_title("time to success"); axes[1, 0].set_xlabel("seconds")
    axes[1, 1].set_title("no-op action fraction"); axes[1, 1].set_xlabel("fraction of steps")
    for ax in axes.flat:
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "distributions.png"), dpi=150)
    plt.close(fig)


def write_csv(path, per_agent_by_mode):
    fields = ["mode", "episode", "agent", "success", "steps", "time_s",
              "steps_to_success", "spawn_dist", "path_len", "net_disp",
              "tortuosity", "spl", "path_efficiency", "min_dist",
              "monotonic_frac", "noop_frac", "zflip_frac", "depth_band_frac",
              "final_dS", "final_dtau", "nc_file"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mode, per in per_agent_by_mode.items():
            for m in per:
                row = {k: m.get(k) for k in fields}
                row["mode"] = mode
                row["episode"] = m["_episode"]
                row["agent"] = m["_agent"]
                w.writerow(row)


# --------------------------------------------------------------------------- #
def main():
    cli = parse_args()
    device = torch.device("cpu")

    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    a = ckpt["args"]
    if "netcdf_file" not in a:
        raise SystemExit(f"'{cli.checkpoint}' is not an Oceananigans checkpoint "
                         f"(no netcdf_file in args; env_id={a.get('env_id')!r}).")

    args = SimpleNamespace(**a)
    if cli.max_steps is not None:
        args.max_steps = cli.max_steps
    if cli.netcdf_file is not None:
        args.netcdf_file = cli.netcdf_file
    args.eval_success_steps = cli.success_steps

    env = build_env(args)
    multi_agent = int(a.get("n_agents", 1)) > 1
    if cli.policy == "checkpoint":
        policy, multi_agent = load_policy(ckpt, env, device)
        obs_mean, obs_var = combine_obs_rms(ckpt["obs_rms"])

        def normalize(raw):
            norm = (raw - obs_mean) / np.sqrt(obs_var + 1e-8)
            return np.clip(norm, -10.0, 10.0).astype(np.float32)

    dt_s = args.dt * args.frame_skip
    base_seed = cli.seed if cli.seed is not None else int(np.random.SeedSequence().entropy % (2**31))

    run_name = ckpt.get("run_name") or os.path.basename(os.path.dirname(os.path.dirname(cli.checkpoint)))
    it = ckpt.get("iteration", "NA")
    # Tag the out dir with the eval split and any scripted baseline so runs
    # against test/ vs train/ or random/gradient don't collide.
    split = os.path.basename(str(args.netcdf_file).rstrip("/")) if cli.netcdf_file else None
    tag = f"{run_name}__iter{it}" + (f"__{split}" if split else "") \
        + (f"__{cli.policy}" if cli.policy != "checkpoint" else "")
    out_dir = cli.out_dir or os.path.join(ROOT, "stats", "out", tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Checkpoint : {cli.checkpoint}")
    print(f"Policy     : {cli.policy}")
    print(f"Run        : {run_name}  (iter {it})")
    print(f"Kind       : {'multi-agent' if multi_agent else 'single-agent'}, "
          f"n_agents={env.n_agents}, obs_dim={int(np.prod(env.observation_space.shape))}")
    print(f"Episodes   : {cli.episodes} per mode, base seed {base_seed}, "
          f"max_steps={args.max_steps}, success_steps={cli.success_steps}")
    print(f"Field      : {args.netcdf_file}  (eps_S={env.epsilon_salinity}, eps_tau={env.epsilon_turbidity})")
    print(f"Out        : {out_dir}")

    # Scripted baselines are deterministic-per-seed policies; greedy/stochastic
    # decoding doesn't apply, so they run as a single mode named after themselves.
    if cli.policy == "checkpoint":
        mode_selectors = {m: make_checkpoint_selector(policy, normalize, m == "greedy")
                          for m in cli.modes}
    elif cli.policy == "random":
        mode_selectors = {"random": make_random_selector(base_seed)}
    else:
        mode_selectors = {"gradient": make_gradient_selector(env)}

    results, curves, per_agent_by_mode, t_any_by_mode = {}, {}, {}, {}
    for mode, select_actions in mode_selectors.items():
        per_agent, traces_all, episode_success = [], [], []
        for ep in range(cli.episodes):
            traces, meta = rollout_episode(
                env, select_actions, base_seed + ep, args.max_steps, multi_agent)
            ep_succ, ep_ts = [], []
            for i, tr in enumerate(traces):
                m = agent_metrics(tr, meta, dt_s)
                m["_episode"], m["_agent"] = ep, i
                per_agent.append(m)
                traces_all.append(tr)
                ep_succ.append(m["success"])
                if m["success"]:
                    ep_ts.append(m["steps_to_success"])
            episode_success.append(dict(any=any(ep_succ), all=all(ep_succ),
                                        t_any=(min(ep_ts) if ep_ts else None)))
        results[mode] = aggregate(per_agent, episode_success)
        curves[mode] = approach_curve(traces_all)
        per_agent_by_mode[mode] = per_agent
        t_any_by_mode[mode] = [e["t_any"] for e in episode_success]
        print(f"  [{mode}] done: success {results[mode]['success_rate']*100:.1f}%  "
              f"SPL {results[mode]['spl_mean']:.3f}")

    header = f"{run_name}  (iter {it})  —  {cli.episodes} episodes/mode"
    print_table(results, header)

    summary = dict(
        checkpoint=cli.checkpoint, run_name=run_name, iteration=it,
        policy=cli.policy,
        multi_agent=multi_agent, n_agents=env.n_agents,
        episodes=cli.episodes, base_seed=base_seed, max_steps=args.max_steps,
        success_steps=cli.success_steps, netcdf_file=args.netcdf_file,
        epsilon_salinity=env.epsilon_salinity, epsilon_turbidity=env.epsilon_turbidity,
        generated=datetime.now().isoformat(timespec="seconds"),
        results=results,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(os.path.join(out_dir, "per_episode.csv"), per_agent_by_mode)
    save_plots(os.path.join(out_dir, "figures"), curves, per_agent_by_mode,
               t_any_by_mode, args.max_steps)

    print(f"\nWrote summary.json, per_episode.csv, figures/ -> {out_dir}")
    env.close()


if __name__ == "__main__":
    main()

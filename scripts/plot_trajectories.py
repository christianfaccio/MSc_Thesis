"""
Roll out a trained policy (IPPO multi-agent, single-agent PPO, OR single-agent DQN)
in its env and plot ONLY the 3D agent trajectories (no salinity/turbidity/current
fields).

The run type is auto-detected:
  - multi vs single agent: from the checkpoint's stored `args` (single-agent
    checkpoints have no `n_agents` key);
  - PPO/IPPO vs DQN: from the model state_dict keys (DQN's QNetwork stores
    `network.*`, the actor-critic policies store `actor.*`/`critic.*`).

Usage:
    # multi-agent IPPO
    python scripts/plot_trajectories.py \
        --checkpoint runs/MultiAgent-v0__ippo__1__1781451054/checkpoints/latest.pt

    # single-agent PPO (same command, different checkpoint)
    python scripts/plot_trajectories.py \
        --checkpoint runs/SingleAgent-v0__ppo__1__<...>/checkpoints/latest.pt

    # single-agent DQN (same command, different checkpoint)
    python scripts/plot_trajectories.py \
        --checkpoint runs/SingleAgent-dqn__dqn__1__<...>/checkpoints/latest.pt

    # sample actions instead of greedy argmax, fix the episode seed, save to file
    # (DQN: --stochastic gives a Boltzmann/softmax-over-Q policy)
    python scripts/plot_trajectories.py --checkpoint <ckpt> --stochastic --seed 7 --save traj.png

    # force synthetic salinity field instead of the NetCDF data the run used
    python scripts/plot_trajectories.py --checkpoint <ckpt> --synthetic
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

# Make the project importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.envs.multi_agent import MultiAgentEnv
from src.envs.single_agent import SingleAgentEnv
# IppoPolicy and the single-agent CustomPolicy share the exact same module
# structure (critic 256/256, actor 128/128, identical names), so one class can
# load either state_dict.
from src.multi_agent.policy import IppoPolicy
# DQN checkpoints store a QNetwork (Q-values per action) instead.
from src.single_agent.policy import QNetwork
# For overlaying the success zone (|S-S*|<eps_S at the depth where |τ-τ*|<eps_τ).
from src.models.salinity import compute_salinity_analytical

K_TURBIDITY = 0.01  # Beer-Lambert coefficient, matches src/models/turbidity.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--seed", type=int, default=None, help="episode reset seed (default: random)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy instead of greedy argmax")
    p.add_argument("--synthetic", action="store_true",
                   help="override the run's NetCDF data and use the synthetic salinity field")
    p.add_argument("--start", choices=["auto", "center", "random", "tail"], default="auto",
                   help="agent spawn. 'auto' (default): keep the env's own spawn — for a "
                        "tail-mode checkpoint that is the opposite-extreme spawn used in "
                        "training; otherwise 'center'. Or force 'center'/'random'/'tail'.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="override max episode length (default: from checkpoint args)")
    p.add_argument("--save", type=str, default=None, help="save figure to this path instead of showing")
    return p.parse_args()


def combine_obs_rms(obs_rms):
    """Return (mean, var) as 1-D arrays.

    IPPO stores a single {mean, var, count} dict; single-agent PPO stores a LIST
    of per-parallel-env dicts. For the list we merge them with a count-weighted
    average (close enough for normalizing a visualization rollout)."""
    if isinstance(obs_rms, dict):
        return np.asarray(obs_rms["mean"], np.float64), np.asarray(obs_rms["var"], np.float64)
    counts = np.array([s["count"] for s in obs_rms], dtype=np.float64)
    w = counts / counts.sum()
    mean = sum(wi * np.asarray(s["mean"], np.float64) for wi, s in zip(w, obs_rms))
    var = sum(wi * np.asarray(s["var"], np.float64) for wi, s in zip(w, obs_rms))
    return mean, var


def build_env(args, is_multi):
    common = dict(
        xml_file=args.xml_file,
        netcdf_file=args.netcdf_file,
        n_sources=args.n_sources,
        k=args.k,
        v_agent=args.v_agent,
        max_steps=args.max_steps,
        dt=args.dt,
        frame_skip=args.frame_skip,
        domain=args.domain,
        sigma_h=args.sigma_h,
        sigma_v=args.sigma_v,
        eddy_length_scale=args.eddy_length_scale,
        # Match the run's NetCDF time handling: a static-frame run froze one random
        # snapshot per episode, so the rollout must too (and the tail target is drawn
        # from that same frozen field). Older checkpoints lack the key -> default True.
        static_frame=getattr(args, "static_frame", True),
    )
    # Rebuild the env exactly as trained, including the "tail" target mode (rare-P%
    # target + opposite-extreme spawn) — now supported by BOTH envs. Older checkpoints
    # lack these keys and fall back to the legacy "random" target.
    common.update(
        target_mode=getattr(args, "target_mode", "random"),
        target_percentile=getattr(args, "target_percentile", 5.0),
        target_samples=getattr(args, "target_samples", 1500),
    )
    if is_multi:
        return MultiAgentEnv(n_agents=args.n_agents, **common)
    return SingleAgentEnv(**common)


def compute_success_zone(env, args, is_multi, nx=90, ny=90):
    """The success zone the env actually checks against, at the CURRENT sim time.

    Turbidity τ(z)=1-exp(-k|z|) depends only on depth, so τ*=target fixes a depth
    plane z* and a depth band [z_lo, z_hi] where |τ-τ*|<eps_τ. Within z* the
    horizontal extent is where |S-S*|<eps_S. The NetCDF field is time-dependent, so
    we sample it after the rollout (final-time snapshot of a dynamic zone).

    Returns a dict (Xg, Yg, mask, zstar, zlo, zhi, Sstar, Tstar) or None.
    """
    Sstar = getattr(env, "target_salinity", None)
    Tstar = getattr(env, "target_turbidity", None)
    if Sstar is None or Tstar is None:
        return None
    eps_S = getattr(env, "epsilon_salinity", 0.05)
    eps_T = getattr(env, "epsilon_turbidity", 0.05)
    X, Y, Z = args.domain

    def z_from_tau(t):  # invert Beer-Lambert; clip to the domain
        t = float(np.clip(t, 0.0, 1.0 - 1e-9))
        return float(np.clip(-np.log(1.0 - t) / K_TURBIDITY, 0.0, Z))

    zstar = z_from_tau(Tstar)
    zlo, zhi = sorted((z_from_tau(Tstar - eps_T), z_from_tau(Tstar + eps_T)))

    xs, ys = np.linspace(0, X, nx), np.linspace(0, Y, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    if getattr(env, "_nc_files", None):  # time-dependent NetCDF field
        f = env.sim.current_3d.salinity_at
        S = np.array([[f(x, y, zstar) for x in xs] for y in ys])
    else:  # synthetic analytical plumes
        S = compute_salinity_analytical(Xg, Yg, np.full_like(Xg, zstar),
                                         env.sources, sigma_h=env.sigma_h, sigma_v=env.sigma_v)
    mask = np.abs(S - Sstar) < eps_S
    return dict(Xg=Xg, Yg=Yg, mask=mask, zstar=zstar, zlo=zlo, zhi=zhi,
                Sstar=float(Sstar), Tstar=float(Tstar))


def main():
    cli = parse_args()
    device = torch.device("cpu")

    ckpt = torch.load(cli.checkpoint, map_location=device, weights_only=False)
    args = SimpleNamespace(**ckpt["args"])
    is_multi = hasattr(args, "n_agents")
    if cli.synthetic:
        args.netcdf_file = None
    if cli.max_steps is not None:
        args.max_steps = cli.max_steps

    env = build_env(args, is_multi)
    n_agents = args.n_agents if is_multi else 1

    obs_space = env.local_observation_space if is_multi else env.observation_space
    local_dim = int(np.array(obs_space.shape).prod())
    n_actions = env.action_space.n

    # PPO/IPPO store actor-critic weights (actor.*/critic.*); DQN stores a QNetwork
    # (network.*). Detect from the state_dict so the right model is rebuilt.
    state_dict = ckpt["model_state_dict"]
    is_dqn = any(key.startswith("network.") for key in state_dict)

    # --- restore policy + observation normalization, expose a common act() ---
    if is_dqn:
        # QNetwork.__init__ reads single_observation_space/single_action_space off a
        # (vector) env; give it a shim wrapping this env's spaces.
        shim = SimpleNamespace(single_observation_space=obs_space,
                               single_action_space=env.action_space)
        policy = QNetwork(shim).to(device)
        policy.load_state_dict(state_dict)
        policy.eval()

        def act(x):  # x: (n_agents, local_dim) normalized tensor -> actions (n_agents,)
            q = policy(x)
            if cli.stochastic:  # Boltzmann/softmax-over-Q policy
                return torch.distributions.Categorical(logits=q).sample()
            return q.argmax(dim=-1)
    else:
        policy = IppoPolicy(local_dim, n_actions).to(device)
        policy.load_state_dict(state_dict)
        policy.eval()

        def act(x):  # x: (n_agents, local_dim) normalized tensor -> actions (n_agents,)
            logits = policy.actor(x)
            if cli.stochastic:
                return torch.distributions.Categorical(logits=logits).sample()
            return logits.argmax(dim=-1)

    obs_mean, obs_var = combine_obs_rms(ckpt["obs_rms"])
    obs_clip, var_eps = 10.0, 1e-8

    def normalize(raw):  # raw: (n_agents, local_dim) -> normalized, clipped
        norm = (raw - obs_mean) / np.sqrt(obs_var + var_eps)
        return np.clip(norm, -obs_clip, obs_clip).astype(np.float32)

    # --- env adapter: present single-agent as a 1-agent swarm ---
    def rebuild_obs():
        '''Rebuild the observation after manually moving agents (action=None, so
        the (action, reward) history buffer is left untouched).'''
        if is_multi:
            return np.stack([env._build_local_state(i)[0] for i in range(n_agents)])
        return np.atleast_2d(env._build_state(env.sim.agents[0])[0])

    # Resolve "auto": keep the env's own spawn for a tail-mode checkpoint (so the
    # rollout matches training — agents spawned in the extreme opposite the target);
    # otherwise centre them as before.
    is_tail = getattr(args, "target_mode", "random") == "tail"
    start_mode = cli.start
    if start_mode == "auto":
        start_mode = "tail" if is_tail else "center"

    def reset(seed):
        obs, _ = env.reset(seed=seed)
        # "tail"/"random": keep whatever reset() placed (tail-mode reset already put
        # the agents in the opposite extreme). "center": override to domain centre.
        if start_mode == "center":
            center = np.array([args.domain[0] / 2, args.domain[1] / 2, args.domain[2] / 2])
            for i in range(n_agents):
                env.sim.agents[i].pos[:] = center
            obs = rebuild_obs()  # refresh obs for the new (centered) positions
        return np.atleast_2d(obs)  # (n_agents, local_dim)

    def step(actions):  # actions: (n_agents,) ints -> obs, terminateds, truncateds (n_agents,)
        if is_multi:
            obs, _, term, trunc, _ = env.step(actions)
            return np.asarray(obs), np.asarray(term, bool), np.asarray(trunc, bool)
        obs, _, term, trunc, _ = env.step(int(actions[0]))
        return np.atleast_2d(obs), np.array([term], bool), np.array([trunc], bool)

    # --- rollout (single episode) ---
    def measure_ST(i):
        '''Salinity, turbidity at agent i's current position, exactly as the env's
        in-zone check sees them (single source of truth: env._measure).'''
        S, tau = env._measure(env.sim.agents[i])[:2]
        return float(S), float(tau)

    obs = reset(cli.seed)
    trajectories = [[env.sim.agents[i].pos.copy()] for i in range(n_agents)]
    done = np.zeros(n_agents, dtype=bool)
    success = np.zeros(n_agents, dtype=bool)
    final_ST = [measure_ST(i) for i in range(n_agents)]   # (S, τ) at each agent's last pos
    succ_pos = [None] * n_agents                          # where the success latch fired
    succ_ST = [None] * n_agents                           # (S, τ) measured at that moment
    steps = 0

    while not done.all() and steps < args.max_steps:
        with torch.no_grad():
            x = torch.tensor(normalize(obs)).to(device)
            actions = act(x).cpu().numpy()

        obs, terminateds, truncateds = step(actions)
        for i in range(n_agents):
            trajectories[i].append(env.sim.agents[i].pos.copy())
            if not done[i]:
                final_ST[i] = measure_ST(i)  # keep updating until the agent is done
            if terminateds[i] and succ_pos[i] is None:
                succ_pos[i] = env.sim.agents[i].pos.copy()
                succ_ST[i] = measure_ST(i)
        done = done | terminateds | truncateds
        success = success | terminateds
        steps += 1

    print(f"Episode finished after {steps} env steps "
          f"({steps * args.dt * args.frame_skip:.0f} s of sim time).")

    # Real target vs. what each agent actually measured. The drawn success zone is
    # the FINAL-time snapshot at depth z*; an agent that succeeded earlier (the field
    # is time-dependent) or at a different depth in the τ-band can sit off that
    # snapshot yet still be a genuine in-zone success — these numbers show why.
    eps_S = getattr(env, "epsilon_salinity", 0.05)
    eps_T = getattr(env, "epsilon_turbidity", 0.05)
    Sstar = getattr(env, "target_salinity", None)
    Tstar = getattr(env, "target_turbidity", None)
    if Sstar is not None:
        print(f"  TARGET (real):  S*={Sstar:.4f}  τ*={Tstar:.4f}   "
              f"(zone = |ΔS|<{eps_S} AND |Δτ|<{eps_T})")
    for i in range(n_agents):
        tag = "REACHED" if success[i] else "did NOT reach"
        fS, fT = final_ST[i]
        fpos = np.asarray(trajectories[i][-1])
        line = (f"  agent A{i+1:02d}: {tag}  | final pos=({fpos[0]:.0f},{fpos[1]:.0f},{fpos[2]:.1f})  "
                f"measured S={fS:.4f} τ={fT:.4f}")
        if Sstar is not None:
            inz = abs(fS - Sstar) < eps_S and abs(fT - Tstar) < eps_T
            line += f"  |ΔS|={abs(fS-Sstar):.4f} |Δτ|={abs(fT-Tstar):.4f}  in-zone-now={inz}"
        print(line)
        if succ_pos[i] is not None:
            sS, sT = succ_ST[i]
            sp = succ_pos[i]
            print(f"            ↳ success latch at pos=({sp[0]:.0f},{sp[1]:.0f},{sp[2]:.1f})  "
                  f"S={sS:.4f} τ={sT:.4f}  |ΔS|={abs(sS-Sstar):.4f} |Δτ|={abs(sT-Tstar):.4f}")

    # Success zone at the FINAL sim time (the field is dynamic, so this is one
    # snapshot of a moving region). Sampled exactly as the env's in-zone check.
    zone = compute_success_zone(env, args, is_multi)
    if zone is not None:
        print(f"  target  S*={zone['Sstar']:.3f}  τ*={zone['Tstar']:.3f}  "
              f"depth band z*≈{zone['zstar']:.1f} m (∈[{zone['zlo']:.1f},{zone['zhi']:.1f}])  "
              f"final-time zone covers {zone['mask'].mean()*100:.2f}% of the x-y plane")

    # --- plot the trajectories: 3D view + top-down (surface) view ---
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)  # top-down x-y, marginalized over depth
    colors = plt.cm.tab10(np.arange(n_agents) % 10)

    # Draw the success zone first so trajectories overlay on top.
    if zone is not None and zone["mask"].any():
        Xg, Yg, mask = zone["Xg"], zone["Yg"], zone["mask"]
        ax2.contourf(Xg, Yg, mask.astype(float), levels=[0.5, 1.5],
                     colors=["gold"], alpha=0.45)
        ax2.contour(Xg, Yg, mask.astype(float), levels=[0.5],
                    colors=["darkorange"], linewidths=1.2)
        ax2.scatter([], [], marker="s", c="gold", edgecolor="darkorange",
                    s=80, label="success zone (final t)")  # legend proxy
        # 3D: scatter the in-zone cells at z* (a slice of the depth band).
        ax.scatter(Xg[mask], Yg[mask], np.full(int(mask.sum()), zone["zstar"]),
                   color="gold", alpha=0.18, s=10, label="success zone (z*)")

    for i in range(n_agents):
        tr = np.asarray(trajectories[i])
        x, y, z = tr[:, 0], tr[:, 1], tr[:, 2]
        c = colors[i]
        end_marker = "*" if success[i] else "X"
        # 3D
        ax.plot(x, y, z, color=c, lw=1.6, label=f"A{i+1:02d}")
        ax.scatter(x[0], y[0], z[0], color=c, marker="o", s=55, edgecolor="k", zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], color=c, marker=end_marker, s=130, edgecolor="k", zorder=5)
        # top-down (looking down at the surface): x-y only
        ax2.plot(x, y, color=c, lw=1.6, label=f"A{i+1:02d}")
        ax2.scatter(x[0], y[0], color=c, marker="o", s=55, edgecolor="k", zorder=5)
        ax2.scatter(x[-1], y[-1], color=c, marker=end_marker, s=130, edgecolor="k", zorder=5)

    ax.set_xlim(0, args.domain[0])
    ax.set_ylim(0, args.domain[1])
    ax.set_zlim(args.domain[2], 0)  # z positive downward -> invert so surface is on top
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("depth [m]")
    ax.set_title("3D view")

    ax2.set_xlim(0, args.domain[0])
    ax2.set_ylim(0, args.domain[1])
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Top-down view (from surface)")
    ax2.legend(loc="upper left")

    kind = "IPPO" if is_multi else ("single-agent DQN" if is_dqn else "single-agent PPO")
    # PPO/IPPO checkpoints log "iteration"; DQN logs "global_step".
    progress = (f"iter {ckpt['iteration']}" if "iteration" in ckpt
                else f"step {ckpt.get('global_step', '?')}")
    fig.suptitle(f"{kind} agent trajectories  "
                 f"({'stochastic' if cli.stochastic else 'greedy'} policy, "
                 f"{progress}, start={start_mode})\n"
                 "o = start   * = reached target   X = did not   shaded = success zone")
    plt.tight_layout()

    if cli.save:
        plt.savefig(cli.save, dpi=150)
        print(f"Saved figure to {cli.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

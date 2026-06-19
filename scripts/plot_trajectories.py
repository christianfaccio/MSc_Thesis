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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--seed", type=int, default=None, help="episode reset seed (default: random)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy instead of greedy argmax")
    p.add_argument("--synthetic", action="store_true",
                   help="override the run's NetCDF data and use the synthetic salinity field")
    p.add_argument("--start", choices=["center", "random"], default="center",
                   help="agent spawn: 'center' of the domain (default) or the env's 'random' spawn")
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
    )
    if is_multi:
        return MultiAgentEnv(n_agents=args.n_agents, **common)
    return SingleAgentEnv(**common)


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

    def reset(seed):
        obs, _ = env.reset(seed=seed)
        if cli.start == "center":
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
    obs = reset(cli.seed)
    trajectories = [[env.sim.agents[i].pos.copy()] for i in range(n_agents)]
    done = np.zeros(n_agents, dtype=bool)
    success = np.zeros(n_agents, dtype=bool)
    steps = 0

    while not done.all() and steps < args.max_steps:
        with torch.no_grad():
            x = torch.tensor(normalize(obs)).to(device)
            actions = act(x).cpu().numpy()

        obs, terminateds, truncateds = step(actions)
        for i in range(n_agents):
            trajectories[i].append(env.sim.agents[i].pos.copy())
        done = done | terminateds | truncateds
        success = success | terminateds
        steps += 1

    print(f"Episode finished after {steps} env steps "
          f"({steps * args.dt * args.frame_skip:.0f} s of sim time).")
    for i in range(n_agents):
        print(f"  agent A{i+1:02d}: {'REACHED TARGET' if success[i] else 'did not reach target'} "
              f"({len(trajectories[i])} waypoints)")

    # --- plot the trajectories: 3D view + top-down (surface) view ---
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)  # top-down x-y, marginalized over depth
    colors = plt.cm.tab10(np.arange(n_agents) % 10)

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
                 f"{progress}, start={cli.start})\n"
                 "o = start   * = reached target   X = did not")
    plt.tight_layout()

    if cli.save:
        plt.savefig(cli.save, dpi=150)
        print(f"Saved figure to {cli.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

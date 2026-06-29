'''
Classical implementation of the DQN algorithm, modified starting from
the implementation of CleanRL and aligned with the structure of
`src/single_agent/ppo.py` (env wrappers, checkpointing, rich UI logging).

Usage (from root):
    - in one terminal start the training    -> `python -m src.single_agent.dqn`
    - in another one start tensorboard      -> `tensorboard --logdir runs --port 6006`

Pseudocode:
```
1.  Initialize Q-network and target network (copy of Q)
2.  for global_step do:
3.      Observe state s_t
4.      With prob epsilon pick random action, else a_t = argmax_a Q(s_t, a)
5.      Apply action, observe reward r_t and next state s_{t+1}, store transition in replay buffer
6.      if global_step > learning_starts and global_step % train_frequency == 0:
7.          Sample minibatch from buffer
8.          y = r + gamma * max_a' Q_target(s', a') * (1 - done)
9.          loss = MSE(Q(s, a), y); gradient step
10.     Every target_network_frequency steps: soft-update target <- tau*Q + (1-tau)*target
```
'''
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.tensorboard import SummaryWriter

from src.single_agent.policy import QNetwork
from src.envs.single_agent import SingleAgentEnv

DEBUG = True
console = Console()
STATS_WINDOW = 100


@dataclass
class Args:
    exp_name: str = "dqn"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Environment arguments
    env_id: str = "SingleAgent-dqn"
    """the id of the environment"""
    xml_file: str = "config/simulation.xml"
    """SwarmSwIM simulation XML"""
    netcdf_file: str = "data/oceananigans/"
    """optional Oceananigans NetCDF data: single file, glob pattern (quote it in the
    shell, e.g. --netcdf-file 'data/oceananigans/hydrostatic_winter_run*.nc'), or
    directory; a random file + snapshot is sampled each episode reset, and currents
    and salinity come from the data instead of the synthetic models"""
    n_sources: int = 4
    """number of pollution sources spawned each reset"""
    k: int = 12
    """history buffer length for (action, reward) pairs; 12 steps × 10 s = 120 s of context"""
    v_agent: float = 1.0
    """agent commanded speed (m/s)"""
    max_steps: int = 720
    """maximum env steps per episode before truncation"""
    dt: float = 1.0
    """simulator timestep (s) per env step"""
    frame_skip: int = 10
    """sim sub-steps per env step (action repeated); 1 disables frame skip.
    One env step is dt · frame_skip = 10 s of sim time; distance per step ≈
    v_agent · dt · frame_skip = 10 m, so max_steps=720 covers ~7.2 km (the 5 km
    domain) and 7200 s — a 2 h battery, spanning ~9 NetCDF snapshots at ~900 s.
    Raising frame_skip lengthens each action's sim time (more sim.tick() calls
    ⇒ more compute per step)."""
    domain: tuple[float, float, float] = (5000.0, 5000.0, 40.0)
    """domain extent in (x, y, z) meters"""
    sigma_h: float = 500.0
    """salinity plume horizontal std [m] — scale with the domain"""
    sigma_v: float = 12.0
    """salinity plume vertical std [m]"""
    eddy_length_scale: float = 1000.0
    """vortex eddy radius [m] — scale with the domain"""
    static_frame: bool = True
    """NetCDF: freeze one random snapshot per episode (no intra-episode time evolution)"""
    target_mode: str = "tail"
    """target selection: "tail" = rare LOW-salinity target with the agent spawned on the
    land strip off the west/south borders (small success zone, far from spawn, must
    navigate); "random" = legacy behaviour"""
    target_percentile: float = 5.0
    """tail width (%) for target_mode="tail" """
    target_samples: int = 1500
    """field samples used to estimate the value distribution each reset"""
    land_clearance: float = 500.0
    """spawn distance (m) off the west/south land borders for target_mode="tail" """

    # Algorithm specific arguments
    total_timesteps: int = 2000000
    """total timesteps of the experiments"""
    learning_rate: float = 3.0e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.995
    """the discount factor gamma; effective horizon 1/(1-γ) = 200 steps ≈ 2000 s,
    matched to 720-step (7200 s) episodes"""
    tau: float = 0.005
    """Polyak target-network update rate, applied every step (1.0 = hard copy)"""
    max_grad_norm: float = 10.0
    """max global norm for gradient clipping"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    # Checkpointing
    save_model: bool = True
    """if toggled, periodically save model + optimizer + RNG state checkpoints"""
    save_every_steps: int = 50000
    """save a checkpoint every N global steps (and always on the final step)"""
    checkpoint_dir: str = "runs"
    """parent directory for checkpoints; full path is <checkpoint_dir>/<run_name>/checkpoints/"""
    resume: str = None
    """path to a checkpoint .pt file to resume training from"""


def make_env(args):
    def thunk():
        env = SingleAgentEnv(
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
            gamma=args.gamma,
            static_frame=args.static_frame,
            target_mode=args.target_mode,
            target_percentile=args.target_percentile,
            target_samples=args.target_samples,
            land_clearance=args.land_clearance,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Running observation normalization (Andrychowicz et al. 2021, §3.3)
        # followed by ±10 clipping as a safety net against simulator outliers.
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space
        )
        # NOTE: no reward normalization for DQN. NormalizeReward rescales rewards by a
        # running return std, which would distort the bootstrapped TD targets the
        # Q-network regresses onto. Raw rewards keep Q-values on a meaningful scale.
        return env

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@dataclass
class ReplaySample:
    """A sampled minibatch of transitions, all tensors on the training device."""
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor       # shape (B, 1), long
    rewards: torch.Tensor       # shape (B, 1)
    dones: torch.Tensor         # shape (B, 1)


class ReplayBuffer:
    """Minimal circular replay buffer for a single (non-vectorised) Discrete-action env.

    Self-contained (numpy-backed) so it works with gymnasium spaces, unlike the
    SB3-backed `cleanrl_utils.buffers.ReplayBuffer` which is pinned to legacy gym.
    """

    def __init__(self, buffer_size, observation_space, action_space, device):
        self.buffer_size = int(buffer_size)
        self.device = device
        obs_shape = observation_space.shape
        self.observations = np.zeros((self.buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        # Inputs are batched over envs (num_envs == 1 here); take the single env slot.
        self.observations[self.pos] = np.asarray(obs[0], dtype=np.float32)
        self.next_observations[self.pos] = np.asarray(next_obs[0], dtype=np.float32)
        self.actions[self.pos] = np.asarray(action[0], dtype=np.int64)
        self.rewards[self.pos] = np.asarray(reward[0], dtype=np.float32)
        self.dones[self.pos] = float(done[0])
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper = self.buffer_size if self.full else self.pos
        idx = np.random.randint(0, upper, size=batch_size)
        to_t = lambda a: torch.as_tensor(a[idx], device=self.device)
        return ReplaySample(
            observations=to_t(self.observations),
            next_observations=to_t(self.next_observations),
            actions=to_t(self.actions),
            rewards=to_t(self.rewards),
            dones=to_t(self.dones),
        )


def _find_norm_obs(env):
    while hasattr(env, "env"):
        if isinstance(env, gym.wrappers.NormalizeObservation):
            return env
        env = env.env
    return None


def get_obs_rms_state(envs):
    states = []
    for env in envs.envs:
        norm = _find_norm_obs(env)
        if norm is None:
            continue
        states.append({
            "mean": norm.obs_rms.mean.copy(),
            "var": norm.obs_rms.var.copy(),
            "count": float(norm.obs_rms.count),
        })
    return states


def set_obs_rms_state(envs, states):
    for env, state in zip(envs.envs, states):
        norm = _find_norm_obs(env)
        if norm is None:
            continue
        norm.obs_rms.mean = state["mean"].copy()
        norm.obs_rms.var = state["var"].copy()
        norm.obs_rms.count = state["count"]


def save_checkpoint(args, run_name, global_step, q_network, target_network, optimizer, envs):
    ckpt_dir = Path(args.checkpoint_dir) / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{global_step:08d}.pt"
    torch.save({
        "global_step": global_step,
        "model_state_dict": q_network.state_dict(),
        "target_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
        "obs_rms": get_obs_rms_state(envs),
        "args": vars(args),
    }, ckpt_path)
    latest = ckpt_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_path.name)
    console.log(f"Saved checkpoint: {ckpt_path}")


def train(args):
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if DEBUG:
        print("--- INFO ---\n")
        print(f"Run name: {run_name}\nBuffer size: {args.buffer_size}\nBatch size: {args.batch_size}\n")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if DEBUG:
        print(f"Device: {device}")

    # env setup
    if DEBUG:
        print("--- Setting up the environment...")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args) for _ in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )

    # Resume from checkpoint if requested
    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        q_network.load_state_dict(ckpt["model_state_dict"])
        target_network.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["global_step"] + 1
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["np_rng"])
        random.setstate(ckpt["py_rng"])
        if "obs_rms" in ckpt:
            set_obs_rms_state(envs, ckpt["obs_rms"])
        if DEBUG:
            print(f"Resumed from {args.resume}: global_step={start_step}")

    # TRY NOT TO MODIFY: start the game
    if DEBUG:
        print("--- GAME START ---")
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    # Rolling stats over the last STATS_WINDOW finished episodes
    ep_returns = deque(maxlen=STATS_WINDOW)
    ep_lengths = deque(maxlen=STATS_WINDOW)
    ep_terminated = deque(maxlen=STATS_WINDOW)  # 1.0 if reached target, 0.0 if truncated

    progress = Progress(
        TextColumn("[bold blue]step"),
        BarColumn(),
        TextColumn(
            "ret={task.fields[ret]:>6.2f}  len={task.fields[len]:>5.1f}  "
            "term={task.fields[term]:>3.0f}%  eps={task.fields[eps]:>4d}  "
            "ε={task.fields[eg]:>4.2f}  SPS={task.fields[sps]:>5d}"
        ),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )
    task_id = progress.add_task(
        "train",
        total=args.total_timesteps,
        completed=start_step,
        ret=float("nan"),
        len=float("nan"),
        term=0.0,
        eps=0,
        eg=1.0,
        sps=0,
    )

    progress.start()
    for global_step in range(start_step, args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Episode-end logging — handle both Gymnasium APIs (mirror ppo.py)
        if "episode" in infos:
            # Gymnasium 1.x: infos["episode"]["r"]/["l"] are arrays, infos["_episode"] is the mask
            ep_r = infos["episode"]["r"]
            ep_l = infos["episode"]["l"]
            mask = infos.get("_episode", [True] * len(ep_r))
            for i, finished in enumerate(mask):
                if finished:
                    r_val = float(ep_r[i])
                    l_val = float(ep_l[i])
                    ep_returns.append(r_val)
                    ep_lengths.append(l_val)
                    succ = 1.0 if bool(terminations[i]) else 0.0
                    ep_terminated.append(succ)
                    writer.add_scalar("charts/episodic_return", r_val, global_step)
                    writer.add_scalar("charts/episodic_length", l_val, global_step)
                    writer.add_scalar("charts/episode_success", succ, global_step)
        elif "final_info" in infos:
            # Older Gymnasium API (≤ 0.29)
            for i, info in enumerate(infos["final_info"]):
                if info and "episode" in info:
                    r_val = float(info["episode"]["r"])
                    l_val = float(info["episode"]["l"])
                    ep_returns.append(r_val)
                    ep_lengths.append(l_val)
                    succ = 1.0 if bool(terminations[i]) else 0.0
                    ep_terminated.append(succ)
                    writer.add_scalar("charts/episodic_return", r_val, global_step)
                    writer.add_scalar("charts/episodic_length", l_val, global_step)
                    writer.add_scalar("charts/episode_success", succ, global_step)

        # Save to replay buffer; recover the true terminal obs on truncation.
        # Gymnasium exposes it under "final_observation" (vector API ≤ 0.29) or
        # "final_obs" (1.x). If neither is present, fall back to the (reset) next_obs.
        real_next_obs = next_obs.copy()
        final_obs = infos.get("final_observation", infos.get("final_obs"))
        if final_obs is not None:
            for idx, trunc in enumerate(truncations):
                if trunc and final_obs[idx] is not None:
                    real_next_obs[idx] = final_obs[idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    # Double DQN: select the next action with the online net, evaluate
                    # it with the target net. Decouples selection from evaluation to
                    # curb the max-operator overestimation bias that compounds at high γ.
                    next_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
                    target_max = target_network(data.next_observations).gather(1, next_actions).squeeze(1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                # Huber (smooth-L1) loss bounds the gradient of large TD errors, which
                # MSE squared and let diverge given the dense + spiky (+10 bonus) rewards.
                loss = F.smooth_l1_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

            # Soft (Polyak) target update every step: target ← τ·online + (1-τ)·target.
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )

        # Periodic scalar + UI updates
        sps = int((global_step - start_step + 1) / (time.time() - start_time))
        if global_step % 100 == 0:
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            if ep_terminated:
                writer.add_scalar("charts/success_rate", float(np.mean(ep_terminated)), global_step)

        progress.update(
            task_id,
            completed=global_step + 1,
            ret=(float(np.mean(ep_returns)) if ep_returns else float("nan")),
            len=(float(np.mean(ep_lengths)) if ep_lengths else float("nan")),
            term=(100.0 * float(np.mean(ep_terminated)) if ep_terminated else 0.0),
            eps=len(ep_returns),
            eg=epsilon,
            sps=sps,
        )

        # Checkpoint save
        if args.save_model and (
            global_step % args.save_every_steps == 0 or global_step == args.total_timesteps - 1
        ):
            save_checkpoint(args, run_name, global_step, q_network, target_network, optimizer, envs)

    progress.stop()
    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)

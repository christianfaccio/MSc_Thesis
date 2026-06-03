'''
Classical implementation of the PPO algorithm, modified starting from
the implementation of CleanRL.

Usage (from root):
    - in one terminal start the training    -> `python -m src.single_agent.ppo`
    - in another one start tensorboard      -> `tensorboard --logdir runs --port 6006`
    (opt) look at it in the web             -> `ssh -L 6006:localhost:6006 username@<ip>`

Pseudocode:
```
1.  Initialize actor and critic networks 
2.  for episode do:
3.      for step do:
4.          Observe current state s_t
5.          Sample action a_t from policy
6.          Apply action, observe reward r_t and next state s_{t+1}
7.          old_policy <- current_policy
8.          for epoch do:
9.              compute importance sampling weights (rho)
10.             if s_{t+1} terminal:
11.                 Adv(s,a) <- r_t - V(s_t)
12.                 y_t <- r_t
13.             else:
14.                 Adv(s,a) <- r_t + disc * V(s_{t+1}) - V(s_t)
15.                 y_t <- r_t + disc * V(s_{t+1})
16.             Compute actor and critic losses
17.             Update networks' params
```
'''
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.tensorboard import SummaryWriter
from src.single_agent.policy import CustomPolicy
from src.envs.single_agent import SingleAgentEnv

from dataclasses import dataclass

DEBUG=True
console = Console()
STATS_WINDOW = 100


@dataclass
class Args:
    exp_name: str = "ppo"
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
    env_id: str = "SingleAgent-v0"
    """the id of the environment"""
    xml_file: str = "config/simulation.xml"
    """SwarmSwIM simulation XML"""
    n_sources: int = 4
    """number of pollution sources spawned each reset"""
    k: int = 4
    """history buffer length for (action, reward) pairs"""
    v_agent: float = 1.0
    """agent commanded speed (m/s)"""
    max_steps: int = 256
    """maximum env steps per episode before truncation"""
    dt: float = 0.5
    """simulator timestep (s) per env step"""
    frame_skip: int = 60
    """sim sub-steps per env step (action repeated); 1 disables frame skip.
    Distance per env step ≈ v_agent · dt · frame_skip ≈ 1·0.5·60 = 30 m, so
    max_steps=256 covers ~7.7 km (the 5 km domain). Raising frame_skip lengthens
    each action's sim time (more sim.tick() calls ⇒ more compute per step)."""
    domain: tuple[float, float, float] = (5000.0, 5000.0, 40.0)
    """domain extent in (x, y, z) meters"""
    sigma_h: float = 500.0
    """salinity plume horizontal std [m] — scale with the domain"""
    sigma_v: float = 12.0
    """salinity plume vertical std [m]"""
    eddy_length_scale: float = 1000.0
    """vortex eddy radius [m] — scale with the domain"""

    # Algorithm specific arguments
    # NOTE: default values are taken from Andrychowicz et. al
    total_timesteps: int = 1000000              # NOTE: default value 1M or 2M
    """total timesteps of the experiments"""
    learning_rate: float = 3.0e-4               # NOTE: default 0.0003
    """the learning rate of the optimizer"""
    num_envs: int = 6                           # NOTE: default value should be 256 but it needs to be tuned regarding the hardware at disposal,
                                                # since they should run in parallel in CPU cores
    """the number of parallel game environments"""
    num_steps: int = 256                    
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.9                     # NOTE: default value 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 12                   # NOTE: paper is permissive here, default 12
    """the number of mini-batches"""
    update_epochs: int = 10                     # NOTE: number of epochs, default value 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.25                     # NOTE: default value is 0.25, but one should try different values
                                                # as in some cases lower (0.1) or bigger (0.5) values help getting better performances
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False                    # NOTE: default False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Checkpointing
    save_model: bool = True
    """if toggled, periodically save model + optimizer + RNG state checkpoints"""
    save_every_iterations: int = 20
    """save a checkpoint every N PPO iterations (and always on the final iteration)"""
    checkpoint_dir: str = "runs"
    """parent directory for checkpoints; full path is <checkpoint_dir>/<run_name>/checkpoints/"""
    resume: str = None
    """path to a checkpoint .pt file to resume training from"""

    # to be filled in runtime
    batch_size: int = 0                         
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0                   
    """the number of iterations (computed in runtime)"""

def make_env(args):
    def thunk():
        env = SingleAgentEnv(
            xml_file=args.xml_file,
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
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Running observation normalization (Andrychowicz et al. 2021, §3.3)
        # followed by ±10 clipping as a safety net against simulator outliers.
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10.0, 10.0), env.observation_space
        )
        # Running reward normalization (Andrychowicz et al. 2021, §3.3, C66).
        # Placed AFTER RecordEpisodeStatistics so logged episodic returns stay raw.
        # ±10 clip mitigates cold-start blow-up when return_rms.var is still tiny.
        env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10.0, 10.0)))
        return env

    return thunk


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


def _find_norm_reward(env):
    while hasattr(env, "env"):
        if isinstance(env, gym.wrappers.NormalizeReward):
            return env
        env = env.env
    return None


def get_return_rms_state(envs):
    states = []
    for env in envs.envs:
        norm = _find_norm_reward(env)
        if norm is None:
            continue
        states.append({
            "mean": float(norm.return_rms.mean),
            "var": float(norm.return_rms.var),
            "count": float(norm.return_rms.count),
        })
    return states


def set_return_rms_state(envs, states):
    for env, state in zip(envs.envs, states):
        norm = _find_norm_reward(env)
        if norm is None:
            continue
        norm.return_rms.mean = np.array(state["mean"])
        norm.return_rms.var = np.array(state["var"])
        norm.return_rms.count = state["count"]

def train(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if DEBUG:
        print("--- INFO ---\n")
        print(f"Run name: {run_name}\nBatch size: {args.batch_size}\nMinibatch size: {args.minibatch_size}\nEpisodes: {args.num_iterations}\n")
    
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
        print(f"--- Setting up the environment...")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args) for _ in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = CustomPolicy(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Resume from checkpoint if requested
    start_iteration = 1
    global_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt["iteration"] + 1
        global_step = ckpt["global_step"]
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["np_rng"])
        random.setstate(ckpt["py_rng"])
        if "obs_rms" in ckpt:
            set_obs_rms_state(envs, ckpt["obs_rms"])
        if "return_rms" in ckpt:
            set_return_rms_state(envs, ckpt["return_rms"])
        if DEBUG:
            print(f"Resumed from {args.resume}: iteration={start_iteration}, global_step={global_step}")

    # ALGO Logic: Storage setup
    # Values are recomputed each epoch (Andrychowicz et al. 2021, §3.5) so they are not stored here.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    if DEBUG:
        print("--- GAME START ---")
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Rolling stats over the last STATS_WINDOW finished episodes
    ep_returns = deque(maxlen=STATS_WINDOW)
    ep_lengths = deque(maxlen=STATS_WINDOW)
    ep_terminated = deque(maxlen=STATS_WINDOW)  # 1.0 if reached target, 0.0 if truncated

    progress = Progress(
        TextColumn("[bold blue]iter"),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn(
            "ret={task.fields[ret]:>6.2f}  len={task.fields[len]:>5.1f}  "
            "term={task.fields[term]:>3.0f}%  eps={task.fields[eps]:>4d}  "
            "SPS={task.fields[sps]:>5d}"
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
        total=args.num_iterations,
        completed=start_iteration - 1,
        ret=float("nan"),
        len=float("nan"),
        term=0.0,
        eps=0,
        sps=0,
    )

    progress.start()
    for iteration in range(start_iteration, args.num_iterations + 1): # NOTE: cycle through iterations (episodes) (outer loop)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):   # NOTE: cycle through steps in single episode (inner loop)
            global_step += args.num_envs        # increase global counter
            obs[step] = next_obs                # save the obs
            dones[step] = next_done             # save termination condition

            # ALGO LOGIC: sample action
            with torch.no_grad():
                action, logprob, _, _ = agent.get_action_and_value(next_obs)    # NOTE: calls the policy with the obs vector
            actions[step] = action                  # record the action
            logprobs[step] = logprob                # record the logprobs

            # TRY NOT TO MODIFY: apply action, observe reward and next state, save
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())    # NOTE: run the single step
            next_done = np.logical_or(terminations, truncations)                                    # termination logic
            rewards[step] = torch.tensor(reward).to(device).view(-1)                                # record the reward
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device) # updates state

            # Episode-end logging — handle both Gymnasium APIs
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
                        ep_terminated.append(1.0 if bool(terminations[i]) else 0.0)
                        writer.add_scalar("charts/episodic_return", r_val, global_step)
                        writer.add_scalar("charts/episodic_length", l_val, global_step)
            elif "final_info" in infos:
                # Older Gymnasium API (≤ 0.29)
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        r_val = float(info["episode"]["r"])
                        l_val = float(info["episode"]["l"])
                        ep_returns.append(r_val)
                        ep_lengths.append(l_val)
                        ep_terminated.append(1.0 if bool(terminations[i]) else 0.0)
                        writer.add_scalar("charts/episodic_return", r_val, global_step)
                        writer.add_scalar("charts/episodic_length", l_val, global_step)

        # NOTE: here the steps are ended, but we are still in a single iteration, so we need to update the policy

        # flatten the batch (advantages, returns, and values are recomputed each epoch below)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # NOTE: cycle through epochs — recompute advantages with the current critic each pass
        # (Andrychowicz et al. 2021, §3.5: "recompute advantages once per data pass")
        for epoch in range(args.update_epochs):
            # Recompute values, advantages, and returns with the current critic, then GAE.
            with torch.no_grad():
                new_values = agent.get_value(b_obs).view(args.num_steps, args.num_envs)
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = new_values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - new_values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + new_values
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_values = new_values.reshape(-1)

            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        # Live UI update
        progress.update(
            task_id,
            completed=iteration,
            ret=(float(np.mean(ep_returns)) if ep_returns else float("nan")),
            len=(float(np.mean(ep_lengths)) if ep_lengths else float("nan")),
            term=(100.0 * float(np.mean(ep_terminated)) if ep_terminated else 0.0),
            eps=len(ep_returns),
            sps=sps,
        )

        # Checkpoint save
        if args.save_model and (iteration % args.save_every_iterations == 0 or iteration == args.num_iterations):
            ckpt_dir = Path(args.checkpoint_dir) / run_name / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"iter_{iteration:04d}.pt"
            torch.save({
                "iteration": iteration,
                "global_step": global_step,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "np_rng": np.random.get_state(),
                "py_rng": random.getstate(),
                "obs_rms": get_obs_rms_state(envs),
                "return_rms": get_return_rms_state(envs),
                "args": vars(args),
            }, ckpt_path)
            latest = ckpt_dir / "latest.pt"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
            console.log(f"Saved checkpoint: {ckpt_path}")

    progress.stop()
    envs.close()
    writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)

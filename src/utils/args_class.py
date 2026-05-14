from dataclasses import dataclass, field

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
    max_steps: int = 1024
    """maximum env steps per episode before truncation"""
    dt: float = 0.1
    """simulator timestep (s) per env step"""
    domain: tuple[float, float, float] = (50.0, 50.0, 50.0)
    """domain extent in (x, y, z) meters"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 6   # NOTE: match the number of CPU cores?
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
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

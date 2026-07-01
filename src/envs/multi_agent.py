from .base import BaseEnv
import glob
import os
import warnings
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SwarmSwIM import Simulator, sim_functions, Agent
from ..single_agent.reward import reward_func
import itertools
from src.models.salinity import compute_salinity_analytical
from src.models.turbidity import compute_turbidity
from src.utils.sources import random_sources
from src.utils.spawn import spawn_positions_along_land_strip


def _resolve_nc_files(spec) -> list:
    '''Resolve a NetCDF spec — single path, glob pattern, directory, or a
    list/tuple of those — to a sorted list of file paths. Empty list for None.'''
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        files = []
        for item in spec:
            files.extend(_resolve_nc_files(item))
        return sorted(set(files))
    p = str(spec)
    if os.path.isdir(p):
        files = sorted(str(f) for f in Path(p).glob("*.nc"))
    elif any(ch in p for ch in "*?["):
        files = sorted(glob.glob(p))
    elif os.path.isfile(p):
        files = [p]
    else:
        raise ValueError(f"NetCDF file not found: {p}")
    if not files:
        raise ValueError(f"No NetCDF files matched: {p}")
    return files


# TODO: implement the BaseEnv and inherit from that instead of gym.Env
class MultiAgentEnv(gym.Env):
    '''
    Multi-agent SwarmSwIM environment wrapped for (I/MA)PPO training.

    The API is the PettingZoo-parallel convention flattened over homogeneous
    agents (every per-agent quantity is the leading axis of a numpy array):

        reset() -> obs (N, local_dim), info
        step(actions (N,)) -> obs (N, local_dim), rewards (N,),
                              terminateds (N,), truncateds (N,), info

    `info["global_state"]` carries the (11*N + 2,) centralized state used by the
    MAPPO critic; IPPO ignores it (its critic uses the local obs only).

    Local observation per agent (4k + 6) — pure local-sensor + mission info,
    no absolute position:
        [ (action, potential) history (2k) | (S, tau) history (2k) |
          u v w (body-frame currents) | S* tau* | depth ]
    The (S, tau) history is a (k, 2) sliding window of the measured salinity and
    turbidity (last row = current), so the present reading is always available
    plus the previous k-1. No salinity gradient is provided: the agent must infer
    direction from the history of what it measured after each action (finite
    differences / dead reckoning, frame-stacked like the action history).
    Heading psi is tracked internally (to rotate currents into the body frame)
    but never exposed to the actor.

    Global state ((2k + 6)*N + 2) — centralized, used only by the MAPPO critic:
        [ S* tau* | per agent: u v w  (S, tau) history (2k)  x y z ]
    Depth is NOT a separate feature here: it is the z component of (x, y, z).

    Each agent's reward is potential-based shaping (Ng et al. 1999):
    r_i = success_bonus + γΦ(s'_i) − Φ(s_i), with Φ = reward_func(...) and Φ
    forced to 0 at a success terminal (truncation keeps the real Φ so the agent
    bootstraps). γ MUST match the trainer's discount for the shaping to stay
    policy-invariant.

    Episode dynamics:
        - Each agent succeeds (terminates, +success_bonus, latched) after staying
          in the target zone for 3 consecutive steps.
        - The episode (all envs) truncates at max_steps.
        - The env signals "needs reset" when every agent is done
          (terminated or truncated); the training loop does the reset.

    Parameters:
        - xml_file -> SwarmSwIM simulation .xml (environment physics only; agents
          are created programmatically, any <agents> block in the XML is ignored)
        - netcdf_file -> optional Oceananigans NetCDF data (single file, glob,
          directory, or list); a random file + time window is sampled each reset
        - n_agents -> number of agents
        - n_sources -> pollution sources spawned each reset (synthetic mode)
        - k -> history buffer length for (action, reward) pairs
        - v_agent -> agent commanded speed (m/s)
        - max_steps -> env steps per episode before truncation
        - dt -> simulator timestep (s) per sub-step
        - frame_skip -> sim sub-steps per env step (action held constant)
        - domain -> (x, y, z) extent of the domain in meters
    '''
    def __init__(self,
                 xml_file: str,
                 netcdf_file: str = None,
                 n_agents: int = 2,
                 n_sources: int = 4,
                 k: int = 4,                    # history length of (action, reward)
                 v_agent: float = 1.0,          # agent speed in m/s
                 z_scale: float = 1.0,          # vertical (heave) speed multiplier; <1 gives finer depth control
                 max_steps: int = 128,          # steps of an episode before truncation
                 dt: float = 0.1,               # seconds per sub-step
                 frame_skip: int = 10,          # sim sub-steps per env step (action constant)
                 domain=(5000.0, 5000.0, 40.0),  # domain size (0-x, 0-y, 0-z) in meters
                 sigma_h: float = 500.0,        # salinity plume horizontal std [m]
                 sigma_v: float = 12.0,         # salinity plume vertical std [m]
                 eddy_length_scale: float = 1000.0,  # vortex eddy radius [m]
                 gamma: float = 0.99,           # RL discount; MUST match the trainer's γ for shaping invariance
                 success_bonus: float = 10.0,   # sparse reward on reaching the target
                 reward_mode: str = "shaped",   # "shaped" (potential-based) | "sparse" (-1/step, +success_bonus to ALL on first success)
                 target_mode: str = "random",   # "random" (legacy) | "tail" (rare LOW-salinity target + land-strip spawn)
                 target_percentile: float = 5.0,  # tail width (%) for target_mode="tail"
                 target_samples: int = 1500,    # field samples used to estimate the value distribution
                 static_frame: bool = True,     # NetCDF: freeze one random snapshot per episode (no time evolution)
                 land_clearance: float = 500.0,  # spawn distance (m) off the west/south land borders
                 end_on_any_success: bool = True,  # end the episode as soon as ANY agent reaches the zone
                 ):
        super().__init__()

        self.sim_xml = xml_file
        self.netcdf_file = netcdf_file
        self._nc_files = _resolve_nc_files(netcdf_file)
        self.n_agents = n_agents
        self.n_sources = n_sources
        self.k = k
        self.v = v_agent
        self.z_scale = z_scale
        self.max_steps = max_steps
        self.dt = dt
        self.frame_skip = frame_skip
        self.domain = domain
        self.sigma_h = sigma_h
        self.sigma_v = sigma_v
        self.eddy_length_scale = eddy_length_scale
        self.gamma = gamma
        self.success_bonus = success_bonus
        self.reward_mode = reward_mode
        self.target_mode = target_mode
        self.target_percentile = target_percentile
        self.target_samples = target_samples
        self.static_frame = static_frame
        self.land_clearance = land_clearance
        self.end_on_any_success = end_on_any_success
        # Per-agent potential Φ(s) of the previous state; set on reset and updated
        # each step. Each agent's reward is the sparse success bonus plus the
        # potential-based shaping term γΦ(s') − Φ(s) (Ng et al. 1999), which is
        # policy-invariant and avoids the loitering incentive of a raw dense reward.
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)

        self.target_salinity = 0.0
        self.target_turbidity = 0.0

        self.epsilon_salinity = 0.05
        # ε_τ widened 0.01 → 0.03 (2026-07-01): at 0.01 the depth success band was
        # exactly one 1 m z-step wide (dτ/dz≈0.01·1 m step), so a discrete ±1 z-action
        # could not move and stay inside → bang-bang z ping-pong. 0.03 gives a ±3 m band
        # (3 steps) the agent can rest in. Depth-only, so domain coverage barely changes.
        self.epsilon_turbidity = 0.03
        self._success_steps_required = 1

        # Per-agent episode state (allocated for real in reset())
        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)
        self.histories = np.zeros((self.n_agents, self.k, 2), dtype=np.float32)
        self.st_histories = np.zeros((self.n_agents, self.k, 2), dtype=np.float32)
        self.t_step = 0

        # Action space (per agent): 27 discrete moves, relative body frame
        self.action_space = gym.spaces.Discrete(27)

        # Local observation (actor input): 2k (action,potential) history
        # + 2k (S, τ) history + 3 currents + 2 (S*, τ*) + 1 depth
        local_obs_dim = 4 * k + 6
        self.local_observation_space = spaces.Box(-np.inf, np.inf, shape=(local_obs_dim,), dtype=np.float32)

        # Global state (MAPPO critic input): targets (2) + (2k + 6) per agent.
        # Depth is the z of (x, y, z), so it is NOT a separate feature here.
        global_obs_dim = (2 * k + 6) * self.n_agents + 2
        self.global_observation_space = spaces.Box(-np.inf, np.inf, shape=(global_obs_dim,), dtype=np.float32)

        # action scalar -> [dx, dy, dz] normalized
        self._action_to_direction = self._build_action_table()

        self.sim = None
        # NetCDF ocean-data loaders, one per file, created lazily and reused.
        self._loaders = {}
        self.active_netcdf_path = None
        self._warned_short_record = False

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Per-agent episode state
        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)
        self.histories = np.zeros((self.n_agents, self.k, 2), dtype=np.float32)
        self.st_histories = np.zeros((self.n_agents, self.k, 2), dtype=np.float32)
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)
        self.t_step = 0

        # Create the simulator (environment physics). A random NetCDF file is
        # drawn each episode; its FieldLoader is cached and reused.
        loader = None
        path = None
        if self._nc_files:
            from SwarmSwIM.ocean_data import FieldLoader
            path = self._nc_files[int(self.np_random.integers(len(self._nc_files)))]
            if path not in self._loaders:
                self._loaders[path] = FieldLoader(path)
            loader = self._loaders[path]
            self.active_netcdf_path = path
        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.sim_xml, netcdf_file=loader)

        # Drop any agents auto-loaded from the XML <agents> block — we create our
        # own below, so the simulation XML only needs to define environment physics.
        self.sim.agents.clear()
        self.sim.history.clear()

        # Create agents with random position + heading
        for i in range(self.n_agents):
            agent = Agent(
                name=f"A{i + 1:02d}",
                Dt=self.dt,
                initialPosition=np.array([
                    self.np_random.uniform(0.0, self.domain[0]),
                    self.np_random.uniform(0.0, self.domain[1]),
                    self.np_random.uniform(0.0, self.domain[2]),
                ]),
                initialHeading=self.np_random.uniform(-180.0, 180.0),
                agent_xml="config/agent.xml",
                rng=int(self.np_random.integers(2 ** 31)),
            )
            self.sim.add(agent)

        # Environment variability
        if self._nc_files:
            if self.static_frame:
                # Static-frame mode: freeze the fields at one randomly chosen
                # snapshot for the whole episode (no intra-episode time evolution).
                # Episode variability comes from the file choice plus this index.
                idx = int(self.np_random.integers(len(loader.times)))
                loader.set_snapshot(idx)
            else:
                # Dynamic mode: random time window; fields interpolated in time as
                # the episode advances (the Simulator passes sim_time each tick).
                episode_seconds = self.max_steps * self.dt * self.frame_skip
                max_start = loader.max_window_start(episode_seconds)
                if (loader.times[max_start] + episode_seconds > loader.times[-1]
                        and not self._warned_short_record):
                    warnings.warn(
                        f"{path}: data record shorter than episode "
                        f"({episode_seconds:.0f}s); fields will freeze at the last snapshot.")
                    self._warned_short_record = True
                start = int(self.np_random.integers(max_start + 1))
                loader.set_window(start)
            salinity_at = loader.salinity_at
        else:
            # Randomize sources (env's seeded PRNG); bounds scale with the domain.
            self.sources = random_sources(
                rng=self.np_random, n_sources=self.n_sources,
                min_x=0.0, max_x=self.domain[0],
                min_y=0.0, max_y=self.domain[1],
                min_depth=0.0, max_depth=self.domain[2],
            )

            def salinity_at(x, y, z):
                return compute_salinity_analytical(
                    x, y, z, self.sources, sigma_h=self.sigma_h, sigma_v=self.sigma_v)

            # Randomize currents (5 surface components + Ekman 3D transform)
            # 1. Uniform background (tidal / geostrophic drift)
            bg_speed = self.np_random.uniform(0.0, 0.3)
            bg_angle = self.np_random.uniform(0.0, 2 * np.pi)
            self.sim.environment['uniform_current'] = np.array([
                bg_speed * np.cos(bg_angle),
                bg_speed * np.sin(bg_angle),
                0.0,
            ])
            self.sim.environment['is_uniform_current'] = True

            # 2. Vortex field (mesoscale eddies / spatial mixing)
            self.sim.vortex_field = sim_functions.VortexField(
                density=10,
                intensity=self.np_random.uniform(0.0, 0.3),
                rng=np.random.default_rng(int(self.np_random.integers(0, 2 ** 31))),
                domain_size=self.domain[0],
                length_scale=self.eddy_length_scale,
            )
            self.sim.environment['is_vortex_currents'] = True

            # 3. Turbulent noise (small-scale temporal fluctuations)
            self.sim.turbolent_noise = sim_functions.TimeNoise(
                time=self.sim.time,
                freq=self.np_random.uniform(0.1, 1.0),
                intensity=self.np_random.uniform(0.0, 0.2),
                rng=np.random.default_rng(int(self.np_random.integers(0, 2 ** 31))),
            )
            self.sim.environment['is_noise_currents'] = True

            # 4. Global waves (time-dependent sinusoidal)
            self.sim.environment['global_waves'] = [{
                'amplitude': self.np_random.uniform(0.0, 0.2),
                'frequency': self.np_random.uniform(0.05, 0.5),
                'direction': self.np_random.uniform(0.0, 360.0),
                'shift':     self.np_random.uniform(0.0, 2 * np.pi),
            }]
            self.sim.environment['is_global_waves'] = True

            # 5. Local waves (position + time dependent)
            self.sim.environment['local_waves'] = [{
                'amplitude':  self.np_random.uniform(0.0, 0.2),
                'wavelength': self.np_random.uniform(500.0, 5000.0),
                'wavespeed':  self.np_random.uniform(0.1, 1.0),
                'direction':  self.np_random.uniform(0.0, 360.0),
                'shift':      self.np_random.uniform(0.0, 2 * np.pi),
            }]
            self.sim.environment['is_local_waves'] = True

            # 6. EkmanSpiral — rotates/decays the 2D surface current with depth
            self.sim.current_3d = sim_functions.EkmanSpiral(
                wind_speed=self.np_random.uniform(0.0, 10.0),
                wind_direction=self.np_random.uniform(0.0, 360.0),
                latitude=24.5,
                eddy_viscosity=0.05,
            )
            self.sim.environment['is_current_3d'] = True
            self.sim.environment['current_3d_model'] = 'ekman'

        # Define the target (S*, tau*), shared by all agents.
        if self.target_mode == "tail":
            # Rare LOW-salinity target; agents spawn on the land strip (off the
            # west/south borders), so the swarm must navigate the field.
            self._select_tail_target_and_starts(salinity_at)
        else:
            # Legacy: pick a point far enough from agent 0's spawn to force navigation.
            spawn = self.sim.agents[0].pos
            spawn_S = salinity_at(spawn[0], spawn[1], spawn[2])
            spawn_T = compute_turbidity(spawn[2])
            cand_S, cand_T = spawn_S, spawn_T
            for _ in range(100):  # safety cap; in practice ~1-2 iterations
                x_sel = self.np_random.uniform(0.0, self.domain[0])
                y_sel = self.np_random.uniform(0.0, self.domain[1])
                z_sel = self.np_random.uniform(0.0, self.domain[2])
                cand_S = salinity_at(x_sel, y_sel, z_sel)
                cand_T = compute_turbidity(z_sel)
                if (abs(cand_S - spawn_S) > 2 * self.epsilon_salinity
                        or abs(cand_T - spawn_T) > 2 * self.epsilon_turbidity):
                    break
            self.target_salinity = cand_S
            self.target_turbidity = cand_T

        # Initial observation (no history update: action=None). Seed each agent's
        # previous potential Φ(s_0) so the first step's shaping term is well-defined.
        obs = np.zeros((self.n_agents, self.local_observation_space.shape[0]), dtype=np.float32)
        for i in range(self.n_agents):
            o, phi0, _, _ = self._build_local_state(i)
            obs[i] = o
            self._prev_potential[i] = phi0
        info = {"global_state": self._build_global_state()}
        return obs, info

    # ------------------------------------------------------------------- step
    def step(self, actions):
        '''
        actions: array-like of shape (n_agents,) of discrete action indices.
        Returns obs (N, local_dim), rewards (N,), terminateds (N,),
        truncateds (N,), info (with "global_state").
        '''
        actions = np.asarray(actions).astype(np.int64)

        # 1. Set commands for all active agents, then advance the sim ONCE for
        #    the whole swarm (a single shared clock).
        for i, agent in enumerate(self.sim.agents):
            if self._success[i]:
                # Already-succeeded agents hold position (no-op).
                agent.cmd_local_vel = np.array([0.0, 0.0])
                agent.cmd_heave = 0.0
                continue
            mov = self._action_to_direction[actions[i]]
            agent.cmd_local_vel = np.array([mov[0] * self.v, mov[1] * self.v])  # surge, sway
            agent.cmd_heave = mov[2] * self.v * self.z_scale                    # heave (z); z_scale<1 -> finer depth control
            agent.cmd_heading = np.rad2deg(np.arctan2(mov[0], mov[1]))          # heading tracks motion

        for _ in range(self.frame_skip):
            self.sim.tick()
            for agent in self.sim.agents:
                agent.pos[0] = np.clip(agent.pos[0], 0.0, self.domain[0])
                agent.pos[1] = np.clip(agent.pos[1], 0.0, self.domain[1])
                agent.pos[2] = np.clip(agent.pos[2], 0.0, self.domain[2])
        self.t_step += 1

        # 2. Build next observation, reward, and success flags per agent.
        obs = np.zeros((self.n_agents, self.local_observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        out_of_time = self.t_step >= self.max_steps

        # Agents active (not already succeeded) at the start of this step; only
        # these receive the sparse +success_bonus when the swarm scores. Frozen
        # agents keep their 0.0 reward.
        active_at_entry = ~self._success.copy()
        # Track which agents reach the zone this step so the sparse reward can pay
        # the +success_bonus to the whole swarm when the first agent finds it.
        newly_terminated = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            if self._success[i]:
                # Frozen: re-emit obs (no history update), zero reward.
                obs[i] = self._build_local_state(i)[0]
                continue
            # phi_next = Φ(s') is the proximity potential of agent i's next state.
            o, phi_next, S, tau = self._build_local_state(i, actions[i])
            obs[i] = o

            if self._is_in_zone(S, tau):
                self._in_zone_steps[i] += 1
            else:
                self._in_zone_steps[i] = 0
            terminated_i = self._in_zone_steps[i] >= self._success_steps_required

            if self.reward_mode == "sparse":
                # Easier sparse reward: a flat -1 penalty per active agent per step.
                # The +success_bonus is paid below to ALL agents once any agent
                # reaches the target. Φ still feeds the observation history but does
                # NOT enter the reward.
                rewards[i] = -1.0
            else:
                # Potential-based reward shaping (Ng et al. 1999): r = r_sparse + γΦ(s') − Φ(s).
                # Φ at a true terminal (success) is 0; truncation is NOT terminal (the
                # agent bootstraps), so it keeps the real Φ(s'). The dense shaping
                # telescopes to a policy-independent constant, guiding the agent without
                # the loitering incentive of a raw positive dense reward.
                phi_next_eff = 0.0 if terminated_i else phi_next
                r = self.gamma * phi_next_eff - self._prev_potential[i]
                if terminated_i:
                    r += self.success_bonus
                rewards[i] = r
            if terminated_i:
                self._success[i] = True
                newly_terminated[i] = True
            self._prev_potential[i] = phi_next

        # Sparse mode: as soon as ANY agent reaches the target this step, every
        # agent (whether or not it personally found the zone) receives a pure
        # +success_bonus, overriding the -1 step penalty for that step.
        if self.reward_mode == "sparse" and bool(newly_terminated.any()):
            rewards[active_at_entry] = self.success_bonus

        # terminated = reached the target; truncated = the episode ended without
        # this agent reaching it. The episode ends when time runs out OR — with
        # end_on_any_success (the multi-agent success rule: first agent to find
        # the zone wins) — as soon as ANY agent has succeeded. The not-yet-
        # succeeded agents are then truncated (they bootstrap, exactly like the
        # max_steps path) so the trainer's `if d.all(): reset` fires this step.
        terminateds = self._success.copy()
        episode_over = out_of_time or (self.end_on_any_success and bool(terminateds.any()))
        truncateds = np.full(self.n_agents, episode_over) & (~self._success)

        info = {"global_state": self._build_global_state()}
        return obs, rewards, terminateds, truncateds, info

    # ----------------------------------------------------------------- helpers
    def _select_tail_target_and_starts(self, salinity_at):
        '''target_mode="tail": draw the target from the rare LOW-salinity tail and
        spawn agents on the land strip (a fixed clearance off the west/south
        borders).

        Salinity is the meaningful axis (turbidity is depth-only). The salinity
        sources sit on the west/south land borders, so high salinity hugs that
        coast where the agents spawn; targeting the LOW tail (interior / far
        corner) keeps the target away from the spawn strip and avoids trivially
        short episodes. We sample the field's value distribution, take an actual
        point from the lower-P% tail as the target (so the (S*, τ*) pair is
        reachable), then place each agent on the land strip via
        spawn_positions_along_land_strip.'''
        X, Y, Z = self.domain
        n = self.target_samples
        P = self.np_random.uniform([0.0, 0.0, 0.0], [X, Y, Z], size=(n, 3))
        try:  # vectorized accessor (analytical); falls back to per-point (NetCDF)
            S = np.asarray(salinity_at(P[:, 0], P[:, 1], P[:, 2]), dtype=float)
            if S.shape != (n,):
                raise ValueError
        except Exception:
            S = np.array([float(salinity_at(p[0], p[1], p[2])) for p in P])

        # Target: a random point from the rare LOW-salinity tail (argmin fallback).
        lo = np.percentile(S, self.target_percentile)
        tgt_idx = np.flatnonzero(S <= lo)
        if tgt_idx.size == 0:
            tgt_idx = np.array([int(np.argmin(S))])
        jt = int(tgt_idx[self.np_random.integers(tgt_idx.size)])
        self.target_salinity = float(S[jt])
        self.target_turbidity = float(compute_turbidity(P[jt, 2]))

        # Agent starts: distinct surface points on the land strip (500 m off the
        # west/south borders). Resample any spawn that lands already in the zone.
        starts = spawn_positions_along_land_strip(
            self.np_random, self.n_agents, self.domain, self.land_clearance)
        for i, agent in enumerate(self.sim.agents):
            pos = starts[i]
            for _ in range(100):  # safety cap; rare given the LOW-tail target
                S_i = float(np.asarray(salinity_at(pos[0], pos[1], pos[2])).reshape(-1)[0])
                if not self._is_in_zone(S_i, compute_turbidity(pos[2])):
                    break
                pos = spawn_positions_along_land_strip(
                    self.np_random, 1, self.domain, self.land_clearance)[0]
            agent.pos[:] = pos

    def _is_in_zone(self, salinity, turbidity) -> bool:
        '''True when measured (S, tau) lie within epsilon of the target couple.'''
        return (
            abs(salinity - self.target_salinity) < self.epsilon_salinity
            and abs(turbidity - self.target_turbidity) < self.epsilon_turbidity
        )

    def _build_action_table(self) -> np.ndarray:
        '''action index -> (dx, dy, dz) unit vector (index 13 = no-op).'''
        table = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=np.float64)
        norms = np.linalg.norm(table, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return table / norms

    def _measure(self, agent):
        '''Salinity, turbidity and body-frame currents at an agent's position.
        Single source of truth for both observation builders and the in-zone
        check. No salinity gradient is exposed (the agent infers direction from
        the (S, tau) history instead).'''
        x, y, z = agent.pos[0], agent.pos[1], agent.pos[2]
        if not self._nc_files:
            S = compute_salinity_analytical(x=x, y=y, z=z, sources=self.sources,
                                            sigma_h=self.sigma_h, sigma_v=self.sigma_v)
        else:
            S = self.sim.current_3d.salinity_at(x, y, z)
        tau = compute_turbidity(depth=z)

        # Currents rotated into the agent's body frame.
        currents = self.sim.depth_current_at(agent)
        psi = np.deg2rad(agent.psi)
        u = currents[0] * np.cos(psi) + currents[1] * np.sin(psi)
        v = currents[0] * np.sin(psi) - currents[1] * np.cos(psi)
        w = currents[2]
        return S, tau, u, v, w

    def _build_local_state(self, i, action=None):
        '''
        Returns (obs (4k+6,), potential, S, tau) for agent i. `potential` is
        Φ(s) = reward_func(...); the caller turns it into the shaped reward, while
        the history stores (action, Φ).

        Layout:
            (2k) history of (action, potential) pairs (agent i's own)
            (2k) history of measured (salinity, turbidity) (last row = current)
            (3)  body-frame currents u, v, w
            (2)  target salinity*, turbidity*
            (1)  depth
        '''
        agent = self.sim.agents[i]
        S, tau, u, v, w = self._measure(agent)
        potential = reward_func(S, tau, self.target_salinity, self.target_turbidity)

        if action is not None:
            self.histories[i] = np.roll(self.histories[i], -1, axis=0)
            self.histories[i, -1] = [action, potential]

        # (S, tau) history rolls on every build (incl. reset and frozen agents):
        # last row is the current measurement, earlier rows the previous k-1 steps.
        self.st_histories[i] = np.roll(self.st_histories[i], -1, axis=0)
        self.st_histories[i, -1] = [S, tau]

        obs = np.concatenate([
            self.histories[i].flatten(),
            self.st_histories[i].flatten(),
            np.array([u, v, w,
                      self.target_salinity, self.target_turbidity,
                      agent.pos[2]], dtype=np.float32),
        ]).astype(np.float32)
        return obs, potential, S, tau

    def _build_global_state(self):
        '''
        Centralized state ((2k + 6)*N + 2,) for the MAPPO critic:
            (2)        target salinity*, turbidity*
            per agent (2k + 6): u v w | (S, tau) history (2k) | x y z
        Depth is the z component of (x, y, z); it is not duplicated. The (S, tau)
        history mirrors the actor's so the critic is at least as informed. Call
        this AFTER the per-agent _build_local_state calls so st_histories holds
        the current measurement as its last row.
        '''
        parts = [self.target_salinity, self.target_turbidity]
        for i, agent in enumerate(self.sim.agents):
            _S, _tau, u, v, w = self._measure(agent)
            parts.extend([u, v, w])
            parts.extend(self.st_histories[i].flatten().tolist())
            parts.extend([agent.pos[0], agent.pos[1], agent.pos[2]])
        return np.array(parts, dtype=np.float32)

    def close(self):
        for loader in self._loaders.values():
            loader.close()
        self._loaders.clear()
        super().close()

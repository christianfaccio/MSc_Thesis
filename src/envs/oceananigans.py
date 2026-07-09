import glob
import os
import warnings
from collections import OrderedDict
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SwarmSwIM import Simulator, Agent
from src.models.reward import reward_func
import itertools
from src.models.turbidity import compute_turbidity


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

class SingleAgentEnv(gym.Env):
    '''
    This class represents the wrapped environment of the simulation. It builds from
    SwarmSwIM and is enclosed with Gymnasium for standardization. Training runs on
    Oceananigans NetCDF fields, following the explicit-gradient observation design
    of BaseEnv/PlumesEnv (no (S, τ) history buffer): the salinity gradient is
    queried directly from the field and handed to the agent.

    Observation (9,) — pure local-sensor + mission info, no global coordinates:
        (3) u v w      -> body-frame current vector (m/s)
        (3) gu gv gw   -> body-frame salinity gradient (PSU/m), gw down-positive
        (2) S-S* τ-τ*  -> target errors in salinity and turbidity
        (1) depth      -> agent depth (m, positive down)

    The salinity gradient (gu, gv, gw) is the field's spatial gradient rotated into
    the agent's body frame (same rotation as the currents), so the agent is told
    which way salinity increases without ever seeing its absolute position.

    Position and heading are deliberately excluded: per the project's design rule
    the agent does not know where it is and acts in its local frame. Heading ψ is
    fixed per episode at a random value (agent.xml heading_control='yawrate', no
    yaw commanded) and used internally to rotate currents and the salinity
    gradient into the body frame — actions are body-frame (surge, sway, heave)
    and SwarmSwIM maps them to world via the same Rot(ψ), so the two frames are
    coherent and the policy never needs to see ψ directly.

    Parameters:
        - xml_file -> SwarmSwIM simulation .xml
        - netcdf_file -> Oceananigans NetCDF data: a single file, a glob pattern, a
          directory, or a list of those. A random file is sampled each reset (in
          static mode a random frozen snapshot; in dynamic mode a random time
          window with fields linearly interpolated across snapshots). Each touched
          file keeps a cached FieldLoader (~90 MB of interpolators, two snapshots),
          so keep the set small (~10 files is fine).
        - k -> history buffer length (vestigial; retained for API parity, unused by
          the gradient-based observation)
        - v_agent -> agent commanded speed (m/s)
        - max_steps -> maximum env steps per episode before truncation
        - dt -> simulator timestep (s) per env step
        - frame_skip -> sim sub-steps per env step (action held constant)
        - gamma -> RL discount; MUST match the trainer's γ for shaping invariance
        - success_bonus -> sparse reward added on reaching the target
        - static_frame -> NetCDF: freeze one random snapshot per episode (no time
          evolution) vs. dynamic time window
        - min_band_grad -> reject targets whose success band is ~flat (PSU/m)
    '''
    def __init__(self,
                 xml_file: str,
                 netcdf_file: str,
                 k: int = 12,                       # history length of (action, reward)
                 v_agent: float = 1.0,              # agent speed in m/s
                 max_steps: int = 7200,             # steps of an episode before truncation - 2 hours
                 dt: float = 0.1,                   # seconds per step
                 domain = (1000.0, 1000.0, 100.0),
                 frame_skip: int = 10,              # sim sub-steps per env step (action held constant)
                 gamma: float = 0.999,              # RL discount; MUST match the trainer's γ for shaping invariance
                 success_bonus: float = 10.0,       # sparse reward on reaching the target
                 static_frame: bool = True,         # NetCDF: freeze one random snapshot per episode (no time evolution)
                 min_band_grad: float = 0.004,      # reject targets whose success band is ~flat (PSU/m); <=0 disables
                 target_min_dist_frac: float = 0.0, # min spawn→target distance as a fraction of the domain diagonal; 0 = no check (targets may land near spawn -> varied difficulty)
                 wall_penalty: float = 0.0,         # per-step reward penalty when the agent is pinned against a domain wall (fraction of frame_skip ticks clamped); 0 disables
                 success_steps_required: int = 1,   # consecutive in-zone steps needed to terminate as success; >1 forces the agent to arrive AND hold (kills single-step luck crossings on turbulent fields)
                 max_cached_loaders: int = 8,       # LRU cap on cached FieldLoaders (~90 MB each) per env instance
                 ):
        super().__init__()

        self.sim_xml = xml_file
        self.netcdf_file = netcdf_file
        self._nc_files = _resolve_nc_files(netcdf_file)
        self.k = k
        self.max_cached_loaders = max_cached_loaders
        self.v = v_agent
        self.max_steps = max_steps
        self.dt = dt
        self.domain = domain
        self.frame_skip = frame_skip
        self.gamma = gamma
        self.success_bonus = success_bonus
        self.static_frame = static_frame
        self.min_band_grad = min_band_grad
        self.target_min_dist_frac = target_min_dist_frac
        self.wall_penalty = wall_penalty
        self._success_steps_required = int(success_steps_required)
        # Potential of the previous state, Φ(s); set on reset and updated each step.
        # Reward is the sparse success bonus plus the potential-based shaping term
        # γΦ(s') − Φ(s) (Ng et al. 1999), which is policy-invariant.
        self._prev_potential = 0.0

        self.target_salinity = 0.0
        self.target_turbidity = 0.0
        self.current_salinity = 0.0
        self.current_turbidity = 0.0

        self._in_zone_steps = 0
        self.epsilon_salinity = 0.3
        self.epsilon_turbidity = 0.05   # TODO: try with epsilon turbidity bound to meters

        self.action_space = gym.spaces.Discrete(27)
        obs_dim = 9
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._action_to_direction = self._build_action_table()  # scalar -> [dx,dy,dz] normalized

        self.sim = None
        # NetCDF ocean-data loaders, created lazily and reused across episodes
        # (re-opening files each reset leaks file handles). Bounded LRU cache: with
        # a large file set, caching every touched file (~90 MB each) blows memory
        # across parallel envs, so the least-recently-used loader is evicted and
        # closed once max_cached_loaders is exceeded — a cache miss just re-opens.
        self._loaders = OrderedDict()
        self.active_netcdf_path = None
        self._warned_short_record = False

    def _build_action_table(self) -> np.array:
        '''Returns an array of action->(dx,dy,dz) normalized.'''
        table = list(itertools.product([-1, 0, 1], repeat=3))
        norms = np.linalg.norm(table, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return table / norms
    
    def _measure(self, agent):
        '''Salinity, turbidity and body-frame currents at an agent's position.
        Single source of truth for the observation builder, the in-zone check and
        external tooling (matches MultiAgentEnv._measure). Does not mutate state.'''
        x, y, z = agent.pos[0], agent.pos[1], agent.pos[2]
        
        S = self.sim.current_3d.salinity_at(x, y, z)
        tau = compute_turbidity(depth=z)

        gx, gy, gz = self.sim.current_3d.salinity_gradient_at(x, y, z)

        currents = self.sim.depth_current_at(agent)
        psi = np.deg2rad(agent.psi)
        # World -> body frame via Rot(psi)^T, the exact inverse of SwarmSwIM's
        # body->world map for cmd_local_vel (agent_class.update_planar applies
        # world = Rot(psi) @ (surge, sway)). Using the true inverse keeps sensed
        # directions coherent with how actions actually move the agent — the
        # previous form negated the body-y (sway) component (a reflection, not
        # a rotation), so "follow the sensed gradient" was laterally mirrored.
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        u = currents[0] * cos_psi + currents[1] * sin_psi
        v = -currents[0] * sin_psi + currents[1] * cos_psi
        w = currents[2]

        # rotate salinity gradient into body frame (same rotation as currents)
        gu = gx * cos_psi + gy * sin_psi
        gv = -gx * sin_psi + gy * cos_psi
        gw = gz

        return S, tau, u, v, w, gu, gv, gw
    
    def _build_state(self, agent, action=None) -> tuple[np.ndarray, float]:
        '''
        Returns the observation of dimension (9,) and the proximity potential
        Φ(s) = reward_func(...). The caller turns Φ into the shaped reward.

        Observation layout (9,):
            (3)     -> body-frame currents (u, v, w)
            (3)     -> body-frame salinity gradient (gu, gv, gw)
            (2)     -> target errors (S - S*, τ - τ*)
            (1)     -> depth
        '''
        new_salinity, new_turbidity, u, v, w, gu, gv, gw = self._measure(agent)

        potential = reward_func(new_salinity, new_turbidity, self.target_salinity, self.target_turbidity,
                                eps_s=self.epsilon_salinity, eps_tau=self.epsilon_turbidity)

        self.current_salinity = new_salinity
        self.current_turbidity = new_turbidity

        agent_depth = agent.pos[2]

        return np.array([
                u, v, w,
                gu, gv, gw,
                self.current_salinity - self.target_salinity,
                self.current_turbidity - self.target_turbidity,
                agent_depth,
        ], dtype=np.float32), potential
    
    def _is_in_zone(self) -> bool:
        '''True when measured (S, tau) lie within epsilon of the target couple.'''
        return (
            abs(self.current_salinity - self.target_salinity) < self.epsilon_salinity
            and abs(self.current_turbidity - self.target_turbidity) < self.epsilon_turbidity
        )
    
    def close(self):
        for loader in self._loaders.values():
            loader.close()
        self._loaders.clear()
        super().close()

    def reset(self, seed=None, options=None):
        '''
        Method that initializes an environment.

        Parameters:
            - seed (int)
            - options (dict | None) — Gymnasium passes this through wrappers; unused here.

        Output:
            - state (np.array)
            - info (dict)
        '''
        super().reset(seed=seed)

        # Reset success counter — without this, a successful previous episode
        # leaves _in_zone_steps == 3 and the next episode could terminate immediately.
        self._in_zone_steps = 0

        # Create the env (Simulator class). A random NetCDF file is drawn each
        # episode; its FieldLoader is cached and shared across resets, and the
        # Simulator accepts the loader instance in place of a file path.
        loader = None
        if self._nc_files:
            from SwarmSwIM.ocean_data import FieldLoader
            path = self._nc_files[int(self.np_random.integers(len(self._nc_files)))]
            if path in self._loaders:
                self._loaders.move_to_end(path)  # mark most-recently-used
            else:
                if len(self._loaders) >= self.max_cached_loaders:
                    _, evicted = self._loaders.popitem(last=False)  # drop LRU
                    evicted.close()
                self._loaders[path] = FieldLoader(path)
            loader = self._loaders[path]
            self.active_netcdf_path = path
        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.sim_xml, netcdf_file=loader)
        
        self.sim.remove(*self.sim.agents)  # drop ALL xml-defined agents, not just the first
        
        # Add agent
        agent = Agent(
                name="A",
                Dt=self.dt,
                initialPosition=np.array([self.np_random.uniform(0.0, self.domain[0]), 
                                          self.np_random.uniform(0.0, self.domain[1]), 
                                          self.np_random.uniform(0.0, self.domain[2])]),
                initialHeading=self.np_random.uniform(-180.0, 180.0),
                agent_xml="config/agent.xml",
                rng=self.np_random.integers(2**31)
            )
        self.sim.add(agent)

        if self.static_frame:
            # Static-frame mode: freeze the fields at one randomly chosen
            # snapshot for the whole episode (no intra-episode time evolution).
            # Episode variability comes from the file choice plus this index.
            idx = int(self.np_random.integers(len(loader.times)))
            loader.set_snapshot(idx)
        else:
            # Dynamic mode: episode variability comes from the file choice plus
            # a random time window in the data; fields are linearly interpolated
            # in time as the episode advances (the Simulator passes sim_time).
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

        # Target selection: sample a point that is (a) at least target_min_dist_frac
        # of the domain diagonal from the spawn (0 -> no distance check, so targets
        # may land near the spawn and episode difficulty varies), (b) has an (S, τ)
        # couple meaningfully different from the spawn's, and (c) sits on a non-flat
        # salinity band so a local-sensing agent has a gradient to home on. (S*, τ*)
        # are read AT that point, so the success zone exists there by construction.
        # The target is ALWAYS assigned to the accepted candidate — the previous logic
        # broke out without assigning, leaving S*≈0 (unreachable).
        grad_at = loader.salinity_gradient_at
        spawn = self.sim.agents[0].pos
        self.current_salinity = salinity_at(spawn[0], spawn[1], spawn[2])
        self.current_turbidity = compute_turbidity(spawn[2])
        min_dist = self.target_min_dist_frac * float(np.linalg.norm(self.domain))
        # Fallback: if no candidate qualifies (extremely unlikely on these fields),
        # keep spawn (S, τ) so the couple is at least valid and reachable.
        self.target_salinity = self.current_salinity
        self.target_turbidity = self.current_turbidity
        for _ in range(50):
            x_sel = self.np_random.uniform(0.0, self.domain[0])
            y_sel = self.np_random.uniform(0.0, self.domain[1])
            z_sel = self.np_random.uniform(0.0, self.domain[2])
            if np.linalg.norm(np.array([x_sel, y_sel, z_sel]) - spawn) < min_dist:
                continue
            cand_S = salinity_at(x_sel, y_sel, z_sel)
            cand_T = compute_turbidity(z_sel)
            # Reject trivial targets: both S and τ already within reach of spawn.
            if (abs(cand_S - self.current_salinity) <= 2 * self.epsilon_salinity
                    and abs(cand_T - self.current_turbidity) <= 2 * self.epsilon_turbidity):
                continue
            # Reject flat-band targets with no local directional signal.
            if self.min_band_grad > 0.0:
                gx, gy, _ = grad_at(x_sel, y_sel, z_sel)
                if np.hypot(gx, gy) < self.min_band_grad:
                    continue
            self.target_salinity = cand_S
            self.target_turbidity = cand_T
            break

        # Initialize history buffers: (action, potential) pairs and (S, τ) measurements
        self.history = np.zeros((self.k, 2), dtype=np.float32)
        self.st_history = np.zeros((self.k, 2), dtype=np.float32)
        self.t_step = 0
        
        obs, phi0 = self._build_state(self.sim.agents[0])
        self._prev_potential = phi0
        return obs, {}
    
    def step(self, action):
        '''
        Method that given an action from the policy updates the environment.

        Parameters:
            - action (scalar)
        
        Output:
            - s' (next state)
            - reward (scalar)
            - terminated (Bool)
            - truncated (Bool)
        '''
        # Translate action into movement. The action triple is a BODY-frame
        # direction (surge, sway, heave): config/agent.xml uses
        # heading_control='yawrate' and no yaw is ever commanded, so psi stays
        # FIXED at its random initial value for the whole episode and SwarmSwIM
        # maps (surge, sway) to world coordinates via Rot(psi). The observation
        # rotates currents/gradient into the same body frame (see _measure), so
        # sensing and actuation share one coherent frame. (A previous version
        # also set cmd_heading here expecting the heading to auto-track the
        # motion direction — silently ignored in yawrate mode.)
        mov = self._action_to_direction[action]
        agent = self.sim.agents[0]
        agent.cmd_local_vel = np.array([mov[0]*self.v, mov[1]*self.v])  # surge (x) and sway (y)
        agent.cmd_heave = mov[2]*self.v                                 # heave (z)

        # Doing the step in the sim
        # NOTE: reward is sampled only at the final sub-step (only-last), not summed across
        # the frame_skip ticks. This preserves the reward scale (and the meaning of the
        # +10 success bonus) when sweeping frame_skip, but PPO loses the integrated
        # signal of any high-reward region the agent passed through mid-skip. Worth
        # revisiting later — compare against summed-reward aggregation (paper convention,
        # Andrychowicz et al. 2021 §3.6) once a frame_skip ablation has been run.
        clamped_ticks = 0
        for _ in range(self.frame_skip):
            self.sim.tick()
            # Keep the agent inside the domain box: motion commands and currents
            # would otherwise push it above the surface (z < 0), below the seabed
            # (z > domain depth) or out of the horizontal extent. Clamped every
            # tick so field queries never run from out-of-bounds positions. A tick
            # that actually needed clamping means the agent was driving into a wall
            # (wasted motion) — counted so step() can penalize wall-pinning.
            clipped = np.clip(agent.pos, [0.0, 0.0, 0.0], self.domain)
            if not np.array_equal(clipped, agent.pos):
                clamped_ticks += 1
            agent.pos[:] = clipped
        self.t_step += 1
        # Fraction of the frame_skip spent pinned against a wall (0..1).
        wall_frac = clamped_ticks / self.frame_skip

        # Next state (s'); phi_next = Φ(s') is the proximity potential.
        next_obs, phi_next = self._build_state(agent, action)

        # Truncation and termination checks. Success requires the agent to STAY in
        # the zone for `_success_steps_required` consecutive steps, not just clip
        # through it once: on a turbulent field a single in-zone step is achievable by
        # stochastic luck (inflates train success, evaporates as entropy drops, and
        # never reproduces under a greedy rollout). Holding is what we actually want.
        truncated = (self.t_step >= self.max_steps)
        if self._is_in_zone():
            self._in_zone_steps += 1
        else:
            self._in_zone_steps = 0
        terminated = self._in_zone_steps >= self._success_steps_required

        # Potential-based reward shaping (Ng et al. 1999): r = r_sparse + γΦ(s') − Φ(s).
        # Φ at a true terminal (success) state is 0; truncation is NOT terminal (the
        # agent bootstraps), so it keeps the real Φ(s') — using 0 there would break
        # policy invariance. The dense shaping telescopes to a policy-independent
        # constant, so it guides without creating an incentive to loiter and avoid
        # finishing the way a raw positive dense reward did.
        phi_next_eff = 0.0 if terminated else phi_next
        reward = self.gamma * phi_next_eff - self._prev_potential
        if terminated:
            reward += self.success_bonus
        # Wall-pinning penalty (breaks strict shaping invariance by design): a small
        # cost proportional to the fraction of the step spent clamped against a
        # domain boundary, to discourage the degenerate "drive into a wall and stall"
        # local optimum. 0 by default (opt-in via the wall_penalty constructor arg).
        if self.wall_penalty > 0.0:
            reward -= self.wall_penalty * wall_frac
        self._prev_potential = phi_next

        return next_obs, reward, terminated, truncated, {}


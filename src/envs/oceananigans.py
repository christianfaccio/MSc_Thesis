import glob
import os
import warnings
from collections import OrderedDict
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial import cKDTree
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

class OceananigansEnv(gym.Env):
    def __init__(self,
                 xml_file: str,
                 netcdf_file: str,
                 k: int = 0,                        # history length
                 n_agents: int = 1,                 # number of agents
                 v_agent: float = 1.0,              # agent speed in m/s
                 max_steps: int = 3600,             # steps of an episode before truncation - 2 hours
                 dt: float = 0.1,                   # seconds per step
                 domain = (1000.0, 1000.0, 100.0),  # domain
                 frame_skip: int = 10,              # sim sub-steps per env step (action held constant)
                 gamma: float = 0.9997,              # RL discount; MUST match the trainer's γ for shaping invariance
                 success_bonus: float = 20.0,       # sparse reward on reaching the target
                 static_frame: bool = True,         # NetCDF: freeze one random snapshot per episode (no time evolution)
                 success_steps_required: int = 1,   # consecutive in-zone steps needed to terminate as success; >1 forces the agent to arrive AND hold (kills single-step luck crossings on turbulent fields)
                 max_cached_loaders: int = 8,       # LRU cap on cached FieldLoaders (~90 MB each) per env instance
                 end_on_any_success: bool = True,   # multi-agent: end the episode on the first success
                 epsilon_salinity: float = 0.3,     # success tolerance on |S - S*| (PSU); size to ~3% of the field's per-snapshot span
                 epsilon_turbidity: float = 0.05,   # success tolerance on |τ - τ*|   TODO: try with epsilon turbidity bound to meters
                 sigma_s: float = 3.0,              # wide shaping-kernel width in S (PSU); size to ~0.3× the field span (3.0 for the ~10 PSU no_buoyancy fields, 1.5 for the ~5 PSU buoyancy_active ones)
                 sigma_tau: float = 0.3,            # wide shaping-kernel width in τ
                 target_mode: str = "random",       # "random" = S* at a uniform random point; "tail" = S* from the low tail of the snapshot's S distribution
                 target_percentile: float = 5.0,    # tail mode: S* below this percentile of the field's salinity values
                 reward_potential: str = "distance",   # "error" = Φ over measurement error (has filament local optima); "distance" = Φ over physical distance to the success zone (monotone, training-time privileged)
                 dead_reckoning: bool = False,      # append the body-frame dead-reckoned displacement from spawn (3) to the obs — the odometry ablation over the 9-dim baseline
                 ):
        super().__init__()

        self.sim_xml = xml_file
        self.netcdf_file = netcdf_file
        self._nc_files = _resolve_nc_files(netcdf_file)
        self.k = k
        self.max_cached_loaders = max_cached_loaders
        self.n_agents = int(n_agents)
        self.v = v_agent
        self.max_steps = max_steps
        self.dt = dt
        self.domain = domain
        self.frame_skip = frame_skip
        self.gamma = gamma
        self.success_bonus = success_bonus
        # BASELINE SCOPE: reset() implements only the frozen-snapshot / uniform-random
        # -target configuration. The tail-target and dynamic-window code paths were
        # stripped for this debug baseline, so reject their flags loudly rather than
        # silently running a different experiment than the one requested.
        if not static_frame:
            raise NotImplementedError(
                "static_frame=False (dynamic time window) is not implemented in the "
                "debug baseline env; reset() always freezes one snapshot.")
        self.static_frame = static_frame
        self._success_steps_required = int(success_steps_required)
        self.end_on_any_success = end_on_any_success
        if target_mode != "random":
            raise NotImplementedError(
                f"target_mode={target_mode!r} is not implemented in the debug baseline "
                "env; reset() always samples (S*, τ*) at a uniform random field point.")
        self.target_mode = target_mode
        self.target_percentile = float(target_percentile)
        if reward_potential not in ("error", "distance"):
            raise ValueError(
                f"reward_potential must be 'error' or 'distance', got {reward_potential!r}")
        self.reward_potential = reward_potential
        # Distance-mode zone index, rebuilt each reset: cKDTree over the grid
        # cells satisfying both success tests (agent-frame coords, z positive down).
        self._zone_tree = None
        self._domain_diag = float(np.linalg.norm(self.domain))

        self.target_salinity = 0.0
        self.target_turbidity = 0.0
        # Latest measured (S, τ) of agent 0, refreshed every reset/step — kept for
        # external tooling/back-compat; per-agent logic uses local values instead.
        self.current_salinity = 0.0
        self.current_turbidity = 0.0

        self.epsilon_salinity = epsilon_salinity
        self.epsilon_turbidity = epsilon_turbidity
        self.sigma_s = sigma_s
        self.sigma_tau = sigma_tau

        # Per-agent episode state (re-allocated in reset()).
        # Potential of the previous state, Φ(s); reward is the sparse success bonus
        # plus the potential-based shaping term γΦ(s') − Φ(s) (Ng et al. 1999),
        # which is policy-invariant.
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)
        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)

        self.dead_reckoning = bool(dead_reckoning)
        self.action_space = gym.spaces.Discrete(27)
        obs_dim = 9 + (3 if self.dead_reckoning else 0) + 5 * self.k
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # Per-agent local obs (== observation_space) and centralized MAPPO state,
        # 11 per agent (see _build_global_state) — named to match the interface
        # the multi-agent trainers consume (MultiAgentBaseEnv/MultiAgentPlumesEnv).
        self.local_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.global_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(11 * self.n_agents,), dtype=np.float32)
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
        external tooling. Does not mutate state.'''
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

    def _build_state(self, i, agent, action=None) -> tuple[np.ndarray, float, float, float]:
        '''
        Returns (obs (9+5k,), Φ(s), S, τ) for agent index `i`; the caller turns
        the potential Φ into the shaped reward and uses (S, τ) for the in-zone
        check. When an `action` index is given (a step, not a reset, and the
        agent is not frozen) the agent's history is advanced first: the newest
        row becomes (action direction, S - S*, τ - τ*) measured AFTER that
        action, so the last history row always mirrors the current errors in the
        frame part.

        Observation layout (9 [+3] + 5k,) — the BASELINE sensor frame plus the
        optional dead-reckoning ablation block:
            (3)     -> body-frame currents (u, v, w)
            (3)     -> body-frame salinity gradient (gu, gv, gw)
            (2)     -> target errors (S - S*, τ - τ*)
            (1)     -> depth
            (3)     -> ONLY if dead_reckoning: body-frame displacement from the
                       spawn point (ddx, ddy, dwz) — purely relative odometry,
                       no absolute position leaks into the actor
            (5k)    -> history, oldest->newest rows of (dx, dy, dz, S-S*, τ-τ*)
                       (empty at the k=0 baseline default)
        '''
        S, tau, u, v, w, gu, gv, gw = self._measure(agent)

        potential = self._potential_at(agent, S, tau)

        dS = S - self.target_salinity
        dT = tau - self.target_turbidity
        if action is not None and self.k > 0:
            self._hist[i] = np.roll(self._hist[i], -1, axis=0)
            self._hist[i, -1, :3] = self._action_to_direction[action]
            self._hist[i, -1, 3] = dS
            self._hist[i, -1, 4] = dT

        frame = [u, v, w,
                 gu, gv, gw,
                 dS, dT,
                 agent.pos[2]]
        if self.dead_reckoning:
            # Displacement from spawn, world -> body with the same Rot(psi)^T as
            # _measure (z shared between frames, no rotation needed).
            dwx, dwy, dwz = agent.pos - self._spawn_pos[i]
            psi = np.deg2rad(agent.psi)
            cos_psi, sin_psi = np.cos(psi), np.sin(psi)
            frame += [dwx * cos_psi + dwy * sin_psi,
                      -dwx * sin_psi + dwy * cos_psi,
                      dwz]
        obs = np.concatenate([
                np.array(frame, dtype=np.float32),
                self._hist[i].reshape(-1),
        ])
        return obs, potential, S, tau

    def _build_global_state(self):
        '''Centralized state (11·N,) for the MAPPO critic — per agent:
            (2): target errors (S - S*, τ - τ*)
            (3): body-frame currents u v w
            (3): body-frame salinity gradient gu gv gw
            (3): absolute position x y z (the critic MAY see global coordinates;
                 only the actor is restricted to local sensing)
        '''
        parts = []
        for agent in self.sim.agents:
            S, tau, u, v, w, gu, gv, gw = self._measure(agent)
            parts.extend([S - self.target_salinity,
                          tau - self.target_turbidity,
                          u, v, w,
                          gu, gv, gw,
                          agent.pos[0], agent.pos[1], agent.pos[2]])
        return np.array(parts, dtype=np.float32)

    def _build_zone_index(self, loader):
        '''Distance-potential support: KD-tree over every grid cell of the
        episode's snapshot that satisfies BOTH success tests (|S-S*|<ε_S and
        |τ-τ*|<ε_τ). Reads the raw NetCDF snapshot (vectorized, ~6 MB) instead
        of thousands of interpolator calls. Points are stored in agent-frame
        coordinates (z positive down). In dynamic mode the zone is built from
        the window-START snapshot and NOT rebuilt as the field evolves — an
        accepted approximation while training is static-first.

        This is TRAINING-TIME privileged information (the simulator knows the
        whole field); it feeds only the reward, never the observation, so
        policy invariance (Ng et al. 1999) and decentralized execution hold.
        '''
        tidx = loader.time_index if loader.time_index is not None else loader.window_start
        S3d = loader.ds["S"].isel(time=int(tidx)).values.astype(float)  # (z, y, x)
        z_down = -loader.z  # loader grid z is negative-up; agents use positive-down
        tau_levels = compute_turbidity(z_down)
        z_ok = np.abs(tau_levels - self.target_turbidity) < self.epsilon_turbidity
        mask = (np.abs(S3d - self.target_salinity) < self.epsilon_salinity) \
            & z_ok[:, None, None]
        iz, iy, ix = np.nonzero(mask)
        if iz.size == 0:
            # (S*, τ*) were read AT a field point, so this only happens through
            # interpolation-vs-cell mismatch at very tight ε: fall back to the
            # single grid cell closest to the target couple (τ band first).
            err = np.abs(S3d - self.target_salinity) \
                + np.where(z_ok, 0.0, np.inf)[:, None, None]
            if not np.isfinite(err).any():
                err = np.abs(S3d - self.target_salinity)
            iz, iy, ix = (idx.reshape(1) for idx in
                          np.unravel_index(np.argmin(err), S3d.shape))
        elif iz.size > 200_000:
            # Grid-spacing-level distance error from subsampling is negligible;
            # keeps tree build/query costs bounded on huge (random-mode) zones.
            keep = self.np_random.choice(iz.size, size=200_000, replace=False)
            iz, iy, ix = iz[keep], iy[keep], ix[keep]
        pts = np.column_stack((loader.x[ix], loader.y[iy], z_down[iz]))
        self._zone_tree = cKDTree(pts)

    def _potential_at(self, agent, S, tau) -> float:
        '''Shaping potential Φ(s) for one agent — the reward_potential switch.
        "error":    Gaussian over the measurement error (agent-sensible, but the
                    turbulent field folds it into filament local optima).
        "distance": 10·(1 − d/diag), d = distance to the nearest success-zone
                    cell — monotone toward the zone, no local optima by
                    construction (linear, not exponential: an exp(−d/σ)
                    potential saturates far from the zone and would recreate a
                    flat far field). The ×10 keeps the per-step shaping term
                    from vanishing next to the +10 success bonus once the
                    reward normalizer rescales by the return std.
        Both are functions of the true state only, so the γΦ(s')−Φ(s) shaping
        telescopes identically and the optimal policy is unchanged.'''
        if self.reward_potential == "distance":
            d, _ = self._zone_tree.query(
                (agent.pos[0], agent.pos[1], agent.pos[2]))
            return 10.0 * (1.0 - float(d) / self._domain_diag)
        return reward_func(S, tau, self.target_salinity, self.target_turbidity,
                           sigma_s=self.sigma_s, sigma_tau=self.sigma_tau,
                           eps_s=self.epsilon_salinity, eps_tau=self.epsilon_turbidity)

    def _is_in_zone(self, salinity, turbidity) -> bool:
        '''True when the given (S, τ) lie within epsilon of the target couple.'''
        return (
            abs(salinity - self.target_salinity) < self.epsilon_salinity
            and abs(turbidity - self.target_turbidity) < self.epsilon_turbidity
        )

    def close(self):
        for loader in self._loaders.values():
            loader.close()
        self._loaders.clear()
        super().close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)

        # Create Sim
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

        # Add N agents with random position + heading.
        for i in range(self.n_agents):
            agent = Agent(
                    name=f"A{i + 1:02d}",
                    Dt=self.dt,
                    initialPosition=np.array([self.np_random.uniform(0.0, self.domain[0]),
                                              self.np_random.uniform(0.0, self.domain[1]),
                                              self.np_random.uniform(0.0, self.domain[2])]),
                    initialHeading=self.np_random.uniform(-180.0, 180.0),
                    agent_xml="config/agent.xml",
                    rng=int(self.np_random.integers(2**31))
                )
            self.sim.add(agent)

        # Set snapshot index
        idx = int(self.np_random.integers(len(loader.times)))
        loader.set_snapshot(idx)

        # Set current and target values
        spawn = self.sim.agents[0].pos
        self.current_salinity = loader.salinity_at(spawn[0], spawn[1], spawn[2])
        self.current_turbidity = compute_turbidity(spawn[2])

        x_sel = self.np_random.uniform(0.0, self.domain[0])
        y_sel = self.np_random.uniform(0.0, self.domain[1])
        z_sel = self.np_random.uniform(0.0, self.domain[2])
        self.target_salinity = loader.salinity_at(x_sel, y_sel, z_sel)
        self.target_turbidity = compute_turbidity(z_sel)

        # Reward 
        if self.reward_potential == "distance":
            self._build_zone_index(loader)

        # Dead-reckoning anchor: displacement in the obs is measured from here.
        self._spawn_pos = np.array([a.pos.copy() for a in self.sim.agents])

        # Init time step
        self.t_step = 0

        # Observation history
        self._hist = np.zeros((self.n_agents, self.k, 5), dtype=np.float32)
        for i, agent in enumerate(self.sim.agents):
            self._hist[i, :, 3] = (loader.salinity_at(agent.pos[0], agent.pos[1], agent.pos[2])
                                   - self.target_salinity)
            self._hist[i, :, 4] = compute_turbidity(agent.pos[2]) - self.target_turbidity

        # Init observation space for all agents
        obs = np.zeros((self.n_agents, self.observation_space.shape[0]), dtype=np.float32)
        for i, agent in enumerate(self.sim.agents):
            o, phi0, S, tau = self._build_state(i, agent)
            obs[i] = o
            self._prev_potential[i] = phi0
            if i == 0:
                self.current_salinity = S
                self.current_turbidity = tau
        info = {"global_state": self._build_global_state()}
        if self.n_agents == 1:
            return obs[0], info
        return obs, info

    def step(self, action):
        actions = np.atleast_1d(np.asarray(action)).astype(np.int64)

        for i, agent in enumerate(self.sim.agents):
            if self._success[i]:
                agent.cmd_local_vel = np.array([0.0, 0.0])
                agent.cmd_heave = 0.0
                continue
            mov = self._action_to_direction[actions[i]]
            agent.cmd_local_vel = np.array([mov[0]*self.v, mov[1]*self.v])  # surge (x) and sway (y)
            agent.cmd_heave = mov[2]*self.v                                 # heave (z)

        for _ in range(self.frame_skip):
            self.sim.tick()
            for agent in self.sim.agents:
                # Hard-clamp into the domain box; no wall penalty at the
                # baseline (agents may slide along a wall for free).
                agent.pos[:] = np.clip(agent.pos, [0.0, 0.0, 0.0], self.domain)
        self.t_step += 1

        # Build next obs, reward and success flags per agent
        obs = np.zeros((self.n_agents, self.observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i, agent in enumerate(self.sim.agents):
            next_obs_i, phi_next, S, tau = self._build_state(
                i, agent, None if self._success[i] else actions[i])
            obs[i] = next_obs_i
            if i == 0:
                self.current_salinity = S
                self.current_turbidity = tau
            if self._success[i]:
                continue

            if self._is_in_zone(S, tau):
                self._in_zone_steps[i] += 1
            else:
                self._in_zone_steps[i] = 0
            terminated_i = self._in_zone_steps[i] >= self._success_steps_required

            phi_next_eff = 0.0 if terminated_i else phi_next
            r = self.gamma * phi_next_eff - self._prev_potential[i]
            if terminated_i:
                r += self.success_bonus
                self._success[i] = True
            rewards[i] = r
            self._prev_potential[i] = phi_next

        terminateds = self._success.copy()
        out_of_time = self.t_step >= self.max_steps
        episode_over = out_of_time or (self.end_on_any_success and bool(terminateds.any()))
        truncateds = np.full(self.n_agents, episode_over) & (~self._success)

        info = {"global_state": self._build_global_state()}
        if self.n_agents == 1:
            return (obs[0], float(rewards[0]), bool(terminateds[0]),
                    bool(truncateds[0]), info)
        return obs, rewards, terminateds, truncateds, info

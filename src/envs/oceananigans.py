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
                 communication: bool = False,       # multi-agent: append a per-neighbor block (5 each) enabling field triangulation — [in_range, rel_x, rel_y, rel_z (body frame), S_j - S*]
                 comms_radius: float = float("inf"),# communication range (m); neighbors farther than this are zeroed (in_range=0). inf = global sharing (default)
                 spawn_mode: str = "random",        # "random" = uniform over the domain; "origin" = every agent at (0,0,0); "max_dist" = evenly spread along the two land walls at z=0 (see _spawn_positions)
                 min_spawn_distance: float = 0.0,   # if >0, reject-sample spawns until every agent starts at least this many metres from the nearest success-zone cell (a distant-start difficulty knob; 0 = original uniform spawn). Only meaningful with spawn_mode="random"
                 spawn_max_tries: int = 200,        # rejection-sampling budget per agent for min_spawn_distance; the farthest candidate found is used if none clears the threshold
                 # --- team-reward block (all defaults reproduce the individual-reward baseline byte-for-byte) ---
                 alpha_individual: float = 1.0,     # weight on the per-agent potential Φ_i (the original shaping term)
                 beta_difference: float = 0.0,      # weight on the difference-reward potential D_i = G(s) − G(s_-i); the division-of-labour incentive
                 lambda_separation: float = 0.0,    # weight on the anti-redundancy potential Φ_sep (saturating nearest-neighbour separation)
                 separation_scale: float = 150.0,   # ℓ (m): separation past this buys no Φ_sep — size to the field's salinity-gradient correlation length (~100-150 m measured)
                 shared_success_bonus: bool = False,# give success_bonus to EVERY live agent, not just the one that reached; this is what turns the race into a cooperative game
                 coverage_cell: float = 50.0,       # (m) voxel edge for the episode coverage/redundancy diagnostic; 0 disables the tracking
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
        if target_mode not in ("random", "tail"):
            raise NotImplementedError(
                f"target_mode={target_mode!r} is not implemented in this env "
                "(only 'random' and 'tail' are supported).")
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

        # --- team reward ---
        # The reward is a weighted sum of THREE potentials, each a function of the
        # joint state, so every one of them keeps the γΦ(s')−Φ(s) telescoping and
        # leaves the optimal JOINT policy unchanged (Ng et al. 1999; Devlin &
        # Kudenko 2011 for the multi-agent extension):
        #   Φ_i   individual distance-to-zone  — dense, per-agent guidance
        #   D_i   difference reward G(s)−G(s_-i) — credit only for progress the
        #         team would NOT have had without agent i (Wolpert & Tumer 2002,
        #         as a potential: Devlin et al. 2014). This is the division-of-
        #         labour term: shadowing a teammate earns exactly zero.
        #   Φ_sep saturating nearest-neighbour separation — dense anti-redundancy.
        # shared_success_bonus is the one change that genuinely moves the
        # equilibrium (a sparse bonus is not a potential): with end_on_any_success
        # the default per-agent bonus makes the episode a RACE (if a teammate wins
        # first, everyone else loses their shot), which is why coordination has
        # nothing to buy at the baseline.
        self.alpha_individual = float(alpha_individual)
        self.beta_difference = float(beta_difference)
        self.lambda_separation = float(lambda_separation)
        self.separation_scale = float(separation_scale)
        self.shared_success_bonus = bool(shared_success_bonus)
        self.team_reward = (self.beta_difference != 0.0) or (self.lambda_separation != 0.0)
        if self.team_reward and self.n_agents < 2:
            raise ValueError(
                "beta_difference/lambda_separation require n_agents >= 2 "
                f"(got n_agents={self.n_agents}); both terms are identically zero for a single agent.")
        if self.separation_scale <= 0.0:
            raise ValueError(f"separation_scale must be > 0, got {self.separation_scale}")
        self._prev_difference = np.zeros(self.n_agents, dtype=np.float64)
        self._prev_separation = np.zeros(self.n_agents, dtype=np.float64)
        # Coverage diagnostic: per-agent set of visited voxel ids, reduced at
        # episode end to a redundancy ratio (see _coverage_stats).
        self.coverage_cell = float(coverage_cell)
        self._visited = None
        self._first_success_step = None

        self.dead_reckoning = bool(dead_reckoning)
        self.communication = bool(communication)
        self.spawn_mode = str(spawn_mode)
        if self.spawn_mode not in ("random", "origin", "max_dist"):
            raise ValueError(
                f"spawn_mode must be 'random', 'origin' or 'max_dist', "
                f"got {spawn_mode!r}")
        self.min_spawn_distance = float(min_spawn_distance)
        if self.min_spawn_distance > 0.0 and self.spawn_mode != "random":
            raise ValueError(
                "min_spawn_distance reject-samples the spawn point, which is "
                f"meaningless with the deterministic spawn_mode={self.spawn_mode!r}. "
                "Use spawn_mode='random', or set min_spawn_distance=0.")
        self.spawn_max_tries = int(spawn_max_tries)
        self.comms_radius = float(comms_radius)
        self.action_space = gym.spaces.Discrete(27)
        # obs = 9 baseline [+3 dead-reckoning] [+5·(N-1) neighbor-comms] + 5k history
        obs_dim = (9
                   + (3 if self.dead_reckoning else 0)
                   + (5 * (self.n_agents - 1) if self.communication else 0)
                   + 5 * self.k)
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

    def _spawn_positions(self):
        '''(N, 3) initial positions for this episode, per `spawn_mode`.

        "random"   — uniform over the whole domain (the training default).
        "origin"   — every agent at (0, 0, 0): the whole swarm dropped at ONE
                     point on the coast corner. Models a single deployment from
                     a boat/pier, and is the hardest case for redundancy: the
                     agents start with identical observations, so any dispersal
                     has to come from the policy (or from the stochastic decode).
        "max_dist" — evenly spread along the two LAND walls at the surface. The
                     land is the West (x=0) and South (y=0) borders, so the
                     deployable coastline is the L-shaped path
                         (L,0) -> (0,0) -> (0,L)
                     of total length 2L. Agent i sits at the centre of the i-th
                     of N equal segments, s_i = (i + 0.5)·2L/N, which is the
                     maximum-separation placement along that path. For N=2 that
                     gives exactly (500,0,0) and (0,500,0) on the 1 km domain.

        z=0 (the surface) for both fixed modes — the agents are thrown in from
        land. Heading stays random in every mode: a deployment does not control
        which way the vehicle is pointing when it hits the water, and it keeps
        some episode-to-episode variability in the otherwise fixed start.
        '''
        Lx, Ly, Lz = self.domain
        n = self.n_agents
        if self.spawn_mode == "random":
            return np.array([[self.np_random.uniform(0.0, Lx),
                              self.np_random.uniform(0.0, Ly),
                              self.np_random.uniform(0.0, Lz)]
                             for _ in range(n)], dtype=float)
        if self.spawn_mode == "origin":
            return np.zeros((n, 3), dtype=float)

        # max_dist: walk the L-shaped coastline, arclength s from (Lx,0) to (0,Ly).
        total = Lx + Ly
        pos = np.zeros((n, 3), dtype=float)
        for i in range(n):
            s = (i + 0.5) * total / n
            if s <= Lx:
                pos[i] = (Lx - s, 0.0, 0.0)       # along the South wall (y=0)
            else:
                pos[i] = (0.0, s - Lx, 0.0)       # along the West wall (x=0)
        return pos

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
            (5·(N-1))-> ONLY if communication: one block PER OTHER AGENT (sorted
                       NEAREST-FIRST) enabling field triangulation:
                       (in_range, rel_x, rel_y, rel_z, S_j - S*). rel_* is the
                       neighbor's position relative to THIS agent, rotated into
                       this agent's body frame; S_j - S* is the neighbor's
                       salinity error. Out-of-comms-range neighbors are zeroed
                       with in_range=0. Still purely relative — no absolute pose.
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
        if self.dead_reckoning or self.communication:
            # Shared world -> body rotation Rot(psi)^T (z shared between frames).
            psi = np.deg2rad(agent.psi)
            cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        if self.dead_reckoning:
            # Displacement from spawn, world -> body.
            dwx, dwy, dwz = agent.pos - self._spawn_pos[i]
            frame += [dwx * cos_psi + dwy * sin_psi,
                      -dwx * sin_psi + dwy * cos_psi,
                      dwz]
        if self.communication:
            # One block per OTHER agent, sorted NEAREST-FIRST. Body-frame relative
            # position + the neighbor's salinity error: with (S_i-S*) already in the
            # frame the actor can form the long-baseline gradient (S_j-S_i)/‖rel‖
            # that a single agent's local gradient cannot see past the ~100-150 m
            # horizon.
            #
            # Distance ordering, not agent-index ordering: the actor is parameter-
            # shared, so under index order slot 0 means "agent 1" to agent 0 but
            # "agent 0" to agent 1 — the same weights would have to decode a
            # different neighbor identity per agent. Sorting by range makes every
            # slot mean the same thing to everyone ("my nearest neighbor", "my
            # second nearest", ...), which is the permutation-invariant encoding
            # the shared actor actually needs. At N=2 there is a single slot, so
            # this is a no-op and earlier 2-agent runs stay comparable.
            others = [(float(np.linalg.norm(other.pos - agent.pos)), j, other)
                      for j, other in enumerate(self.sim.agents) if j != i]
            others.sort(key=lambda t: (t[0], t[1]))   # index breaks distance ties
            for dist, _, other in others:
                if dist <= self.comms_radius:
                    rel = other.pos - agent.pos
                    Sj = self._measure(other)[0]
                    frame += [1.0,
                              rel[0] * cos_psi + rel[1] * sin_psi,
                              -rel[0] * sin_psi + rel[1] * cos_psi,
                              rel[2],
                              Sj - self.target_salinity]
                else:
                    frame += [0.0, 0.0, 0.0, 0.0, 0.0]
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

    def _sample_far_position(self):
        '''One spawn position at least min_spawn_distance metres from the nearest
        success-zone cell. Rejection-samples up to spawn_max_tries uniform points;
        returns the first that clears the threshold, else the farthest one found
        (so a threshold larger than the domain can't hang the reset).'''
        best_pos, best_d = None, -1.0
        for _ in range(self.spawn_max_tries):
            pos = np.array([self.np_random.uniform(0.0, self.domain[0]),
                            self.np_random.uniform(0.0, self.domain[1]),
                            self.np_random.uniform(0.0, self.domain[2])])
            d, _ = self._zone_tree.query((pos[0], pos[1], pos[2]))
            if d >= self.min_spawn_distance:
                return pos
            if d > best_d:
                best_d, best_pos = float(d), pos
        return best_pos

    def _respawn_far_from_zone(self, loader):
        '''Replace all agents with fresh ones whose start is far from the zone.
        Agents are RE-CREATED (not moved) because Agent caches measured/commanded
        state from initialPosition; the zone tree is built on demand when the
        reward isn't the distance potential.'''
        if self._zone_tree is None:
            self._build_zone_index(loader)
        self.sim.remove(*self.sim.agents)
        for i in range(self.n_agents):
            agent = Agent(
                    name=f"A{i + 1:02d}",
                    Dt=self.dt,
                    initialPosition=self._sample_far_position(),
                    initialHeading=self.np_random.uniform(-180.0, 180.0),
                    agent_xml="config/agent.xml",
                    rng=int(self.np_random.integers(2**31))
                )
            self.sim.add(agent)

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

    def _zone_distances(self, live=None):
        '''Distance from every agent to the nearest success-zone cell, (N,).
        Agents excluded by `live` get +inf so they can never win a min().'''
        pos = np.array([a.pos for a in self.sim.agents], dtype=float)
        d = np.full(self.n_agents, np.inf, dtype=np.float64)
        sel = np.ones(self.n_agents, dtype=bool) if live is None else live
        if sel.any():
            d[sel], _ = self._zone_tree.query(pos[sel])
        return d

    def _g(self, d) -> float:
        '''The distance potential 10·(1 − d/diag) shared by Φ_i and the team terms.'''
        return 10.0 * (1.0 - float(d) / self._domain_diag)

    def _team_potentials(self):
        '''(D, Φ_sep), each (N,) and each a function of the JOINT state.

        D_i = G(s) − G(s_-i), with G(s) = g(min_j d_j) the team objective implied
        by end_on_any_success ("the closest agent is what matters"). The
        leave-one-out min of a scalar set is the runner-up for the current
        leader and the leader for everybody else, so D is ZERO for every agent
        except the closest one, for which it equals its margin over the
        runner-up. Used as a potential, so γD(s')−D(s) preserves invariance.
        The incentive: an agent trailing a teammate contributes nothing to
        min_j d_j, earns nothing, and can only earn by leading somewhere the
        others are not — division of labour as a gradient, not a heuristic.

        Φ_sep,i = 10·min(d_i^NN / ℓ, 1): the agent's own distance to its nearest
        live teammate, saturating at ℓ = separation_scale. The saturation is
        load-bearing — an unbounded dispersion term is maximised by fleeing to
        opposite corners, whereas past one gradient-correlation length there is
        no more independent information to gain.

        Agents that have already succeeded are excluded from both terms: a frozen
        agent sits at d≈0 and would otherwise be the permanent leader (zeroing
        every other agent's D forever when end_on_any_success is False), and it
        no longer samples the field, so it should not repel anyone either.
        '''
        D = np.zeros(self.n_agents, dtype=np.float64)
        sep = np.zeros(self.n_agents, dtype=np.float64)
        if not self.team_reward:
            return D, sep
        live = ~self._success
        if int(live.sum()) < 2:
            # With fewer than two live agents there is no team: no runner-up to
            # be credited against, and no neighbour to be separated from.
            return D, sep

        if self.beta_difference != 0.0:
            d = self._zone_distances(live)
            lead, runner = np.argsort(d)[:2]
            D[lead] = self._g(d[lead]) - self._g(d[runner])

        if self.lambda_separation != 0.0:
            p = np.array([a.pos for a in self.sim.agents], dtype=float)[live]
            dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            np.fill_diagonal(dist, np.inf)
            sep[live] = 10.0 * np.minimum(dist.min(axis=1) / self.separation_scale, 1.0)
        return D, sep

    def _track_coverage(self):
        '''Accumulate each agent's visited voxel (coverage diagnostic only).'''
        if self._visited is None:
            return
        for i, agent in enumerate(self.sim.agents):
            if self._success[i]:
                continue
            self._visited[i].add(tuple(np.floor_divide(agent.pos, self.coverage_cell).astype(np.int64)))

    def _coverage_stats(self) -> dict:
        '''Episode-end coverage summary.

        redundancy = |union of visited voxels| / Σ_i |agent i's visited voxels|.
        1.0 = perfectly disjoint search (ideal division of labour); 1/N = every
        agent swept exactly the same water. This is the metric the difference and
        separation terms are supposed to move, and it is independent of whether
        the episode happened to succeed.
        '''
        if self._visited is None:
            return {}
        per_agent = sum(len(s) for s in self._visited)
        if per_agent == 0:
            return {}
        union = set().union(*self._visited)
        return {"coverage_redundancy": len(union) / per_agent,
                "coverage_cells": float(len(union))}

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
        self._prev_difference = np.zeros(self.n_agents, dtype=np.float64)
        self._prev_separation = np.zeros(self.n_agents, dtype=np.float64)
        self._first_success_step = None
        self._visited = ([set() for _ in range(self.n_agents)]
                         if self.coverage_cell > 0.0 else None)
        # The zone tree is snapshot- AND target-specific: drop last episode's or
        # the on-demand rebuild below silently measures against a stale zone.
        self._zone_tree = None

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

        # Add N agents at the spawn_mode's positions, with random heading.
        spawn_pos = self._spawn_positions()
        for i in range(self.n_agents):
            agent = Agent(
                    name=f"A{i + 1:02d}",
                    Dt=self.dt,
                    initialPosition=spawn_pos[i].copy(),
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

        salinity_at = loader.salinity_at
        if self.target_mode == "tail":
            # S* constrained to a rare tail of the salinity distribution OVER ITS
            # OWN DEPTH PLANE — LOW (below target_percentile %) or HIGH (above
            # 100 - target_percentile %), drawn 50/50 per episode so the policy
            # can't specialize on "rare == fresher". Plane-rarity (not 3D rarity)
            # is what shrinks the success zone: the τ* constraint pins the zone to
            # the plane at z*, and the fields are depth-stratified, so the 3D tail
            # is just "the freshest layer" (measured 2026-07-11: 3D-tail ~14% zone
            # vs plane-tail the intended few %). A tail S* alone is still not
            # enough — at strongly stratified depths the whole plane spread is ~ε,
            # so keep REDRAWING the depth until the candidate's |ΔS|<ε zone is
            # ≤ target_percentile % of the plane (Monte Carlo over 256 points),
            # remembering the smallest-zone candidate as the fallback. Spawn stays
            # uniform (set above).
            zone_cap = self.target_percentile / 100.0
            low_tail = bool(self.np_random.random() < 0.5)
            pct = self.target_percentile if low_tail else 100.0 - self.target_percentile
            best = None  # (zone_frac_estimate, S*, τ*)
            for _ in range(12):  # depth attempts
                z_sel = self.np_random.uniform(0.0, self.domain[2])
                cand_T = compute_turbidity(z_sel)
                xy = self.np_random.uniform([0.0, 0.0], self.domain[:2], size=(256, 2))
                svals = np.array([salinity_at(px, py, z_sel) for px, py in xy])
                thr = float(np.percentile(svals, pct))
                for _ in range(100):  # plane attempts until one lands in the tail
                    x_sel = self.np_random.uniform(0.0, self.domain[0])
                    y_sel = self.np_random.uniform(0.0, self.domain[1])
                    cand_S = salinity_at(x_sel, y_sel, z_sel)
                    if (cand_S > thr) if low_tail else (cand_S < thr):
                        continue  # not in the requested tail
                    zone_est = float(np.mean(np.abs(svals - cand_S) < self.epsilon_salinity))
                    if best is None or zone_est < best[0]:
                        best = (zone_est, cand_S, cand_T)
                    break
                if best is not None and best[0] <= zone_cap:
                    break
            if best is None:  # degenerate fallback (extremely unlikely): plain uniform
                z_sel = self.np_random.uniform(0.0, self.domain[2])
                best = (1.0,
                        salinity_at(self.np_random.uniform(0.0, self.domain[0]),
                                    self.np_random.uniform(0.0, self.domain[1]), z_sel),
                        compute_turbidity(z_sel))
            self.target_salinity = best[1]
            self.target_turbidity = best[2]
        else:
            x_sel = self.np_random.uniform(0.0, self.domain[0])
            y_sel = self.np_random.uniform(0.0, self.domain[1])
            z_sel = self.np_random.uniform(0.0, self.domain[2])
            self.target_salinity = salinity_at(x_sel, y_sel, z_sel)
            self.target_turbidity = compute_turbidity(z_sel)

        # Reward. The difference term is defined on distance-to-zone regardless of
        # which potential the individual term uses, so it needs the tree too.
        if self.reward_potential == "distance" or self.beta_difference != 0.0:
            self._build_zone_index(loader)

        # Distant-start difficulty knob: re-place every agent at least
        # min_spawn_distance metres from the success zone (needs the zone tree,
        # built here on demand if the reward doesn't already use it). Guarded so
        # the default (0.0) path keeps the original uniform spawn byte-for-byte.
        if self.min_spawn_distance > 0.0:
            self._respawn_far_from_zone(loader)
            spawn = self.sim.agents[0].pos
            self.current_salinity = loader.salinity_at(spawn[0], spawn[1], spawn[2])
            self.current_turbidity = compute_turbidity(spawn[2])

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
        # Seed the joint-state potentials so the first step's γΦ(s')−Φ(s) is a
        # true difference and not a one-off Φ(s') windfall.
        self._prev_difference, self._prev_separation = self._team_potentials()
        self._track_coverage()
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

        # Joint-state potentials at s', evaluated ONCE before the per-agent loop
        # so every agent is scored against the same success set (the loop below
        # mutates self._success as agents terminate).
        live_at_entry = ~self._success
        D_next, sep_next = self._team_potentials()

        # Build next obs, reward and success flags per agent
        obs = np.zeros((self.n_agents, self.observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        newly_succeeded = np.zeros(self.n_agents, dtype=bool)
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

            # Every potential is forced to 0 at the agent's own terminal state
            # (Ng et al. require Φ(terminal)=0); truncation keeps the real Φ(s')
            # so the trainer can bootstrap from it.
            phi_next_eff = 0.0 if terminated_i else phi_next
            r = self.alpha_individual * (self.gamma * phi_next_eff - self._prev_potential[i])
            if self.beta_difference != 0.0:
                d_next_eff = 0.0 if terminated_i else D_next[i]
                r += self.beta_difference * (self.gamma * d_next_eff - self._prev_difference[i])
            if self.lambda_separation != 0.0:
                sep_next_eff = 0.0 if terminated_i else sep_next[i]
                r += self.lambda_separation * (self.gamma * sep_next_eff - self._prev_separation[i])
            if terminated_i:
                self._success[i] = True
                newly_succeeded[i] = True
                if self._first_success_step is None:
                    self._first_success_step = self.t_step
                if not self.shared_success_bonus:
                    r += self.success_bonus
            rewards[i] = r
            self._prev_potential[i] = phi_next
        self._prev_difference = D_next
        self._prev_separation = sep_next

        # Cooperative bonus: every agent that was still live at the start of this
        # step is paid for EACH success scored during it, the succeeder included.
        # Without this the game is a race and coordination has nothing to buy.
        if self.shared_success_bonus and newly_succeeded.any():
            rewards[live_at_entry] += self.success_bonus * float(newly_succeeded.sum())

        self._track_coverage()

        terminateds = self._success.copy()
        out_of_time = self.t_step >= self.max_steps
        episode_over = out_of_time or (self.end_on_any_success and bool(terminateds.any()))
        truncateds = np.full(self.n_agents, episode_over) & (~self._success)

        # TIME-LIMIT vs TEAM-TERMINAL. A trainer must bootstrap γV(s') for an agent
        # cut off by a time limit (the episode would have continued) but NOT for one
        # whose episode genuinely ended. Under end_on_any_success the non-succeeding
        # agents are flagged `truncated`, yet a teammate's success is a TERMINAL event
        # for the whole team — there is no future to bootstrap. Bootstrapping it while
        # also paying the shared success bonus double-counts the bonus and makes
        # hanging back strictly more profitable than finding the target. Expose which
        # kind of ending this is so the trainers can tell them apart.
        info = {"global_state": self._build_global_state(),
                "timeout": bool(out_of_time and not terminateds.any())}
        if episode_over or bool(terminateds.all()):
            info.update(self._episode_stats())
        if self.n_agents == 1:
            return (obs[0], float(rewards[0]), bool(terminateds[0]),
                    bool(truncateds[0]), info)
        return obs, rewards, terminateds, truncateds, info

    def _episode_stats(self) -> dict:
        '''Coordination diagnostics, emitted in `info` on the episode's last step.

        time_to_first_success is the headline efficiency metric for the swarm:
        with end_on_any_success the team's job is to make SOMEONE arrive early,
        and that is what N agents should buy over one. NaN on a failed episode,
        so it must be aggregated over successful episodes only (a mean over all
        episodes would silently reward failing fast).
        '''
        stats = {"time_to_first_success": (float(self._first_success_step)
                                           if self._first_success_step is not None
                                           else float("nan"))}
        if self.n_agents > 1:
            pos = np.array([a.pos for a in self.sim.agents], dtype=float)
            dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            np.fill_diagonal(dist, np.inf)
            stats["nn_distance"] = float(dist.min(axis=1).mean())
            stats["spread"] = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
        stats.update(self._coverage_stats())
        return stats

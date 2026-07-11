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

class OceananigansEnv(gym.Env):
    '''
    This class represents the wrapped environment of the simulation. It builds from
    SwarmSwIM and is enclosed with Gymnasium for standardization. Training runs on
    Oceananigans NetCDF fields with the explicit-gradient sensors of
    BaseEnv/PlumesEnv PLUS a k-deep (action, ΔS, Δτ) history: on the
    buoyancy-active filament fields the |S-S*| landscape is multimodal (the
    gradient's basin of attraction covers only ~40% of the plane, measured
    2026-07-11), so a memoryless gradient-follower is structurally insufficient —
    the history is what lets the policy dead-reckon, detect a dead-end filament
    and commit to an escape direction.

    ONE class serves both the single- and the multi-agent scenario; the switch is
    `n_agents` (default 1). N homogeneous agents share the same frozen/dynamic
    NetCDF field and ONE (S*, τ*) target.

    API — single-agent (n_agents == 1), plain Gymnasium, wrapper-compatible:
        reset() -> obs (9+5k,), info
        step(action scalar) -> obs (9+5k,), reward float, terminated bool,
                               truncated bool, info

    API — multi-agent (n_agents > 1), PettingZoo-parallel flattened, as consumed
    by src/multi_agent/ippo.py and mappo.py:
        reset() -> obs (N, 9+5k), info
        step(actions (N,)) -> obs (N, 9+5k), rewards (N,),
                              terminateds (N,), truncateds (N,), info

    In BOTH modes info["global_state"] carries the centralized state (11·N,) for
    a MAPPO critic (see _build_global_state); IPPO and single-agent PPO simply
    ignore it. `local_observation_space` / `global_observation_space` expose the
    per-agent and centralized shapes to the trainers.

    Per-agent observation (9 + 5k,) — pure local-sensor + mission info, no global
    coordinates:
        (3)  u v w      -> body-frame current vector (m/s)
        (3)  gu gv gw   -> body-frame salinity gradient (PSU/m), gw down-positive
        (2)  S-S* τ-τ*  -> target errors in salinity and turbidity
        (1)  depth      -> agent depth (m, positive down)
        (5k) history    -> last k (dx, dy, dz, S-S*, τ-τ*) tuples, oldest first,
                           newest last (== the current errors). The action is
                           stored as its body-frame unit direction, and heading
                           is fixed per episode, so the action history is a
                           dead-reckoned displacement log — currents and depth
                           are NOT stacked (static field: their history adds
                           nothing over the current frame).

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

    Multi-agent episode logic (mirrors MultiAgentBaseEnv): each agent's success
    flag LATCHES — a succeeded agent is frozen (no-op commands, zero reward,
    obs still emitted) — and the episode ends when time runs out OR, with
    end_on_any_success (default), as soon as any agent scores. `truncateds` marks
    the not-succeeded agents when the episode ends.

    Parameters:
        - xml_file -> SwarmSwIM simulation .xml
        - netcdf_file -> Oceananigans NetCDF data: a single file, a glob pattern, a
          directory, or a list of those. A random file is sampled each reset (in
          static mode a random frozen snapshot; in dynamic mode a random time
          window with fields linearly interpolated across snapshots). Each touched
          file keeps a cached FieldLoader (~90 MB of interpolators, two snapshots),
          so keep the set small (~10 files is fine).
        - k -> length of the (action, ΔS, Δτ) observation history (5k obs values)
        - n_agents -> number of agents (1 = single-agent Gym API, >1 = flattened
          PettingZoo-parallel API)
        - v_agent -> agent commanded speed (m/s)
        - max_steps -> maximum env steps per episode before truncation
        - dt -> simulator timestep (s) per env step
        - frame_skip -> sim sub-steps per env step (action held constant)
        - gamma -> RL discount; MUST match the trainer's γ for shaping invariance
        - success_bonus -> sparse reward added on reaching the target
        - static_frame -> NetCDF: freeze one random snapshot per episode (no time
          evolution) vs. dynamic time window
        - min_band_grad -> reject targets whose success band is ~flat (PSU/m)
        - end_on_any_success -> multi-agent only: end the episode on the first
          agent success (2026-06-29 meeting decision); no effect for n_agents=1
        - target_mode -> "random" (default): (S*, τ*) read at a uniform random
          field point — note that a typical S* makes the |ΔS|<ε zone cover
          ~10-20% of the plane at z*, so success rates carry a large luck floor;
          "tail": S* constrained to a tail of the salinity distribution over the
          target's own depth plane — LOW (bottom target_percentile % of
          S(·,·,z*)) or HIGH (top target_percentile %), drawn 50/50 per episode —
          which shrinks the zone to a rare filament and makes the task require
          actual navigation (2026-06-29 meeting scenario). Spawn stays uniform
          in both modes.
        - target_percentile -> tail mode only: tail width in percent — S* below
          this percentile (low side) or above 100 minus it (high side) of the
          salinity values on its depth plane (Monte Carlo estimate)
    '''
    def __init__(self,
                 xml_file: str,
                 netcdf_file: str,
                 k: int = 12,                       # history length of (action, reward)
                 n_agents: int = 1,                 # number of agents
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
                 wall_penalty: float = 0.0,         # per-step reward penalty when an agent is pinned against a domain wall (fraction of frame_skip ticks clamped); 0 disables
                 success_steps_required: int = 1,   # consecutive in-zone steps needed to terminate as success; >1 forces the agent to arrive AND hold (kills single-step luck crossings on turbulent fields)
                 max_cached_loaders: int = 8,       # LRU cap on cached FieldLoaders (~90 MB each) per env instance
                 end_on_any_success: bool = True,   # multi-agent: end the episode on the first success
                 epsilon_salinity: float = 0.3,     # success tolerance on |S - S*| (PSU); size to ~3% of the field's per-snapshot span
                 epsilon_turbidity: float = 0.05,   # success tolerance on |τ - τ*|   TODO: try with epsilon turbidity bound to meters
                 sigma_s: float = 3.0,              # wide shaping-kernel width in S (PSU); size to ~0.3× the field span (3.0 for the ~10 PSU no_buoyancy fields, 1.5 for the ~5 PSU buoyancy_active ones)
                 sigma_tau: float = 0.3,            # wide shaping-kernel width in τ
                 target_mode: str = "random",       # "random" = S* at a uniform random point; "tail" = S* from the low tail of the snapshot's S distribution
                 target_percentile: float = 5.0,    # tail mode: S* below this percentile of the field's salinity values
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
        self.static_frame = static_frame
        self.min_band_grad = min_band_grad
        self.target_min_dist_frac = target_min_dist_frac
        self.wall_penalty = wall_penalty
        self._success_steps_required = int(success_steps_required)
        self.end_on_any_success = end_on_any_success
        if target_mode not in ("random", "tail"):
            raise ValueError(f"target_mode must be 'random' or 'tail', got {target_mode!r}")
        self.target_mode = target_mode
        self.target_percentile = float(target_percentile)

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

        self.action_space = gym.spaces.Discrete(27)
        obs_dim = 9 + 5 * self.k
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
        the proximity potential Φ = reward_func(...) into the shaped reward and
        uses (S, τ) for the in-zone check. When an `action` index is given (a
        step, not a reset, and the agent is not frozen) the agent's history is
        advanced first: the newest row becomes (action direction, S - S*, τ - τ*)
        measured AFTER that action, so the last history row always mirrors the
        current errors in the frame part.

        Observation layout (9 + 5k,):
            (3)     -> body-frame currents (u, v, w)
            (3)     -> body-frame salinity gradient (gu, gv, gw)
            (2)     -> target errors (S - S*, τ - τ*)
            (1)     -> depth
            (5k)    -> history, oldest->newest rows of (dx, dy, dz, S-S*, τ-τ*)
        '''
        S, tau, u, v, w, gu, gv, gw = self._measure(agent)

        potential = reward_func(S, tau, self.target_salinity, self.target_turbidity,
                                sigma_s=self.sigma_s, sigma_tau=self.sigma_tau,
                                eps_s=self.epsilon_salinity, eps_tau=self.epsilon_turbidity)

        dS = S - self.target_salinity
        dT = tau - self.target_turbidity
        if action is not None:
            self._hist[i] = np.roll(self._hist[i], -1, axis=0)
            self._hist[i, -1, :3] = self._action_to_direction[action]
            self._hist[i, -1, 3] = dS
            self._hist[i, -1, 4] = dT

        obs = np.concatenate([
                np.array([u, v, w, gu, gv, gw, dS, dT, agent.pos[2]], dtype=np.float32),
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

    def _global_state_from_obs(self, obs) -> np.ndarray:
        '''Same layout as _build_global_state, assembled from the per-agent obs
        rows already built this step (obs = [u v w gu gv gw S-S* τ-τ* depth]) —
        avoids re-querying the field interpolators a second time per step.'''
        parts = []
        for i, agent in enumerate(self.sim.agents):
            o = obs[i]
            parts.extend([o[6], o[7],                    # S - S*, τ - τ*
                          o[0], o[1], o[2],              # u v w
                          o[3], o[4], o[5],              # gu gv gw
                          agent.pos[0], agent.pos[1], agent.pos[2]])
        return np.array(parts, dtype=np.float32)

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
        '''
        Method that initializes an environment.

        Parameters:
            - seed (int)
            - options (dict | None) — Gymnasium passes this through wrappers; unused here.

        Output:
            - obs: (9+5k,) for n_agents == 1, (N, 9+5k) otherwise
            - info (dict) with "global_state" (11·N,)
        '''
        super().reset(seed=seed)

        # Reset per-agent episode state — without this, a successful previous
        # episode leaves _in_zone_steps at the threshold (or _success latched)
        # and the next episode could terminate immediately.
        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)

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

        # Target selection (ONE shared target for the whole swarm): sample a point
        # that is (a) at least target_min_dist_frac of the domain diagonal from
        # agent 0's spawn (0 -> no distance check, so targets may land near the
        # spawn and episode difficulty varies), (b) has an (S, τ) couple
        # meaningfully different from that spawn's, and (c) sits on a non-flat
        # salinity band so a local-sensing agent has a gradient to home on. (S*, τ*)
        # are read AT that point, so the success zone exists there by construction.
        # The target is ALWAYS assigned to the accepted candidate — the previous logic
        # broke out without assigning, leaving S*≈0 (unreachable).
        # In tail mode a candidate must additionally sit in the LOW tail of the
        # salinity distribution OVER ITS OWN DEPTH PLANE (below the
        # target_percentile-th percentile of S(·, ·, z_sel), Monte Carlo): the τ*
        # constraint pins the success zone to the plane at z*, so plane-rarity is
        # what shrinks it. Rarity in the full 3D distribution does NOT work here —
        # the fields are depth-stratified, so the 3D low tail is just "the
        # freshest layer", within which |ΔS|<ε can cover MORE of the plane
        # (measured 2026-07-11: 3D-tail targets gave ~14% zone vs ~8% random).
        # A typical (random-mode) S* makes |ΔS|<ε hold on ~10-20% of the plane —
        # a luck floor that dominates success rates.
        grad_at = loader.salinity_gradient_at
        spawn = self.sim.agents[0].pos
        spawn_S = salinity_at(spawn[0], spawn[1], spawn[2])
        spawn_T = compute_turbidity(spawn[2])
        self.current_salinity = spawn_S
        self.current_turbidity = spawn_T
        min_dist = self.target_min_dist_frac * float(np.linalg.norm(self.domain))

        def _accept(x_sel, y_sel, z_sel, cand_S, cand_T):
            '''Shared target checks: far enough from spawn, non-trivial, non-flat.'''
            if np.linalg.norm(np.array([x_sel, y_sel, z_sel]) - spawn) < min_dist:
                return False
            # Reject trivial targets: both S and τ already within reach of spawn.
            if (abs(cand_S - spawn_S) <= 2 * self.epsilon_salinity
                    and abs(cand_T - spawn_T) <= 2 * self.epsilon_turbidity):
                return False
            # Reject flat-band targets with no local directional signal.
            if self.min_band_grad > 0.0:
                gx, gy, _ = grad_at(x_sel, y_sel, z_sel)
                if np.hypot(gx, gy) < self.min_band_grad:
                    return False
            return True

        # Fallback: if no candidate qualifies (extremely unlikely on these
        # fields), keep spawn (S, τ) so the couple is at least valid and reachable.
        self.target_salinity = spawn_S
        self.target_turbidity = spawn_T
        if self.target_mode == "tail":
            # A tail S* alone is not enough: at strongly stratified depths the
            # plane's whole S spread is comparable to ε, so |ΔS|<ε can cover much
            # of the plane even around a 5th-percentile S*. The plane sample gives
            # the achieved zone fraction for free, so keep drawing depths until a
            # tail candidate's zone is ≤ ~target_percentile% of the plane,
            # remembering the smallest-zone candidate as the fallback.
            # The tail SIDE is drawn 50/50 per episode: LOW (below
            # target_percentile) or HIGH (above 100 - target_percentile), so the
            # policy can't specialize on "rare always means fresher".
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
                for _ in range(100):
                    x_sel = self.np_random.uniform(0.0, self.domain[0])
                    y_sel = self.np_random.uniform(0.0, self.domain[1])
                    cand_S = salinity_at(x_sel, y_sel, z_sel)
                    if (cand_S > thr) if low_tail else (cand_S < thr):
                        continue
                    if _accept(x_sel, y_sel, z_sel, cand_S, cand_T):
                        zone_est = float(np.mean(np.abs(svals - cand_S)
                                                 < self.epsilon_salinity))
                        if best is None or zone_est < best[0]:
                            best = (zone_est, cand_S, cand_T)
                        break
                if best is not None and best[0] <= zone_cap:
                    break
            if best is not None:
                self.target_salinity = best[1]
                self.target_turbidity = best[2]
        else:
            for _ in range(50):
                x_sel = self.np_random.uniform(0.0, self.domain[0])
                y_sel = self.np_random.uniform(0.0, self.domain[1])
                z_sel = self.np_random.uniform(0.0, self.domain[2])
                cand_S = salinity_at(x_sel, y_sel, z_sel)
                cand_T = compute_turbidity(z_sel)
                if _accept(x_sel, y_sel, z_sel, cand_S, cand_T):
                    self.target_salinity = cand_S
                    self.target_turbidity = cand_T
                    break

        self.t_step = 0

        # Observation history, per agent: k rows of (dx, dy, dz, S-S*, τ-τ*).
        # Pre-filled as if the agent had been sitting still at its spawn (zero
        # action, spawn errors) — zeros in the ΔS/Δτ columns would fake a
        # "was at the target" signal.
        self._hist = np.zeros((self.n_agents, self.k, 5), dtype=np.float32)
        for i, agent in enumerate(self.sim.agents):
            self._hist[i, :, 3] = (salinity_at(agent.pos[0], agent.pos[1], agent.pos[2])
                                   - self.target_salinity)
            self._hist[i, :, 4] = compute_turbidity(agent.pos[2]) - self.target_turbidity

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
        '''
        Method that given the action(s) from the policy updates the environment.

        Parameters:
            - action: scalar (n_agents == 1) or array-like (n_agents,) of
              discrete action indices

        Output (n_agents == 1):
            - s' (9+5k,), reward float, terminated bool, truncated bool, info
        Output (n_agents > 1):
            - s' (N, 9+5k), rewards (N,), terminateds (N,), truncateds (N,), info
        In both modes info["global_state"] carries the centralized (11·N,) state.
        '''
        actions = np.atleast_1d(np.asarray(action)).astype(np.int64)

        # Translate actions into movement; a succeeded (frozen) agent no-ops.
        # The action triple is a BODY-frame direction (surge, sway, heave):
        # config/agent.xml uses heading_control='yawrate' and no yaw is ever
        # commanded, so psi stays FIXED at its random initial value for the whole
        # episode and SwarmSwIM maps (surge, sway) to world coordinates via
        # Rot(psi). The observation rotates currents/gradient into the same body
        # frame (see _measure), so sensing and actuation share one coherent
        # frame. (A previous version also set cmd_heading here expecting the
        # heading to auto-track the motion direction — silently ignored in
        # yawrate mode.)
        for i, agent in enumerate(self.sim.agents):
            if self._success[i]:
                agent.cmd_local_vel = np.array([0.0, 0.0])
                agent.cmd_heave = 0.0
                continue
            mov = self._action_to_direction[actions[i]]
            agent.cmd_local_vel = np.array([mov[0]*self.v, mov[1]*self.v])  # surge (x) and sway (y)
            agent.cmd_heave = mov[2]*self.v                                 # heave (z)

        # Doing the step in the sim
        # NOTE: reward is sampled only at the final sub-step (only-last), not summed across
        # the frame_skip ticks. This preserves the reward scale (and the meaning of the
        # +10 success bonus) when sweeping frame_skip, but PPO loses the integrated
        # signal of any high-reward region the agent passed through mid-skip. Worth
        # revisiting later — compare against summed-reward aggregation (paper convention,
        # Andrychowicz et al. 2021 §3.6) once a frame_skip ablation has been run.
        clamped_ticks = np.zeros(self.n_agents, dtype=np.int64)
        for _ in range(self.frame_skip):
            self.sim.tick()
            # Keep the agents inside the domain box: motion commands and currents
            # would otherwise push them above the surface (z < 0), below the seabed
            # (z > domain depth) or out of the horizontal extent. Clamped every
            # tick so field queries never run from out-of-bounds positions. A tick
            # that actually needed clamping means the agent was driving into a wall
            # (wasted motion) — counted so step() can penalize wall-pinning.
            for i, agent in enumerate(self.sim.agents):
                clipped = np.clip(agent.pos, [0.0, 0.0, 0.0], self.domain)
                if not np.array_equal(clipped, agent.pos):
                    clamped_ticks[i] += 1
                agent.pos[:] = clipped
        self.t_step += 1
        # Fraction of the frame_skip each agent spent pinned against a wall (0..1).
        wall_frac = clamped_ticks / self.frame_skip

        # Build next obs, reward and success flags per agent. Success requires an
        # agent to STAY in the zone for `_success_steps_required` consecutive
        # steps, not just clip through it once: on a turbulent field a single
        # in-zone step is achievable by stochastic luck (inflates train success,
        # evaporates as entropy drops, and never reproduces under a greedy
        # rollout). Holding is what we actually want.
        obs = np.zeros((self.n_agents, self.observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        for i, agent in enumerate(self.sim.agents):
            # Frozen agents pass action=None: their forced no-op must not keep
            # writing history rows.
            next_obs_i, phi_next, S, tau = self._build_state(
                i, agent, None if self._success[i] else actions[i])
            obs[i] = next_obs_i
            if i == 0:
                self.current_salinity = S
                self.current_turbidity = tau
            if self._success[i]:
                # Frozen: re-emit obs, zero reward, no state updates.
                continue

            if self._is_in_zone(S, tau):
                self._in_zone_steps[i] += 1
            else:
                self._in_zone_steps[i] = 0
            terminated_i = self._in_zone_steps[i] >= self._success_steps_required

            # Potential-based reward shaping (Ng et al. 1999): r = r_sparse + γΦ(s') − Φ(s).
            # Φ at a true terminal (success) state is 0; truncation is NOT terminal (the
            # agent bootstraps), so it keeps the real Φ(s') — using 0 there would break
            # policy invariance. The dense shaping telescopes to a policy-independent
            # constant, so it guides without creating an incentive to loiter and avoid
            # finishing the way a raw positive dense reward did.
            phi_next_eff = 0.0 if terminated_i else phi_next
            r = self.gamma * phi_next_eff - self._prev_potential[i]
            if terminated_i:
                r += self.success_bonus
                self._success[i] = True
            # Wall-pinning penalty (breaks strict shaping invariance by design): a small
            # cost proportional to the fraction of the step spent clamped against a
            # domain boundary, to discourage the degenerate "drive into a wall and stall"
            # local optimum. 0 by default (opt-in via the wall_penalty constructor arg).
            if self.wall_penalty > 0.0:
                r -= self.wall_penalty * wall_frac[i]
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

import gymnasium as gym
from gymnasium import spaces
from abc import abstractmethod
from SwarmSwIM import Simulator, Agent, sim_functions
import numpy as np
import itertools
from src.models.salinity import (
    compute_salinity_gaussian,
    compute_salinity_gradient_gaussian,
    gaussian_field_norm,
)
from src.models.turbidity import compute_turbidity
from src.models.reward import reward_func
from src.utils.sources import random_sources

class PlumesEnv(gym.Env):
    def __init__(self,
                 xml_file: str,
                 k: int = 12,
                 v_agent: float = 1.0,
                 max_steps: int = 3600,            
                 dt: float = 0.1,
                 frame_skip: int = 10,
                 domain = (1000.0, 1000.0, 100.0),
                 gamma: float = 0.999,
                 success_bonus: float = 10.0,
                 eddy_length_scale: float = 300.0,  # vortex eddy radius [m] (used by randomize_currents)
                 salinity_sigma_h: float = 300.0,   # field horizontal std [m] (domain-scale -> navigable gradient)
                 salinity_sigma_v: float = 40.0,    # field vertical std [m] (< the 40 m column -> vertical gradient)
                 salinity_span: float = 10.0,       # field span [PSU] across the domain (max - min)
                 n_sources: int = 30,               # per episode a random min_sources..n_sources land-anchored sources
                 min_sources: int = 10,              # lower bound on the per-episode source count (fills the flat far-field)
                 field_grid_n: int = 32,            # grid resolution used to normalize the field to span
                 min_band_grad: float = 0.004,      # reject targets whose success band is ~flat (PSU/m); <=0 disables
                 ):
        self.xml_file = xml_file
        self.k = k
        self.v_agent = v_agent
        self.max_steps = max_steps
        self.dt = dt
        self.frame_skip = frame_skip
        self.domain = domain
        self.gamma = gamma
        self.success_bonus = success_bonus
        self.eddy_length_scale = eddy_length_scale
        self.salinity_sigma_h = salinity_sigma_h
        self.salinity_sigma_v = salinity_sigma_v
        self.salinity_span = salinity_span
        self.n_sources = n_sources
        self.min_sources = min_sources
        self.field_grid_n = field_grid_n
        self.min_band_grad = min_band_grad

        # Per-episode salinity field (source list + blob centers/weights + span
        # normalization); set in randomize_salinity_field().
        self._sources = None
        self._salinity_centers = None
        self._salinity_weights = None
        self._salinity_raw_min = None
        self._salinity_raw_max = None

        # Success zone: |ΔS| and |Δτ| below these of the target couple.
        self.epsilon_salinity = 0.3
        self.epsilon_turbidity = 0.05

        self._prev_potential = 0.0

        self.action_space = gym.spaces.Discrete(27)
        # obs = 9 baseline sensor frame + 5k history (same layout as OceananigansEnv)
        obs_dim = 9 + 5 * self.k
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._action_to_direction = self._build_action_table()

        self.sim = None

    def randomize_currents(self):
        # Randomize currents
            # The 5 components below form the 2D surface current; EkmanSpiral then
            # rotates and decays them with depth to produce the 3D field used in calculate_currents().

            # 1. Uniform background (tidal / geostrophic drift)
            bg_speed = self.np_random.uniform(0.0, 0.3)
            bg_angle = self.np_random.uniform(0.0, 2 * np.pi)
            self.sim.environment['uniform_current'] = np.array([
                bg_speed * np.cos(bg_angle),
                bg_speed * np.sin(bg_angle),
                0.0,
            ])
            self.sim.environment['is_uniform_current'] = True

            # 2. Vortex field (mesoscale eddies / spatial mixing).
            # domain_size ties the periodic tiling to the domain; eddy_length_scale
            # sizes the swirls so they are resolved (not point-like).
            # NOTE: VortexField's `intensity` is an ANGULAR rate, not a speed — the
            # peak eddy speed is ~intensity * length_scale / 2**0.75. So pick a
            # target peak speed and back out intensity; otherwise scaling
            # length_scale up blows the current up (0.3 * 300 / 1.68 ≈ 54 m/s).
            peak_eddy_speed = self.np_random.uniform(0.0, 0.3)   # [m/s]
            self.sim.vortex_field = sim_functions.VortexField(
                density=10,
                intensity=peak_eddy_speed * 2 ** 0.75 / self.eddy_length_scale,
                rng=np.random.default_rng(int(self.np_random.integers(0, 2**31))),
                domain_size=self.domain[0],
                length_scale=self.eddy_length_scale,
            )
            self.sim.environment['is_vortex_currents'] = True

            # 3. Turbulent noise (small-scale temporal fluctuations)
            self.sim.turbolent_noise = sim_functions.TimeNoise(
                time=self.sim.time,
                freq=self.np_random.uniform(0.1, 1.0),
                intensity=self.np_random.uniform(0.0, 0.2),
                rng=np.random.default_rng(int(self.np_random.integers(0, 2**31))),
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
                'wavelength': self.np_random.uniform(500.0, 5000.0),   # km-scale wavelengths
                'wavespeed':  self.np_random.uniform(0.1, 1.0),
                'direction':  self.np_random.uniform(0.0, 360.0),
                'shift':      self.np_random.uniform(0.0, 2 * np.pi),
            }]
            self.sim.environment['is_local_waves'] = True

            # 6. EkmanSpiral — transformer that rotates/decays the 2D surface current with depth.
            # wind_speed adds an additional wind-driven term on top of the surface stack above.
            self.sim.current_3d = sim_functions.EkmanSpiral(
                wind_speed=self.np_random.uniform(0.0, 10.0),
                wind_direction=self.np_random.uniform(0.0, 360.0),
                latitude=24.5,
                eddy_viscosity=0.05,
            )
            self.sim.environment['is_current_3d'] = True
            self.sim.environment['current_3d_model'] = 'ekman'

    def randomize_salinity_field(self):
        '''Draw 2..n_sources pollution sources anchored to the west/south land
        borders (random_sources) and use them as the Gaussian blob centers, each
        weighted by its emission rate Q. The summed field is then span-normalized
        exactly like BaseEnv — only the center placement (coastline vs anywhere)
        differs.'''
        low = min(self.min_sources, self.n_sources)
        n = int(self.np_random.integers(low, self.n_sources + 1))
        self._sources = random_sources(
            rng=self.np_random, n_sources=n,
            min_x=0.0, max_x=self.domain[0],
            min_y=0.0, max_y=self.domain[1],
            min_depth=0.0, max_depth=self.domain[2],
            min_q=0.6, max_q=1.0,   # relative blob strengths (span-normalized away)
        )
        self._salinity_centers = np.array(
            [(s["x"], s["y"], s["depth"]) for s in self._sources], dtype=float)
        self._salinity_weights = np.array(
            [s["Q"] for s in self._sources], dtype=float)

        self._salinity_raw_min, self._salinity_raw_max = gaussian_field_norm(
            self._salinity_centers, self._salinity_weights,
            self.salinity_sigma_h, self.salinity_sigma_v, self.domain, self.field_grid_n,
        )

    def _salinity_at(self, x, y, z):
        return compute_salinity_gaussian(
            x, y, z,
            centers=self._salinity_centers, weights=self._salinity_weights,
            sigma_h=self.salinity_sigma_h, sigma_v=self.salinity_sigma_v,
            span=self.salinity_span,
            raw_min=self._salinity_raw_min, raw_max=self._salinity_raw_max,
        )

    def _salinity_grad_at(self, x, y, z):
        return compute_salinity_gradient_gaussian(
            x, y, z,
            centers=self._salinity_centers, weights=self._salinity_weights,
            sigma_h=self.salinity_sigma_h, sigma_v=self.salinity_sigma_v,
            span=self.salinity_span,
            raw_min=self._salinity_raw_min, raw_max=self._salinity_raw_max,
        )

    def _measure(self, agent):
        '''Salinity, turbidity, body-frame currents and body-frame salinity
        gradient at the agent's position — the single source of truth for
        _build_state and the in-zone check.'''
        x, y, z = agent.pos[0], agent.pos[1], agent.pos[2]
        S = self._salinity_at(x, y, z)
        tau = compute_turbidity(z)
        currents = self.sim.depth_current_at(agent)
        psi = np.deg2rad(agent.psi)
        u = currents[0] * np.cos(psi) + currents[1] * np.sin(psi)
        v = currents[0] * np.sin(psi) - currents[1] * np.cos(psi)
        w = currents[2]
        # Salinity gradient rotated into the body frame (same convention as the
        # currents) so the observation stays free of absolute position/heading.
        gx, gy, gz = self._salinity_grad_at(x, y, z)
        gu = gx * np.cos(psi) + gy * np.sin(psi)
        gv = gx * np.sin(psi) - gy * np.cos(psi)
        gw = gz
        return S, tau, u, v, w, gu, gv, gw

    def _build_action_table(self) -> np.array:
        '''Returns an array of action->(dx,dy,dz) normalized.'''
        table = list(itertools.product([-1, 0, 1], repeat=3))
        norms = np.linalg.norm(table, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return table / norms

    def _build_state(self, agent, action=None) -> tuple[np.ndarray, float]:
        '''
        Returns the observation and the proximity potential Φ(s) = reward_func(...).
        The caller turns Φ into the shaped reward.

        When an `action` index is given (a step, not a reset) the history is
        advanced first, so the newest row holds that action's direction together
        with the errors measured AFTER it — i.e. the last history row always
        mirrors the (S - S*, τ - τ*) pair in the frame part.

        Observation layout (9 + 5k,):
            (3)     -> body-frame currents (u, v, w)
            (3)     -> body-frame salinity gradient (gu, gv, gw)
            (2)     -> target errors (S - S*, τ - τ*)
            (1)     -> depth
            (5k)    -> history, oldest->newest rows of (dx, dy, dz, S-S*, τ-τ*)
        '''
        # Salinity, turbidity, body-frame currents and gradient come from _measure
        # (single source of truth, shared with the in-zone check / external tooling).
        new_salinity, new_turbidity, u, v, w, gu, gv, gw = self._measure(agent)

        potential = self._potential_at(agent, new_salinity, new_turbidity)

        self.current_salinity = new_salinity
        self.current_turbidity = new_turbidity

        dS = new_salinity - self.target_salinity
        dT = new_turbidity - self.target_turbidity
        if action is not None and self.k > 0:
            self._hist = np.roll(self._hist, -1, axis=0)
            self._hist[-1, :3] = self._action_to_direction[action]
            self._hist[-1, 3] = dS
            self._hist[-1, 4] = dT

        frame = np.array([
            u, v, w,
            gu, gv, gw,
            dS, dT,
            agent.pos[2],
        ], dtype=np.float32)

        return np.concatenate([frame, self._hist.reshape(-1)]), potential

    def _zone_reachable(self, n_xy: int = 64, n_z_band: int = 5) -> bool:
        '''True if some domain point satisfies both |S - S*| < eps_S and
        |tau - tau*| < eps_tau, i.e. the episode has a target zone at all.

        Turbidity is depth-only, so the tau condition pins a depth band: find it
        by fine 1D sampling (no Beer-Lambert inversion needed), then test the
        salinity condition on an xy-grid at a few depths inside the band.'''
        zs = np.linspace(0.0, self.domain[2], 512)
        band = zs[np.abs(compute_turbidity(zs) - self.target_turbidity) < self.epsilon_turbidity]
        if band.size == 0:
            return False
        z_levels = band[np.linspace(0, band.size - 1, min(n_z_band, band.size)).astype(int)]
        xs = np.linspace(0.0, self.domain[0], n_xy)
        ys = np.linspace(0.0, self.domain[1], n_xy)
        X, Y, Z = np.meshgrid(xs, ys, z_levels, indexing="ij")
        S = self._salinity_at(X, Y, Z)
        return bool(np.any(np.abs(S - self.target_salinity) < self.epsilon_salinity))

    def _band_grad_ok(self, n_xy: int = 64, n_z_band: int = 5) -> bool:
        '''True if the success band carries a usable horizontal salinity gradient:
        the median |∇_xy S| over band points is >= self.min_band_grad.

        Because the sources are anchored to the west/south borders, the field ramps
        off the SW corner and the whole NE far-field is nearly flat. A target that
        lands there has a valid zone (_zone_reachable) yet no local directional
        signal, so a local-sensing agent can only random-walk to it -> wander /
        timeout. This guard rejects those ill-posed targets at reset. Mirrors the
        band-finding in _zone_reachable. min_band_grad <= 0 disables the check.'''
        if self.min_band_grad <= 0.0:
            return True
        zs = np.linspace(0.0, self.domain[2], 512)
        band = zs[np.abs(compute_turbidity(zs) - self.target_turbidity) < self.epsilon_turbidity]
        if band.size == 0:
            return False
        z_levels = band[np.linspace(0, band.size - 1, min(n_z_band, band.size)).astype(int)]
        xs = np.linspace(0.0, self.domain[0], n_xy)
        ys = np.linspace(0.0, self.domain[1], n_xy)
        X, Y, Z = np.meshgrid(xs, ys, z_levels, indexing="ij")
        S = self._salinity_at(X, Y, Z)
        mask = np.abs(S - self.target_salinity) < self.epsilon_salinity
        if not np.any(mask):
            return False
        gx, gy, _ = self._salinity_grad_at(X[mask], Y[mask], Z[mask])
        return bool(np.median(np.sqrt(gx ** 2 + gy ** 2)) >= self.min_band_grad)

    def _potential_at(self, agent, S, tau) -> float:
        '''Shaping potential Phi(s): the multi-scale Gaussian over the MEASUREMENT
        error, i.e. a function of exactly the two quantities the agent senses.

        eps_* MUST be the env's success tolerances: reward_func sizes its two
        narrow kernels at 5x and 1.5x eps, so leaving them at the function
        defaults (0.1 / 0.01) builds the precision endgame around a box 3x (S)
        and 5x (tau) tighter than the one _is_in_zone actually tests — the sharp
        kernel then reads ~0 at the success boundary, exactly where the agent
        needs the gradient.

        KNOWN WEAKNESS (measured 2026-08-06): this potential saturates. One 1 m
        step toward the zone moves Phi by 0.033 within 50 m of it but only 0.0003
        beyond 400 m, and >50% of the domain sits beyond 200 m — so over half the
        domain the reward carries no directional information and the agent must
        random-walk into the basin before shaping helps. A distance-to-zone
        potential 10*(1 - d/diag) was tried as a replacement and was clearly
        WORSE (success decayed 0.34 -> 0.03 over 2M steps): d is not in the
        observation, so the critic cannot predict its own shaping and the
        advantage becomes noise. Oceananigans needs that potential because its
        TURBULENT field makes the error potential non-monotone; this analytic
        Gaussian field has no such local optima. Any future fix should stay a
        function of (S - S*, tau - tau*) and attack the saturation instead —
        e.g. a form that is linear rather than exponential in the errors.'''
        return reward_func(S, tau, self.target_salinity, self.target_turbidity,
                           eps_s=self.epsilon_salinity, eps_tau=self.epsilon_turbidity)

    def _is_in_zone(self) -> bool:
        '''True when measured (S, tau) lie within epsilon of the target couple.'''
        return (
            abs(self.current_salinity - self.target_salinity) < self.epsilon_salinity
            and abs(self.current_turbidity - self.target_turbidity) < self.epsilon_turbidity
        )

    @abstractmethod
    def reset(self, seed=None, options=None):
        """
        Env is initialized random at each reset:
            - agent position
            - salinity field
            - currents
        """
        super().reset(seed=seed)

        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.xml_file)
        self.sim.remove(*self.sim.agents)   # drop ALL xml-defined agents so the
                                            # randomly initialized one is agents[0]

        # Add agent randomly initialized
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

        # Randomize current field
        self.randomize_currents()

        # Randomize the source-based salinity field and target, resampling both
        # until the episode actually has a target zone (_zone_reachable). The
        # target is a point far enough from the spawn that the agent must navigate
        # the field gradient to reach it (target == spawn would be trivial);
        # (S*, tau*) are sampled at that point, so the zone exists at it by
        # construction — the grid check guards against degenerate (vanishingly
        # small) zones.
        spawn = self.sim.agents[0].pos
        min_dist = 0.3 * float(np.linalg.norm(self.domain))
        dom = np.array(self.domain, dtype=float)
        for _ in range(20):
            self.randomize_salinity_field()
            tgt = self.np_random.uniform(0.0, 1.0, size=3) * dom
            for _ in range(100):
                if np.linalg.norm(tgt - spawn) >= min_dist:
                    break
                tgt = self.np_random.uniform(0.0, 1.0, size=3) * dom
            self.target_salinity = self._salinity_at(tgt[0], tgt[1], tgt[2])
            self.target_turbidity = compute_turbidity(tgt[2])
            if self._zone_reachable() and self._band_grad_ok():
                break
        else:
            # for...else: only fires when the loop was never broken out of, i.e.
            # all 20 resamples failed. Previously this `else` was bound to the
            # `if` above, so the FIRST rejected target raised instead of
            # resampling — which made single-agent resets fail outright.
            raise RuntimeError(
                "reset(): no reachable target zone after 20 field/target resamples"
            )

        # Observation history: rows (dx, dy, dz, S-S*, τ-τ*), oldest->newest.
        # The error columns are SEEDED with the spawn measurement rather than
        # left at zero, so an empty buffer never reads as "the agent just
        # measured a perfect match"; the direction columns stay 0 (no action yet).
        agent0 = self.sim.agents[0]
        S0, tau0 = self._measure(agent0)[:2]
        self._hist = np.zeros((self.k, 5), dtype=np.float32)
        self._hist[:, 3] = S0 - self.target_salinity
        self._hist[:, 4] = tau0 - self.target_turbidity
        self.t_step = 0

        obs, phi0 = self._build_state(self.sim.agents[0])
        self._prev_potential = phi0
        return obs, {}

    @abstractmethod
    def step(self, action):
        # execute action
        mov = self._action_to_direction[action]
        agent = self.sim.agents[0]
        agent.cmd_local_vel = np.array([mov[0]*self.v_agent, mov[1]*self.v_agent])
        agent.cmd_heave = mov[2]*self.v_agent
        # No heading command: config/agent.xml uses heading_control="yawrate",
        # whose update_heading branch integrates cmd_yawrate and never reads
        # cmd_heading — so setting it was a silent no-op. (It was also wrong on
        # its own terms: arctan2(mov[0], mov[1]) swaps the arguments and returns
        # 0 for every pure-vertical and no-op action.) Heading therefore stays at
        # its random spawn value; that is fine, because the observation is
        # expressed in the body frame, so the policy sees a self-consistent
        # frame either way.

        for _ in range(self.frame_skip):
             self.sim.tick()

             # Check if agent is out of domain
             agent.pos[0] = np.clip(agent.pos[0], 0.0, self.domain[0])
             agent.pos[1] = np.clip(agent.pos[1], 0.0, self.domain[1])
             agent.pos[2] = np.clip(agent.pos[2], 0.0, self.domain[2])

        self.t_step += 1

        # next state and potential (reward shaped)
        next_obs, phi_next = self._build_state(agent, action)

        # truncation and termination checks
        truncated = (self.t_step >= self.max_steps)
        terminated = self._is_in_zone()

        phi_next_eff = 0.0 if terminated else phi_next
        reward = self.gamma * phi_next_eff - self._prev_potential
        if terminated:
             reward += self.success_bonus
        self._prev_potential = phi_next

        return next_obs, reward, terminated, truncated, {}


class MultiAgentPlumesEnv(PlumesEnv):
    '''
    Multi-agent version of PlumesEnv: N homogeneous agents share ONE synthetic
    domain (same randomized currents + source-based salinity field + target).

    It reuses all of PlumesEnv's field machinery (randomize_currents,
    randomize_salinity_field, _salinity_at, _salinity_grad_at, _zone_reachable,
    _measure, _build_action_table) and only re-implements reset/step/state for a
    swarm, exposing the PettingZoo-parallel-flattened API that
    src/multi_agent/ippo.py (and mappo.py) consume:

        reset() -> obs (N, 9+5k), info
        step(actions (N,)) -> obs (N, 9+5k), rewards (N,),
                              terminateds (N,), truncateds (N,), info

    The per-agent LOCAL observation is exactly PlumesEnv's observation — the
    9-dim gradient frame followed by the k-deep history:
        [ u v w (body-frame currents) | gu gv gw (body-frame salinity gradient)
          | S - S* | tau - tau* | depth
          | k rows of (dx, dy, dz, S - S*, tau - tau*), oldest -> newest ]

    info["global_state"] carries a (9N + 2,) centralized state for a MAPPO
    critic; IPPO ignores it (its critic uses the local obs only):
        [ S* tau* | per agent: u v w  gu gv gw  x y z ]

    Reward, per-agent success latching and end_on_any_success mirror
    MultiAgentEnv (src/envs/multi_agent.py): each agent's reward is
    potential-based shaping r = success_bonus·[term] + γΦ(s') − Φ(s), a frozen
    (succeeded) agent no-ops and stops accruing reward, and the episode ends when
    time runs out OR (with end_on_any_success) as soon as any agent scores.
    '''
    def __init__(self,
                 xml_file: str,
                 n_agents: int = 2,
                 z_scale: float = 1.0,           # vertical (heave) speed multiplier; <1 = finer depth control
                                                 # KEEP AT 1.0. Lowering it does not buy precision: once
                                                 # reward_func is given the env tolerance eps_tau=0.05 (see
                                                 # _build_state) its sharp kernel spans 17-27 m of depth, which
                                                 # 1 m/step already resolves 20x over. It only buys cost: the
                                                 # spawn->tau-band vertical gap is mean 26 m / p90 61 m, so
                                                 # z_scale=0.1 turns a ~26-step descent into ~264 (458 on
                                                 # diagonal actions) out of an 1800-step budget.
                 reward_mode: str = "shaped",    # "shaped" (potential-based) | "sparse" (-1/step, +bonus to all on first success)
                 end_on_any_success: bool = True,
                 **base_kwargs):
        super().__init__(xml_file=xml_file, **base_kwargs)
        self.n_agents = n_agents
        self.z_scale = z_scale
        self.reward_mode = reward_mode
        self.end_on_any_success = end_on_any_success
        self._success_steps_required = 1

        # Per-agent LOCAL obs = PlumesEnv's 9-dim gradient observation.
        local_obs_dim = self.observation_space.shape[0]
        self.local_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(local_obs_dim,), dtype=np.float32)
        # Global state (MAPPO critic): targets (2) + 9 per agent (u v w gu gv gw x y z).
        global_obs_dim = 9 * self.n_agents + 2
        self.global_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(global_obs_dim,), dtype=np.float32)

        # Per-agent episode state (allocated for real in reset()).
        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        # Go straight to gym.Env.reset (seeds self.np_random); PlumesEnv.reset is
        # single-agent, so we re-implement the body here reusing its field helpers.
        gym.Env.reset(self, seed=seed)

        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.xml_file)
        # Drop any XML-defined agents; we create our own below.
        self.sim.agents.clear()
        self.sim.history.clear()

        # Create N agents with random position + heading.
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

        # Randomize the current field.
        self.randomize_currents()

        # Randomize the salinity field and target, resampling until the episode
        # actually has a target zone (_zone_reachable). Target is far enough from
        # agent 0's spawn that the swarm must navigate the field (same rule as
        # PlumesEnv's single-agent reset).
        spawn = self.sim.agents[0].pos
        min_dist = 0.3 * float(np.linalg.norm(self.domain))
        dom = np.array(self.domain, dtype=float)
        for _ in range(20):
            self.randomize_salinity_field()
            tgt = self.np_random.uniform(0.0, 1.0, size=3) * dom
            for _ in range(100):
                if np.linalg.norm(tgt - spawn) >= min_dist:
                    break
                tgt = self.np_random.uniform(0.0, 1.0, size=3) * dom
            self.target_salinity = self._salinity_at(tgt[0], tgt[1], tgt[2])
            self.target_turbidity = compute_turbidity(tgt[2])
            if self._zone_reachable() and self._band_grad_ok():
                break
        else:
            raise RuntimeError(
                "reset(): no reachable target zone after 20 field/target resamples")

        # Per-agent episode state.
        self._in_zone_steps = np.zeros(self.n_agents, dtype=np.int64)
        self._success = np.zeros(self.n_agents, dtype=bool)
        self._prev_potential = np.zeros(self.n_agents, dtype=np.float64)
        # Per-agent observation history, rows (dx, dy, dz, S-S*, τ-τ*),
        # oldest->newest. Error columns seeded with each agent's own spawn
        # measurement (see PlumesEnv.reset for why).
        self._hist = np.zeros((self.n_agents, self.k, 5), dtype=np.float32)
        for i, agent in enumerate(self.sim.agents):
            S0, tau0 = self._measure(agent)[:2]
            self._hist[i, :, 3] = S0 - self.target_salinity
            self._hist[i, :, 4] = tau0 - self.target_turbidity
        self.t_step = 0

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
        actions: array-like (n_agents,) of discrete action indices.
        Returns obs (N, 9), rewards (N,), terminateds (N,), truncateds (N,), info.
        '''
        actions = np.asarray(actions).astype(np.int64)

        # 1. Set commands for active agents, advance the shared sim once.
        for i, agent in enumerate(self.sim.agents):
            if self._success[i]:
                agent.cmd_local_vel = np.array([0.0, 0.0])
                agent.cmd_heave = 0.0
                continue
            mov = self._action_to_direction[actions[i]]
            agent.cmd_local_vel = np.array([mov[0] * self.v_agent, mov[1] * self.v_agent])
            agent.cmd_heave = mov[2] * self.v_agent * self.z_scale
            # cmd_heading dropped — silent no-op under heading_control="yawrate".
            # See PlumesEnv.step.

        for _ in range(self.frame_skip):
            self.sim.tick()
            for agent in self.sim.agents:
                agent.pos[0] = np.clip(agent.pos[0], 0.0, self.domain[0])
                agent.pos[1] = np.clip(agent.pos[1], 0.0, self.domain[1])
                agent.pos[2] = np.clip(agent.pos[2], 0.0, self.domain[2])
        self.t_step += 1

        # 2. Build next obs, reward and success flags per agent.
        obs = np.zeros((self.n_agents, self.local_observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        out_of_time = self.t_step >= self.max_steps

        active_at_entry = ~self._success.copy()
        newly_terminated = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            if self._success[i]:
                # Frozen: re-emit obs, zero reward.
                obs[i] = self._build_local_state(i)[0]
                continue
            o, phi_next, S, tau = self._build_local_state(i, actions[i])
            obs[i] = o

            if self._is_in_zone(S, tau):
                self._in_zone_steps[i] += 1
            else:
                self._in_zone_steps[i] = 0
            terminated_i = self._in_zone_steps[i] >= self._success_steps_required

            if self.reward_mode == "sparse":
                rewards[i] = -1.0
            else:
                phi_next_eff = 0.0 if terminated_i else phi_next
                r = self.gamma * phi_next_eff - self._prev_potential[i]
                if terminated_i:
                    r += self.success_bonus
                rewards[i] = r
            if terminated_i:
                self._success[i] = True
                newly_terminated[i] = True
            self._prev_potential[i] = phi_next

        # Sparse mode: pay +success_bonus to all agents active this step once any scores.
        if self.reward_mode == "sparse" and bool(newly_terminated.any()):
            rewards[active_at_entry] = self.success_bonus

        terminateds = self._success.copy()
        episode_over = out_of_time or (self.end_on_any_success and bool(terminateds.any()))
        truncateds = np.full(self.n_agents, episode_over) & (~self._success)

        # TIME-LIMIT vs TEAM-TERMINAL. A trainer must bootstrap γV(s') for an agent
        # cut off by the time limit (the episode would have continued) but NOT for
        # one whose episode genuinely ended. With end_on_any_success the
        # non-succeeding agents are flagged `truncated`, yet a teammate's success is
        # a TERMINAL event for the whole team — there is no future to bootstrap.
        # Expose which kind of ending this is so the trainer can tell them apart.
        info = {"global_state": self._build_global_state(),
                "timeout": bool(out_of_time and not terminateds.any())}
        return obs, rewards, terminateds, truncateds, info

    # ----------------------------------------------------------------- helpers
    def _is_in_zone(self, salinity, turbidity) -> bool:
        '''True when (S, tau) lie within epsilon of the target couple. Overrides
        PlumesEnv's zero-arg version (which reads single-agent shared state).'''
        return (
            abs(salinity - self.target_salinity) < self.epsilon_salinity
            and abs(turbidity - self.target_turbidity) < self.epsilon_turbidity
        )

    def _build_local_state(self, i, action=None):
        '''Per-agent version of PlumesEnv._build_state (no shared state).
        Returns (obs (9+5k,), potential, S, tau) for agent i. When `action` is
        given (a step by a live agent) that agent's history is advanced first;
        a frozen agent is re-observed with action=None so its buffer stops.'''
        agent = self.sim.agents[i]
        S, tau, u, v, w, gu, gv, gw = self._measure(agent)
        potential = self._potential_at(agent, S, tau)
        dS = S - self.target_salinity
        dT = tau - self.target_turbidity
        if action is not None and self.k > 0:
            self._hist[i] = np.roll(self._hist[i], -1, axis=0)
            self._hist[i, -1, :3] = self._action_to_direction[action]
            self._hist[i, -1, 3] = dS
            self._hist[i, -1, 4] = dT
        frame = np.array([
            u, v, w,
            gu, gv, gw,
            dS, dT,
            agent.pos[2],
        ], dtype=np.float32)
        obs = np.concatenate([frame, self._hist[i].reshape(-1)])
        return obs, potential, S, tau

    def _build_global_state(self):
        '''Centralized state (9N + 2,) for the MAPPO critic:
            (2)        target salinity*, turbidity*
            per agent (9): u v w | gu gv gw | x y z '''
        parts = [self.target_salinity, self.target_turbidity]
        for agent in self.sim.agents:
            S, tau, u, v, w, gu, gv, gw = self._measure(agent)
            parts.extend([u, v, w, gu, gv, gw, agent.pos[0], agent.pos[1], agent.pos[2]])
        return np.array(parts, dtype=np.float32)

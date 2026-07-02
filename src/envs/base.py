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
from src.single_agent.reward import reward_func

"""
Env baseline with a (1000,1000,100) domain, synthetic currents and fields 
and no salinity sources.
"""

class BaseEnv(gym.Env):
    def __init__(self,
                 xml_file: str,
                 k: int = 12,
                 v_agent: float = 1.0,
                 max_steps: int = 5120,
                 dt: float = 0.1,
                 frame_skip: int = 10,
                 domain = (1000.0, 1000.0, 100.0),
                 gamma: float = 0.999,
                 success_bonus: float = 10.0,
                 eddy_length_scale: float = 300.0,   # vortex eddy radius [m] (used by randomize_currents)
                 salinity_sigma_h: float = 300.0,    # field horizontal std [m] (domain-scale -> navigable gradient)
                 salinity_sigma_v: float = 40.0,     # field vertical std [m]
                 salinity_span: float = 10.0,        # field span [PSU] across the domain (max - min)
                 n_blobs: int = 3,                   # per episode a random 2..n_blobs Gaussian blobs
                 field_grid_n: int = 32,             # grid resolution used to normalize the field to span
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
        self.n_blobs = n_blobs
        self.field_grid_n = field_grid_n

        # Per-episode salinity field (blob centers/weights + span normalization);
        # set in randomize_salinity_field().
        self._salinity_centers = None
        self._salinity_weights = None
        self._salinity_raw_min = None
        self._salinity_raw_max = None

        # Success zone: |ΔS| and |Δτ| below these of the target couple.
        self.epsilon_salinity = 0.3
        self.epsilon_turbidity = 0.05

        self.t_step = 0
        self._prev_potential = 0.0

        self.action_space = gym.spaces.Discrete(27)
        obs_dim = 9 
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
        n = int(self.np_random.integers(2, self.n_blobs + 1))
        dom = np.array(self.domain, dtype=float)

        self._salinity_centers = self.np_random.uniform(0.15 * dom, 0.85 * dom, size=(n, 3))
        self._salinity_weights = self.np_random.uniform(0.6, 1.0, size=n)

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

        obs_mode="minimal" (9,):
            (3)     -> body-frame currents (u, v, w)
            (3)     -> body-frame salinity gradient (gu, gv, gw)
            (2)     -> target errors (S - S*, τ - τ*)
            (1)     -> depth
        '''
        # Salinity, turbidity, body-frame currents and gradient come from _measure
        # (single source of truth, shared with the in-zone check / external tooling).
        new_salinity, new_turbidity, u, v, w, gu, gv, gw = self._measure(agent)

        potential = reward_func(new_salinity, new_turbidity, self.target_salinity, self.target_turbidity)

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

        #if action is not None:
        #    self.history = np.roll(self.history, -1, axis=0)
        #    self.history[-1] = [action, potential]

        # (S, τ) history rolls on every build (incl. reset): the measurement is
        # available at every observation, so the last row is always the current
        # (S, τ) and earlier rows are the previous k-1 steps.
        #self.st_history = np.roll(self.st_history, -1, axis=0)
        #self.st_history[-1] = [new_salinity, new_turbidity]
    
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

        # Randomize the synthetic salinity field and target, resampling both until
        # the episode actually has a target zone (_zone_reachable). The target is
        # a point far enough from the spawn that the agent must navigate the field
        # gradient to reach it (target == spawn would be trivial); (S*, tau*) are
        # sampled at that point, so the zone exists at it by construction — the
        # grid check guards against degenerate (vanishingly small) zones.
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
            if self._zone_reachable():
                break
        else:
            raise RuntimeError(
                "reset(): no reachable target zone after 20 field/target resamples"
            )

        # Init RL vars
        self.history = np.zeros((self.k, 2), dtype=np.float32)
        self.st_history = np.zeros((self.k, 2), dtype=np.float32)
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
        agent.cmd_heading = np.rad2deg(np.arctan2(mov[0], mov[1]))

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
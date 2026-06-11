from .base import BaseEnv
import glob
import os
import warnings
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SwarmSwIM import Simulator, sim_functions
from ..single_agent.reward import reward_func
import itertools
from src.models.salinity import compute_salinity_analytical, compute_salinity_gradient_analytical
from src.models.turbidity import compute_turbidity
from src.utils.sources import random_sources


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
class SingleAgentEnv(gym.Env):
    '''
    This class represents the wrapped environment of the simulation. It builds from
    SwarmSwIM and is enclosed with Gymnasium for standardization.

    Observation (2k + 11) — pure local-sensor + mission info, no global coordinates:
        [ history (2k) | u v w (body-frame currents) | abs_S abs_τ | S* τ* | field gradients | depth]

    Depth and heading are deliberately excluded: per the project's design rule the
    agent does not know where it is and acts in its local frame. Heading is still
    tracked internally by the simulator so currents can be rotated into body
    frame — the policy just never sees ψ directly.

    Parameters:
        - xml_file -> SwarmSwIM simulation .xml
        - netcdf_file -> optional Oceananigans NetCDF data: a single file, a glob
          pattern, a directory, or a list of those. A random file (and a random
          time window within it, fields linearly interpolated in time across
          snapshots) is sampled each reset. Each touched file keeps a cached
          FieldLoader (~90 MB of interpolators, two snapshots), so keep the set
          small (~10 files is fine).
        - n_sources -> number of pollution sources spawned each reset (on domain borders)
        - k -> history buffer length for (action, reward) pairs
        - v_agent -> agent commanded speed (m/s)
        - max_steps -> maximum env steps per episode before truncation
        - dt -> simulator timestep (s) per env step
        - domain -> (x, y, z) extent of the domain in meters
    '''
    def __init__(self,
                 xml_file: str,
                 netcdf_file: str = None,
                 n_sources: int = 4,
                 k: int = 4,                    # history length of (action, reward)
                 v_agent: float = 1.0,          # agent speed in m/s
                 max_steps: int = 128,         # steps of an episode before truncation
                 dt: float = 0.1,               # seconds per step
                 frame_skip: int = 10,          # sim sub-steps per env step (action held constant)
                 domain = (5000.0, 5000.0, 40.0),   # domain size (0.0-x, 0.0-y, 0.0-z) in meters
                 sigma_h: float = 500.0,        # salinity plume horizontal std [m] (scales with domain)
                 sigma_v: float = 12.0,         # salinity plume vertical std [m]
                 eddy_length_scale: float = 1000.0,  # vortex eddy radius [m] (scales with domain)
                 ):
        super().__init__()

        self.sim_xml = xml_file
        self.netcdf_file = netcdf_file
        self._nc_files = _resolve_nc_files(netcdf_file)
        self.n_sources = n_sources
        self.k = k
        self.v = v_agent
        self.max_steps = max_steps
        self.dt = dt
        self.frame_skip = frame_skip
        self.domain = domain
        self.sigma_h = sigma_h
        self.sigma_v = sigma_v
        self.eddy_length_scale = eddy_length_scale

        self.target_salinity = 0.0
        self.target_turbidity = 0.0
        self.current_salinity = 0.0
        self.current_turbidity = 0.0

        self._in_zone_steps = 0
        self.epsilon_salinity = 0.05     
        self.epsilon_turbidity = 0.05   

        # Action Space
        self.action_space = gym.spaces.Discrete(27) # NOTE: remember that you have to use a relative PoV, not global

        # State/Observation Space: 2k history + 3 body-frame currents + 2 absolute (S, τ) + 2 target (S*, τ*)
        obs_dim = 2*k + 11
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Map actions to movements on the grid
        self._action_to_direction = self._build_action_table()  # scalar -> [dx,dy,dz] normalized

        self.sim = None
        # NetCDF ocean-data loaders, one per file, created lazily and reused
        # across episodes (re-opening files each reset leaks file handles).
        self._loaders = {}
        self.active_netcdf_path = None
        self._warned_short_record = False

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
            if path not in self._loaders:
                self._loaders[path] = FieldLoader(path)
            loader = self._loaders[path]
            self.active_netcdf_path = path
        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.sim_xml,
                             netcdf_file=loader)

        # Randomize agent position and heading (psi in degrees, SwarmSwIM NED convention)
        for agent in self.sim.agents:
            agent.pos[0] = self.np_random.uniform(0.0, self.domain[0])
            agent.pos[1] = self.np_random.uniform(0.0, self.domain[1])
            agent.pos[2] = self.np_random.uniform(0.0, self.domain[2])
            agent.psi = self.np_random.uniform(-180.0, 180.0)

        # NOTE: this implementation does not prevent agent spotting close to sources, for now not a problem

        if self._nc_files:
            # Episode variability comes from the file choice above plus a random
            # time window in the data: fields are linearly interpolated in time
            # as the episode advances (the Simulator passes sim_time each tick).
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
            # Randomize sources (using the env's seeded PRNG, not the global one).
            # Bounds are tied to the domain so sources scale with it (default 5 km).
            self.sources = random_sources(
                rng=self.np_random, n_sources=self.n_sources,
                min_x=0.0, max_x=self.domain[0],
                min_y=0.0, max_y=self.domain[1],
                min_depth=0.0, max_depth=self.domain[2],
            )

            def salinity_at(x, y, z):
                return compute_salinity_analytical(
                    x, y, z, self.sources, sigma_h=self.sigma_h, sigma_v=self.sigma_v)

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
            # sizes the swirls so they are resolved at km scale (not point-like).
            self.sim.vortex_field = sim_functions.VortexField(
                density=10,
                intensity=self.np_random.uniform(0.0, 0.3),
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

        # Compute spawn-side (S, τ) FIRST so we can pick a target that's actually far from it.
        spawn = self.sim.agents[0].pos
        self.current_salinity = salinity_at(spawn[0], spawn[1], spawn[2])
        self.current_turbidity = compute_turbidity(spawn[2])

        # Resample the target point until it's outside the success zone w.r.t. the spawn.
        # 2*epsilon margin so the agent has to actually navigate, not just nudge.
        for _ in range(100):  # safety cap; in practice ~1-2 iterations
            x_sel = self.np_random.uniform(0.0, self.domain[0])
            y_sel = self.np_random.uniform(0.0, self.domain[1])
            z_sel = self.np_random.uniform(0.0, self.domain[2])
            cand_S = salinity_at(x_sel, y_sel, z_sel)
            cand_T = compute_turbidity(z_sel)
            if (abs(cand_S - self.current_salinity) > 2 * self.epsilon_salinity
                    or abs(cand_T - self.current_turbidity) > 2 * self.epsilon_turbidity):
                break
        self.target_salinity = cand_S
        self.target_turbidity = cand_T

        # Initialize history buffer
        self.history = np.zeros((self.k, 2), dtype=np.float32)
        self.t_step = 0
        
        obs, _ = self._build_state(self.sim.agents[0])
        return obs, {}
    
    # NOTE: ok
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
        # Translate action into movement 
        mov = self._action_to_direction[action]
        agent = self.sim.agents[0]
        agent.cmd_local_vel = np.array([mov[0]*self.v, mov[1]*self.v])  # surge (x) and sway (y)
        agent.cmd_heave = mov[2]*self.v                                 # heave (z)
        agent.cmd_heading = np.rad2deg(np.arctan2(mov[0], mov[1]))      # NOTE: heading now auto-tracks motion direction,
                                                                        # simple but not fully realistic. Probably needs 
                                                                        # to be changed or at least discussed.
        
        # Doing the step in the sim
        # NOTE: reward is sampled only at the final sub-step (only-last), not summed across
        # the frame_skip ticks. This preserves the reward scale (and the meaning of the
        # +250 success bonus) when sweeping frame_skip, but PPO loses the integrated
        # signal of any high-reward region the agent passed through mid-skip. Worth
        # revisiting later — compare against summed-reward aggregation (paper convention,
        # Andrychowicz et al. 2021 §3.6) once a frame_skip ablation has been run.
        for _ in range(self.frame_skip):
            self.sim.tick()
            # Keep the agent inside the domain box: motion commands and currents
            # would otherwise push it above the surface (z < 0), below the seabed
            # (z > domain depth) or out of the horizontal extent. Clamped every
            # tick so field queries never run from out-of-bounds positions.
            agent.pos[0] = np.clip(agent.pos[0], 0.0, self.domain[0])
            agent.pos[1] = np.clip(agent.pos[1], 0.0, self.domain[1])
            agent.pos[2] = np.clip(agent.pos[2], 0.0, self.domain[2])
        self.t_step += 1

        # Next state (s')
        next_obs, reward = self._build_state(agent, action)

        # Truncation and termination checks
        if self._is_in_zone():
            self._in_zone_steps += 1
        else:
            self._in_zone_steps = 0
        truncated = (self.t_step >= self.max_steps)
        terminated = (self._in_zone_steps >= 3)     # NOTE: to define the right number of _in_zone_steps before success is met
        if terminated: 
            reward += 250  # bonus reward for success    
        
        return next_obs, reward, terminated, truncated, {}
    
    def _is_in_zone(self) -> bool:
        '''True when measured (S, tau) lie within epsilon of the target couple.'''
        return (
            abs(self.current_salinity - self.target_salinity) < self.epsilon_salinity
            and abs(self.current_turbidity - self.target_turbidity) < self.epsilon_turbidity
        )

    def _build_action_table(self) -> np.array:
        '''Returns an array of action->(dx,dy,dz) normalized.'''
        table = list(itertools.product([-1, 0, 1], repeat=3))
        norms = np.linalg.norm(table, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return table / norms 

    def _build_state(self, agent, action=None) -> tuple[np.ndarray, float]:
        '''
        Returns the observation of dimension (2k+11,) and the scalar reward.

        Layout:
            (2k)    -> history of (action, reward) pairs
            (3)     -> body-frame currents (u, v, w)
            (2)     -> absolute (salinity, turbidity) at the agent's current position
            (2)     -> target (salinity*, turbidity*)
            (3)     -> Field gradients
            (1)     -> depth
        '''
        if not self._nc_files:
            new_salinity = compute_salinity_analytical(x=agent.pos[0], y=agent.pos[1], z=agent.pos[2], sources=self.sources,
                                                    sigma_h=self.sigma_h, sigma_v=self.sigma_v)
            dSdx, dSdy, dSdz = compute_salinity_gradient_analytical(agent.pos[0], agent.pos[1], agent.pos[2], self.sources,
                                                                sigma_h=self.sigma_h, sigma_v=self.sigma_v)
        else:
            new_salinity = self.sim.current_3d.salinity_at(agent.pos[0], agent.pos[1], agent.pos[2])
            dSdx, dSdy, dSdz = self.sim.current_3d.salinity_gradient_at(agent.pos[0], agent.pos[1], agent.pos[2])
        new_turbidity = compute_turbidity(depth=agent.pos[2])

        # Currents the agent is actually advected by (NetCDF data currents in
        # data mode, synthetic surface stack + Ekman otherwise), rotated into
        # the body frame so both modes share the local-frame observation.
        currents = self.sim.depth_current_at(agent)
        u = currents[0] * np.cos(np.deg2rad(agent.psi)) + currents[1] * np.sin(np.deg2rad(agent.psi))
        v = currents[0] * np.sin(np.deg2rad(agent.psi)) - currents[1] * np.cos(np.deg2rad(agent.psi))
        w = currents[2]

        reward = reward_func(new_salinity, new_turbidity, self.target_salinity, self.target_turbidity)
        if action is not None:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = [action, reward]

        self.current_salinity = new_salinity
        self.current_turbidity = new_turbidity
        
        agent_depth = agent.pos[2]

        return np.concatenate([
            self.history.flatten(),
            np.array([u, v, w,
                      new_salinity, new_turbidity,
                      self.target_salinity, self.target_turbidity,
                      dSdx, dSdy, dSdz,
                      agent_depth]),
        ]).astype(np.float32), reward

    def close(self):
        for loader in self._loaders.values():
            loader.close()
        self._loaders.clear()
        super().close()

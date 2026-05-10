from .base import BaseEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SwarmSwIM import Simulator
from ..single_agent.reward import reward_func
import itertools

# TODO: implement the BaseEnv and inherit from that instead of gym.Env
class SingleAgentEnv(gym.Env):
    '''
    This class represents the wrapped environment of the simulation. It builds from 
    SwarmSwIM and is enclosed with Gymnasium for standardization.

    Parameters:
        - sim_xml -> .xml file containing the env configuration
        - manifest -> helper file to select the environment from the pre-computed ones
        - k -> history buffer length
        - v_agent -> agent velocity
        - max_steps -> maximum duration of an episode
        - target_S -> optimal salinity value
        - target_tau -> optimal turbidity value
        - dt -> time interval (s) for each sim step
    '''
    # NOTE: ok
    def __init__(self,
                 sim_xml,
                 manifest,
                 k=4,
                 v_agent = 0.5,
                 max_steps = 512,
                 target_S = 40.0,
                 target_tau = 0.3,
                 dt = 0.1
                 ):
        super().__init__()

        self.sim_xml = sim_xml 
        self.manifest = manifest 
        self.k = k 
        self.v = v_agent
        self.max_steps = max_steps
        self.target_S = target_S
        self.target_tau = target_tau
        self.dt = dt

        # Action Space
        self.action_space = gym.spaces.Discrete(27) # NOTE: remember that you have to use a relative PoV, not global

        # State/Observation Space
        obs_dim = 2*k + 4 + 3 + 2   # history k*(action,reward) + polar currents (4) + salinity + turbidity + depth + heading (2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Map actions to movements on the grid
        self._action_to_direction = self._build_action_table()  # scalar -> [dx,dy,dz] normalized

        self.sim = None

    # NOTE: ok (except TODOs)
    def reset(self, seed=None):
        '''
        Method that initializes an environment. 

        Parameters:
            - seed (int)

        Output:
            - state (np.array)
        '''
        super().reset(seed=seed)

        # pick a random precomputed env
        env_id = self.np_random.integers(len(self.manifest))
        netCdf_path = self.manifest[env_id]
        # TODO: subsample it to have a 1km x 1km domain

        # Create the env (Simulator class)
        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.sim_xml, netCdf_path=netCdf_path)    # NOTE: it instantiates an OceanDataCurrent 
                                                                                                        # model (configured via .xml) and the relative 
                                                                                                        # params need to be written in the file

        # Initialize agents randomly. NOTE: already initialized in the Simulator class,
        # but TODO: add the possibility to initialize randomly

        # Initialize history buffer
        self.history = np.zeros((self.k, 2), dtype=np.float32)
        self.t_step = 0

        return self._build_state(self.sim.agents[0]), {}
    
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
        agent.cmd_local_vel = np.array([mov[0]*self.v, mov[1]*self.v])
        agent.cmd_heave = mov[2]*self.v
        agent.cmd_heading = np.rad2deg(np.arctan2(mov[0], mov[1])) # NOTE: heading now auto-tracks motion direction,
                                                                # simple but not fully realistic. Probably needs 
                                                                # to be changed or at least discussed.
        
        # Doing the step in the sim
        self.sim.tick()
        self.t_step += 1

        # Reward computation
        S = self.sim.current_3d.query_salinity(agent, sim_time=self.sim.time)
        tau = self.sim.current_3d.query_turbidity(agent, sim_time=self.sim.time)
        reward = reward_func(S, tau)

        # Update history
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = [action, reward]

        # Next state (s')
        next_obs = self._build_state(agent)
        truncated = (self.t_step >= self.max_steps)
        terminated = False 
        
        return next_obs, reward, terminated, truncated, {}
    
    def _build_action_table(self) -> np.array:
        '''Returns an array of action->(dx,dy,dz) normalized.'''
        table = list(itertools.product([-1, 0, 1], repeat=3))
        norms = np.linalg.norm(table, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        return table / norms 

    def _build_state(self, agent) -> np.array:
         '''
         Returns the state of dimension 2k+9.

         (2k)   -> history (already an attribute)
         (4)    -> polar currents' coordinates
            - horizontal current magnitude
            - horizontal current direction (sin,cos -> 2)
            - vertical current
         (3)    -> salinity, turbidity, depth (, temperature?)
         (2)    -> heading (sin,cos)

         Returns:
            - np.array of dim (2k+9,)
         '''
         currents = self.sim.current_3d.calculate(agent, sim_time=self.sim.time)
         u = currents[0] * np.cos(np.deg2rad(agent.psi)) + currents[1] * np.sin(np.deg2rad(agent.psi))
         v = currents[0] * np.sin(np.deg2rad(agent.psi)) - currents[1] * np.cos(np.deg2rad(agent.psi))
         r_h = np.hypot(u, v)                                                                           # rotation-invariant
         theta = (v/r_h, u/r_h) if r_h > 1e-9 else (0.0, 1.0) 
         w = currents[2]                                                                                # rotation-invariant

         salinity = self.sim.current_3d.query_salinity(agent)
         turbidity = self.sim.current_3d.query_turbidity(agent)
         depth = agent.measured_depth

         heading = (np.sin(np.deg2rad(agent.measured_heading)), np.cos(np.deg2rad(agent.measured_heading)))

         return np.concatenate([
             self.history.flatten(),
             np.array([r_h, w]),
             theta,
             np.array([salinity, turbidity, depth]),
             heading
             ]).astype(np.float32)



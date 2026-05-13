from .base import BaseEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from SwarmSwIM import Simulator
from ..single_agent.reward import reward_func
import itertools
from src.models.salinity import compute_salinity_analytical
from src.models.turbidity import compute_turbidity
from src.utils.sources import load_sources

# TODO: implement the BaseEnv and inherit from that instead of gym.Env
class SingleAgentEnv(gym.Env):
    '''
    This class represents the wrapped environment of the simulation. It builds from 
    SwarmSwIM and is enclosed with Gymnasium for standardization.

    Parameters:
        - sim_xml -> .xml file containing the env configuration
        - source_file -> .json file encoding the sources for salinity
        - k -> history buffer length
        - v_agent -> agent velocity
        - max_steps -> maximum duration of an episode
        - dt -> time interval (s) for each sim step
        - domain -> list with max domain length for each axis
    '''
    # NOTE: ok
    def __init__(self,
                 xml_file: str,
                 source_file: str,
                 k=4,                           # history length of (action, reward)
                 v_agent = 0.5,                 # agent speed in m/s
                 max_steps = 512,               # steps of an episode before truncation
                 dt = 0.1,                      # seconds per step
                 domain = [100.0, 100.0, 100.0] # domain size (0.0-x, 0.0-y, 0.0-z)
                 ):
        super().__init__()

        self.sim_xml = xml_file
        self.sources = load_sources(source_file)
        self.k = k 
        self.v = v_agent
        self.max_steps = max_steps
        self.dt = dt
        self.domain = domain

        self.target_salinity = 0.0
        self.target_turbidity = 0.0

        self._in_zone_steps = 0.0       # used for success termination
        self.epsilon_salinity = 1e-3
        self.epsilon_turbidity = 1e-3

        # Action Space
        self.action_space = gym.spaces.Discrete(27) # NOTE: remember that you have to use a relative PoV, not global

        # State/Observation Space
        obs_dim = 2*k + 3 + 3   # history k*(action,reward) + local currents + salinity delta + turbidity delta + depth
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Map actions to movements on the grid
        self._action_to_direction = self._build_action_table()  # scalar -> [dx,dy,dz] normalized

        self.sim = None

    # NOTE: ok (except TODOs)
    def reset(self, seed=None) -> np.array:
        '''
        Method that initializes an environment. 

        Parameters:
            - seed (int)

        Output:
            - state (np.array)
        '''
        super().reset(seed=seed)

        # Create the env (Simulator class)
        self.sim = Simulator(timeSubdivision=self.dt, sim_xml=self.sim_xml)    

        # Randomize agent position
        for agent in self.sim.agents:
            agent.pos[0] = self.np_random.uniform(0.0, self.domain[0])
            agent.pos[1] = self.np_random.uniform(0.0, self.domain[1])
            agent.pos[2] = self.np_random.uniform(0.0, self.domain[2])
            agent.psi = self.np_random.uniform(-np.pi, np.pi)

        # NOTE: this implementation does not prevent agent spotting close to sources, for now not a problem

        # Randomize target
        x_sel = self.np_random.uniform(0.0, self.domain[0])
        y_sel = self.np_random.uniform(0.0, self.domain[1])
        z_sel = self.np_random.uniform(0.0, self.domain[2])
        self.target_salinity = compute_salinity_analytical(x_sel, y_sel, z_sel, self.sources)
        self.target_turbidity = compute_turbidity(z_sel)

        self.current_salinity = 0.0
        self.current_turbidity = 0.0

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
        self.sim.tick()
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
            reward += reward * 0.5  # bonus reward for success     
        
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

    def _build_state(self, agent, action = None) -> np.array:
         '''
         Returns the state of dimension 2k+9.

         (2k)   -> history (already an attribute)
         (3)    -> local currents
         (3)    -> salinity delta, turbidity delta, depth 

         Returns:
            - np.array of dim (2k+6,)
         '''
         new_salinity = compute_salinity_analytical(x=agent.pos[0],y=agent.pos[1],z=agent.pos[2], sources=self.sources)
         new_turbidity = compute_turbidity(depth=agent.pos[2])
         reward = reward_func(new_salinity, new_turbidity, self.target_salinity, self.target_turbidity)
         if action is not None:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = [action, reward]

         currents = self.sim.current_3d.calculate(agent)
         u = currents[0] * np.cos(np.deg2rad(agent.psi)) + currents[1] * np.sin(np.deg2rad(agent.psi))
         v = currents[0] * np.sin(np.deg2rad(agent.psi)) - currents[1] * np.cos(np.deg2rad(agent.psi))
         w = currents[2]                                                                                # rotation-invariant

         salinity_delta = new_salinity - self.current_salinity
         turbidity_delta = new_turbidity - self.current_turbidity
         self.current_salinity = new_salinity
         self.current_turbidity = new_turbidity
         depth = agent.measured_depth

         return np.concatenate([
             self.history.flatten(),
             np.array([u, v, w]),
             np.array([salinity_delta, turbidity_delta, depth]),
             ]).astype(np.float32), reward



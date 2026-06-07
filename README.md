<div align='center'>
    <h1>Underwater Search and Navigation in Realistic Environment</h1>
    <h3>Author: Christian Faccio</h3>
</div>

This work concerns the training and evaluation of a MARL algorithm suitable for underwater search and navigation in a realistic environment. The agents have find the optimal spot according to some conditions given beforehand, navigating through ocean currents. 

References at [this](https://drive.google.com/drive/folders/1lFe4wINNHlWUfY0CVZirNqOKKaz1Idst?usp=drive_link) Drive folder.

---

## Hardware stack

- MacAir M3 16GB
- NVIDIA Jetson Orin Nano 8GB
- (TBD) HPC Cluster

## Software stack

- uv: package manager
- SwarmSwIM: simulator
- Gymnasium: simulator wrapper
- torch: for neural networks

## Setup

First of all, create a virtual environment and install the dependencies:
```
git submodule update --init
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Then, make sure you have installed SwarmSwIM in developer mode:
```
cd SwarmSwIM
uv pip install -e .
```

## Structure

```
.
├── config              # configuration files
├── conftest.py
├── data                # real data for the env (TBD)
├── docs                # useful resources and references
├── pyproject.toml
├── README.md
├── requirements.txt
├── runs                # training runs
├── scripts             # useful scripts
├── src
│   ├── __init__.py
│   ├── envs            # gym wrappers
│   ├── eval.py         # evaluation script
│   ├── models          # models used
│   ├── multi_agent     # MARL algorithms
│   ├── single_agent    # RL algorithms
│   ├── train.py        # training script
│   └── utils           # utility functions
├── SwarmSwIM           # simulator
├── tests               # unit tests
└── thesis              # latex files
```

## Key choices

- Analytical env as starter, using solver for real-physics env later. Real-data will be put as a future work
- (1-10km, 1-10km, 35-50m) domain, agent max speed of 1m/s, battery life hypothetically of 2-8 hours
- discrete action space (27 actions which are the 3D neighbors + stall)
- continuous obs space (2k+11,)
- agents only know relative variables, no GPS, yes depth
- default PPO hyperparams from Andrychowicz et al.
- envs randomization at each episode to introduce variability
- optimal coral reef areas identified with (salinity, turbidity) pairs with analytical approach, with (salinity, turbidity, temperature) triplet when using Oceananigans
- an agent actual step is made of multiple equal actions, such that it can navigate the whole env with a fair amount of steps (not too many cause in reality would be in the order of 10000), also because field values don't change too much in the 1-10km domain
- Communication between agents is possible in the 1km domain using the BlueME technology, but not in the 10km domain, where only acoustic signals are available but have low latency
- Number of agents scale up to 32

## TO-DO

- Implement battery
- Less actions?
- 1km domain vs 10km domain analysis and comparison
- Scale number of agents
- Does initial agents' distribution in the domain affect the mission?
- Comparison between MARL algo and parallel RL (with full communication so to have a global state)
- Use small grid size for real data even with 1-10km domain?

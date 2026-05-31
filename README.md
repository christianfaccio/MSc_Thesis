<div align='center'>
    <h1>Underwater Search and Navigation in Realistic Environment</h1>
    <h3>Author: Christian Faccio</h3>
</div>

This work concerns the training and evaluation of a MARL algorithm suitable for underwater search and navigation in a realistic environment. The agents have find the optimal spot according to some conditions given beforehand, navigating through ocean currents. 

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

- Start with synthetic env and models, get success with MARL algo and then add env realism and complexity (p.s. discuss ROMS usage)
- discrete action space (27 actions which are the 3D neighbors + stall)
- continuous obs space (2k+11,)
- agents only know relative variables, no GPS, yes depth
- default PPO hyperparams from Andrychowicz et al.
- envs randomization at each episode to introduce variability
- targets done via salinity and turbidity values
# MSc_Thesis
Thesis for the Master Degree in Data Science and Artificial Intelligence held at University of Trieste (y.y. 2024-2026) and the Master Degree in Artificial Intelligence held at the University of Alicante in a Double Degree program (1st semester y.y. 2025/2026), in collaboration with the Sorbonne University of Abu Dhabi.

## Setup

First of all, create a virtual environment and install the dependencies:
```
git submodule update --init
uv venv .venv --python 3.11
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
├── config                          # OK
│   ├── default.xml
│   ├── simulation.xml
│   ├── single_agent.xml
│   └── sources.json
├── data                            # OK
│   ├── abu_dhabi_ocean_data.nc
│   ├── envs
│   ├── GSHHS
│   └── plots
├── docs                            # OK
├── README.md
├── requirements.txt
├── scripts                         # OK
│   ├── __init__.py
│   ├── plot_env.py
│   └── precompute_envs.py
├── src                            
│   ├── __init__.py
│   ├── envs                        # TODO
│   ├── eval.py                     # TODO
│   ├── models                      # OK
│   ├── multi_agent                 # TODO
│   ├── single_agent                # OK
│   ├── train.py                    # TODO
│   └── utils
├── SwarmSwIM
├── tests                           # TODO
└── thesis
```

## Roadmap

### Step 1: Environment

- 3D env
- Currents -> surface data from Copernicus, Ekman spirals for depth
- Dynamic env

### Step 2: Single Agent

- Params: salinity, light/turbidity (2/3 max)
- Create points from which salinity distributes (-> distribution model)
- Use equation for light/turbidity

### Step 3: Multi-Agent



### TBC

Other improvements/enhancements to be defined ... 

## How to download and load surface data

Use the following script to download surface data from Copernicus:
```
import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao", "so", "uo", "vo"],  # temperature, salinity, u-current, v-current
    minimum_longitude=53.5,
    maximum_longitude=55.5,
    minimum_latitude=23.5,
    maximum_latitude=25.5,
    minimum_depth=0,
    maximum_depth=200,
    start_datetime="2020-01-01",
    end_datetime="2020-12-31",
    output_filename="abu_dhabi_ocean_data.nc",
)
```

and the following to load it:
```
import xarray as xr

ds = xr.open_dataset("abu_dhabi_ocean_data.nc")
print(ds)  # shows dimensions, variables, coordinates
print(ds.thetao.sel(depth=5, method="nearest").isel(time=0))  # temperature at 5m depth
```

## Further work

- Use ROMS as numerical simulator for currents
- Use the Pareto fronts for the reward function instead of computing a single function for all the variables (meaning an agent will try to find "good enough" areas instead of global optima)
- Optimize the code for parallel working

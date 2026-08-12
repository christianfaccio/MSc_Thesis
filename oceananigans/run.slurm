#!/bin/bash
#SBATCH -J ocean_env
#SBATCH -A uTS26_Tornator
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod          
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                 # 32 cores / 4 GPUs = 8 per GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# Walltime raised for the 2026-08-05 spin-up fix. Each run is now 12 h warmup +
# 6 h recording = 18 h simulated (was 1.5 h). The max_Δt cap also went 3 -> 10 s,
# so the step count rises ~3.6x, not 12x: ~6500 steps/run vs ~1800 before.
# Budget ~3.6x whatever a run used to cost, and scale -t with --n-runs.
#SBATCH -t 04:00:00
#SBATCH -o ocean_env_%j.out -e ocean_env_%j.err

module purge
module load cuda/12.2

export PATH="$HOME/.juliaup/bin:$PATH"    
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}

export OCEAN_ARCH=GPU
export JULIA_NUM_THREADS=8

# SINGLE VALIDATION RUN. Check the currents with
#   python scripts/plot_oceananigans_currents.py --netcdf <the .nc>
# before generating the rest.
#
# ACCEPTANCE TEST = does the `umean` trace in the .out file FLATTEN by the end
# of the warmup? If it is still climbing, raise --warmup-minutes. Judge it on
# that, NOT on hitting a target speed: there is no reliable prior for the
# magnitude here. The "3% of wind speed" rule (~0.15 m/s) assumes a frictional
# sink, but the walls and bottom are free-slip, so the only momentum sink is
# LES dissipation; and the 2026-08-06 topology change closed the basin, which
# removes the endless along-shore jet a periodic channel could sustain. Both
# push the expected speed DOWN, plausibly to a few cm/s. Anything well above
# the old 0.004 m/s with a flat trace is a good result — record what you get.
#
# Also worth a look, both consequences of the closed basin: current pile-up
# against the east (x=LX) and north (y=LY) walls, which are open ocean in
# reality; and whether the salinity span is still ~10 PSU with corr(first,
# last snapshot) well below the old pinned-field value of 0.9993. If it is
# still near that, S_DECAY_TIME wants raising.
#
# Then bump --n-runs for the full set.
#
# Add --tide-amplitude 0.2 for a barotropic M2 tide (off by default; it is the
# largest remaining physical omission for a Gulf setting, but validate the
# wind-only baseline first so the two effects stay separable).
srun julia --project=oceananigans oceananigans/non_hydrostatic.jl \
    	--n-runs 1 --season winter --seed 2 \
	--output-dir $CINECA_SCRATCH/oceananigans/test

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
# before generating the rest — expect mean |u,v| ~0.1-0.15 m/s and a `umean`
# trace in the .out file that has FLATTENED by the end of the warmup. If it is
# still climbing, raise --warmup-minutes. Then bump --n-runs for the full set.
#
# Add --tide-amplitude 0.2 for a barotropic M2 tide (off by default; it is the
# largest remaining physical omission for a Gulf setting, but validate the
# wind-only baseline first so the two effects stay separable).
srun julia --project=oceananigans oceananigans/non_hydrostatic.jl \
    	--n-runs 1 --season winter --seed 2 \
	--output-dir $CINECA_SCRATCH/oceananigans/test

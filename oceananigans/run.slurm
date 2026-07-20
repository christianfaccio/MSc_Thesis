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
#SBATCH -t 02:00:00
#SBATCH -o ocean_env_%j.out -e ocean_env_%j.err

module purge
module load cuda/12.2

export PATH="$HOME/.juliaup/bin:$PATH"    
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}

export OCEAN_ARCH=GPU
export JULIA_NUM_THREADS=8

srun julia --project=oceananigans oceananigans/non_hydrostatic.jl \
    	--n-runs 10 --season winter --seed 2 \
	--output-dir $CINECA_SCRATCH/oceananigans/test

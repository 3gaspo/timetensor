#!/bin/bash
#SBATCH --wckey=P12CU:DATASCIENCE
#SBATCH --partition=an
#SBATCH --qos=an_all_long_rd
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=load_synthetic
#SBATCH --output=scripts/synthetic/load.out
#SBATCH --error=scripts/synthetic/load.err

source .venv/bin/activate

data=synthetic
srun python3 load_dataset.py "data=$data" "task.lags=168" "task.horizon=24" "load.rebuild=True" "load.example=True" "load.replot=True" "load.heterogeneity=True" "load.clusters=True" "load.logs=True"
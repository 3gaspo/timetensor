#!/bin/bash
#SBATCH --wckey=P12CU:DATASCIENCE
#SBATCH --partition=an
#SBATCH --qos=an_all_long_rd
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=load_traffic
#SBATCH --output=scripts/traffic/load.out
#SBATCH --error=scripts/traffic/load.err

source .venv/bin/activate

data=traffic
srun python3 load_dataset.py "data=$data" "task.lags=168" "task.horizon=24" "load.rebuild=True" "load.example=True" "load.replot=True" "load.heterogeneity=True" "load.clusters=True" "load.logs=True"
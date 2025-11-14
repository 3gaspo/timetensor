#!/bin/bash
#SBATCH --job-name=load
#SBATCH --gres=gpu:1
#SBATCH --output=new_scripts/load.out
#SBATCH --error=new_scripts/load.err
#SBATCH --time=02:00:00
#SBATCH --partition=a100
source .venv/bin/activate

for data in electricity solar traffic synthetic
do
    srun python3 load_dataset.py "data=$data" "task.lags=336" "task.horizon=48" "load.rebuild=False" "load.example=False" "load.replot=False" "load.heterogeneity=False" "load.clusters=False" "load.logs=True"
done
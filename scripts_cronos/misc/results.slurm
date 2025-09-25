#!/bin/bash
#SBATCH --wckey=P12CU:DATASCIENCE
#SBATCH --partition=an
#SBATCH --qos=an_all_short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --job-name=results
#SBATCH --output=scripts/misc/results.out
#SBATCH --error=scripts/misc/results.err
source .venv/bin/activate

for data in synthetic #electricity solar traffic
do
    lags=336
    horizon=48

    srun python3 load_dataset.py "data=$data" "task.lags=$lags" "task.horizon=$horizon" "load.rebuild=True" "load.example=True" "load.replot=True" "load.heterogeneity=True" "load.clusters=True" "load.logs=True"
done
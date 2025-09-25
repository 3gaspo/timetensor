#!/bin/bash
#SBATCH --wckey=P12CU:DATASCIENCE
#SBATCH --partition=an
#SBATCH --qos=an_all_short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --job-name=read
#SBATCH --output=scripts/misc/read.out
#SBATCH --error=scripts/misc/read.err
source .venv/bin/activate

for bs in 64
do
    srun python3 test_model.py "data=traffic" "task.lags=504" "task.horizon=168" "training.bs=$bs" "normalization=revin" "training.epochs=1200"
done
# nohup bash scripts/read.sh > read.log 2>&1 &
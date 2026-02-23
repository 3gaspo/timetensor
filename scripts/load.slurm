#!/bin/bash
#SBATCH --job-name=load
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/load.out
#SBATCH --error=script_outputs/load.err
#SBATCH --time=02:00:00
#SBATCH --partition=h100
source .venv/bin/activate

for data in raw_electricity #electricity traffic weather ETTh1
do
    srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=168" "task.horizon=24" "load.rebuild=True" "load.example=True" "load.shapes=True"

    # srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=672" "task.horizon=168" "load.rebuild=False" "load.example=True" "load.shapes=True"

    # srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=24" "task.horizon=24" "load.rebuild=False" "load.example=True" "load.shapes=True"

done

# for data in solar
# do
#     srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=168" "task.horizon=24" "load.rebuild=True" "load.example=True" "load.shapes=True"

#     srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=672" "task.horizon=168" "load.rebuild=False" "load.example=True" "load.shapes=True"

#     srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=24" "task.horizon=24" "load.rebuild=False" "load.example=True" "load.shapes=True"

# done

# for data in synthetic exchange_rate
# do
#     srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=30" "task.horizon=30" "load.rebuild=True" "load.example=True" "load.shapes=True"

#     srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=100" "task.horizon=10" "load.rebuild=False" "load.example=True" "load.shapes=True"

#     srun python3 load_dataset.py "data.sampling.use_context=False" "data=$data" "task.lags=100" "task.horizon=100" "load.rebuild=False" "load.example=True" "load.shapes=True"

# done


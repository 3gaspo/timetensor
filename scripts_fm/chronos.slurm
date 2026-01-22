#!/bin/bash
#SBATCH --job-name=chronos
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/chronos.out
#SBATCH --error=script_outputs/chronos.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1

lags=672
horizon=168
stride=672
for data in electricity traffic solar exchange_rate
do
    srun python3 eval_model.py \
    "model=chronos" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='all'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.epochs=1" \
    "training.bs=350" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='chronos2'" \
    "misc.output_dir='../outputs/fm/central/$data/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=chronos" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='all'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.epochs=1" \
    "training.bs=350" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='chronos2'" \
    "misc.output_dir='../outputs/fm/central/$data/${lags}_$horizon/'"
done
#!/bin/bash
#SBATCH --job-name=users
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/users.out
#SBATCH --error=script_outputs/users.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1

lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.epochs=3" \
    "training.bs=256" \
    "training.eval_runs=1" \
    "training.print_freq=1000" \
    "training.eval_freq=1000" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_users'" \
    "misc.output_dir='../outputs/fm/central/$data/${lags}_$horizon/'"
done

lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.epochs=3" \
    "training.bs=256" \
    "training.eval_runs=1" \
    "training.print_freq=1000" \
    "training.eval_freq=1000" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_users'" \
    "misc.output_dir='../outputs/fm/central/$data/${lags}_$horizon/'"
done

lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_users.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='dates'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.epochs=3" \
    "training.bs=256" \
    "training.eval_runs=1" \
    "training.print_freq=1000" \
    "training.eval_freq=1000" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_users'" \
    "misc.output_dir='../outputs/fm/central/$data/${lags}_$horizon/'"
done
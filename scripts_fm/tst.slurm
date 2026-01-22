#!/bin/bash
#SBATCH --job-name=tst
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tst.out
#SBATCH --error=script_outputs/tst.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1

lags=672
horizon=168
stride=672
for data in electricity traffic solar exchange_rate
do
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1500" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='../outputs/fm/central/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "training.init=../outputs/fm/central_tst/PatchTST/trained_model.pt" \
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
    "training.bs=350" \
    "training.epochs=1" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='../outputs/fm/central/'"
done


    # srun python3 train_model.py \
    # "model=PatchTST" \
    # "task.lags=672" \
    # "task.horizon=168" \
    # "normalization=instance" \
    # "data=$data" \
    # "data.sampling.use_context=False" \
    # "data.splits.date_splits='0.6;0.2'" \
    # "data.splits.indiv_split=1" \
    # "data.sampling.train_idx_mode='random'" \
    # "data.sampling.eval_idx_mode='random'" \
    # "data.sampling.train_stride=1" \
    # "data.sampling.eval_stride=168" \
    # "data.sampling.remove_train_cte=True" \
    # "data.sampling.remove_eval_cte=True" \
    # "training.complete_evaluation=False" \
    # "training.bs=316" \
    # "training.epochs=1" \
    # "training.eval_runs=1" \
    # "data.sampling.train_len_multiplier=250_000" \
    # "data.sampling.eval_len_multiplier=1000" \
    # "training.print_freq=10_000" \
    # "training.eval_freq=10_000" \
    # "misc.seed=$seed" \
    # "misc.save_name='PatchTST_random'" \
    # "misc.output_dir='../outputs/fm/central_tst/'"

    # srun python3 eval_users.py \
    # "model=PatchTST" \
    # "training.init=../outputs/fm/central_tst/PatchTST_random/trained_model.pt" \
    # "task.lags=672" \
    # "task.horizon=168" \
    # "normalization=instance" \
    # "data=$data" \
    # "data.sampling.use_context=False" \
    # "data.splits.date_splits='0.6;0.2'" \
    # "data.splits.indiv_split=1" \
    # "data.sampling.train_idx_mode='all'" \
    # "data.sampling.eval_idx_mode='all'" \
    # "data.sampling.train_stride=168" \
    # "data.sampling.eval_stride=168" \
    # "data.sampling.remove_train_cte=True" \
    # "data.sampling.remove_eval_cte=True" \
    # "training.complete_evaluation=False" \
    # "training.bs=350" \
    # "training.epochs=1" \
    # "training.eval_runs=1" \
    # "data.sampling.train_len_multiplier=1" \
    # "data.sampling.eval_len_multiplier=1" \
    # "misc.seed=$seed" \
    # "misc.save_name='PatchTST_random'" \
    # "misc.output_dir='../outputs/fm/central_tst/'"
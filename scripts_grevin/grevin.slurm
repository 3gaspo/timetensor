#!/bin/bash
#SBATCH --job-name=grevin
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/grevin.out
#SBATCH --error=script_outputs/grevin.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/grevin/"

lags=672
horizon=168
stride=512
for data in electricity #traffic solar exchange_rate
do
    for normalization in PRevIN cmIN
    do
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=$normalization" \
        "data=$data" \
        "data.sampling.use_context=True" \
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
        "misc.save_name='PatchTST_$normalization'" \
        "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"

    for normalization in instance revin grevin
    do
        srun python3 train_model.py \
        "model=PatchTST" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=$normalization" \
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
        "misc.save_name='PatchTST_$normalization'" \
        "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"
    done
done


# lags=168
# horizon=24
# stride=512

# lags=24
# horizon=24
# stride=512

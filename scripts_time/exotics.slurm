#!/bin/bash
#SBATCH --job-name=exotics
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/exotics.out
#SBATCH --error=script_outputs/exotics.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/exotics/"

lags=672
horizon=168
stride=512
for data in raw_electricity #traffic solar exchange_rate
do
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits='0.6;0.2;0.2'" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=False" \
    "data.sampling.remove_eval_cte=False" \
    "training.complete_evaluation=False" \
    "training.bs=175" \
    "training.epochs=750" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_all'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits='0.6;0.2;0.2'" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=False" \
    "training.complete_evaluation=False" \
    "training.bs=175" \
    "training.epochs=750" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_rmvtrain'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits='0.6;0.2;0.2'" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=175" \
    "training.epochs=750" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_rmvall'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits='0.6;0.2;0.2'" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=False" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=175" \
    "training.epochs=750" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST_rmvtest'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done


settings="1344_336;672_168;168_24;24_24"
model_names="PatchTST_all;PatchTST_rmvtrain;PatchTSTrmvall;PatchTST_rmvtest"

for metric in nMSE 'w10 nMSE' 'train time (min)' 'eval time (min)'
do
    for prefix in train test1
    do
        file_name="${prefix}_mean_results.json"
        output_path="${experiment_dir}${prefix}_${metric}_results.tex"
        srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
    done
done
#!/bin/bash
#SBATCH --job-name=robust
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/robust.out
#SBATCH --error=script_outputs/robust.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/robust/"

lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
do
    for augment in none identity "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
    do
        srun python3 eval_users.py \
        "model=chronos" \
        "model.configs.self_augment='$augment'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='$augment'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for augment in none identity "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
    do
        srun python3 eval_users.py \
        "model=chronos" \
        "model.configs.self_augment='$augment'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='$augment'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for augment in none identity "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
    do
        srun python3 eval_users.py \
        "model=chronos" \
        "model.configs.self_augment='$augment'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='$augment'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

file_name=test1_mean_results.json

model_names="none;identity"
prefix="identity"
for metric in nMSE 'w10 nMSE' 'compute (min)'
do
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names')"
done

model_names="none;constant=0;constant=1"
prefix="constants"
for metric in nMSE 'w10 nMSE' 'compute (min)'
do
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names')"
done

model_names="none;noise=0.01;noise=0.1;noise=1"
prefix="noise"
for metric in nMSE 'w10 nMSE' 'compute (min)'
do
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names')"
done
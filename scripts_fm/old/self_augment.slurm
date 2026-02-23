#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/augment.out
#SBATCH --error=script_outputs/augment.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/augment/"

lags=672
horizon=168
stride=512
for data in solar #electricity traffic solar exchange_rate
do
    for augment in none #kernel square root sign mirror all 'kernel;square;root'
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
        "misc.save_name='chronos2_augment$augment'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=168
horizon=24
stride=512
for data in solar #electricity traffic solar exchange_rate
do
    for augment in none #none kernel square root sign mirror all 'kernel;square;root'
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
        "misc.save_name='chronos2_augment$augment'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


lags=24
horizon=24
stride=512
for data in solar #electricity traffic solar exchange_rate
do
    for augment in none #none kernel square root sign mirror all 'kernel;square;root'
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
        "misc.save_name='chronos2_augment$augment'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

file_name=mean_results.json
for metric in nMSE 'w10 nMSE' 'compute (min)'
do
    output_path="${experiment_dir}${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path')"
done
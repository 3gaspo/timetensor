#!/bin/bash
#SBATCH --job-name=aug_cr
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/aug_cr.out
#SBATCH --error=script_outputs/aug_cr.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/robust/"

lags=1344
horizon=336
stride=512
for model in chronos
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in sign #identity square
        do
            srun python3 eval_users.py \
            "model='$model'" \
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
            "misc.save_name='${model}_$augment'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

lags=672
horizon=168
stride=512
for model in chronos
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in sign #identity square
        do
            srun python3 eval_users.py \
            "model='$model'" \
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
            "misc.save_name='${model}_$augment'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

lags=168
horizon=24
stride=512
for model in chronos
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in sign #identity square
        do
            srun python3 eval_users.py \
            "model='$model'" \
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
            "misc.save_name='${model}_$augment'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

lags=24
horizon=24
stride=512
for model in chronos
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in sign #identity square
        do
            srun python3 eval_users.py \
            "model='$model'" \
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
            "misc.save_name='${model}_$augment'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

file_name=test1_mean_results.json
settings="1344_336;672_168;168_24;24_24"
metric=nMSE

for model in chronos
do
    #noises
    model_names="${model}_none;${model}_square;${model}_identity${model}_sign;"
    prefix="${model}_augments"
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

done
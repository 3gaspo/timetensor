#!/bin/bash
#SBATCH --job-name=robust_cr
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/robust_cr.out
#SBATCH --error=script_outputs/robust_cr.err
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
        for augment in none "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
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

        for augment in "2noise=1" "3noise=1" "5noise=1"
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

        for augment in "2constant=0" "3constant=0" "5constant=0"
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
        for augment in none "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
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

        for augment in "2noise=1" "3noise=1" "5noise=1"
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

        for augment in "2constant=0" "3constant=0" "5constant=0"
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
        for augment in none "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
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

        for augment in "2noise=1" "3noise=1" "5noise=1"
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

        for augment in "2constant=0" "3constant=0" "5constant=0"
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
        for augment in none "noise=1" "noise=0.1" "noise=0.01" "constant=0" "constant=1"
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

        for augment in "2noise=1" "3noise=1" "5noise=1"
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

        for augment in "2constant=0" "3constant=0" "5constant=0"
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
    #constant
    model_names="${model}_none;${model}_constant=0;${model}_constant=1"
    prefix="${model}_constant"
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

    #noise
    model_names="${model}_none;${model}_noise=0.01;${model}_noise=0.1;${model}_noise=1"
    prefix="${model}_noise"
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

    #noises
    model_names="${model}_none;${model}_noise=1;${model}_2noise=1;${model}_3noise=1;${model}_5noise=1"
    prefix="${model}_noises"
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

    #constants
    model_names="${model}_none;${model}_constant=0;${model}_2constant=0;${model}_3constant=0;${model}_5constant=0"
    prefix="${model}_constants"
    output_path="${experiment_dir}${prefix}_${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

done
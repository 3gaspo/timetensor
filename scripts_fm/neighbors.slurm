#!/bin/bash
#SBATCH --job-name=neighbors
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/neighbors.out
#SBATCH --error=script_outputs/neighbors.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/neighbors/"
model=chronos
date_splits='0.6;0.4'

lags=1344
horizon=336
stride=512
for data in electricity traffic solar
do
    #bs1
    srun python3 neighbors_distance.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    for bs in 2 3 4 5 10
    do
        #cos distance
        srun python3 neighbors_distance.py \
        "extra.distance=cosine" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_cos_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #eucl distance
        srun python3 neighbors_distance.py \
        "extra.distance=euclidean" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_eucl_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #windows
        srun python3 neighbors_windows.py \
        "extra.distance=euclidean" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_windows_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #past
        srun python3 covariates_past.py \
        "model='$model'" \
        "model.configs.context_mode=future" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_past_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #all
        srun python3 neighbors_all.py \
        "model='$model'" \
        "model.configs.context_mode=future" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_all_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=672
horizon=168
stride=512
for data in electricity traffic solar
do
    #bs1
    srun python3 neighbors_distance.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    for bs in 2 3 4 5 10
    do
        #cos distance
        srun python3 neighbors_distance.py \
        "extra.distance=cosine" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_cos_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #eucl distance
        srun python3 neighbors_distance.py \
        "extra.distance=euclidean" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_eucl_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #windows
        srun python3 neighbors_windows.py \
        "extra.distance=euclidean" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_windows_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #past
        srun python3 covariates_past.py \
        "model='$model'" \
        "model.configs.context_mode=future" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_past_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #all
        srun python3 neighbors_all.py \
        "model='$model'" \
        "model.configs.context_mode=future" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_all_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


lags=168
horizon=24
stride=512
for data in electricity traffic solar
do
    #bs1
    srun python3 neighbors_distance.py \
    "model='$model'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=1" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_baseline'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    for bs in 2 3 4 5 10
    do
        #cos distance
        srun python3 neighbors_distance.py \
        "extra.distance=cosine" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_cos_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #eucl distance
        srun python3 neighbors_distance.py \
        "extra.distance=euclidean" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_eucl_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #windows
        srun python3 neighbors_windows.py \
        "extra.distance=euclidean" \
        "model='$model'" \
        "model.configs.context_mode=past_only" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors_windows_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #past
        srun python3 covariates_past.py \
        "model='$model'" \
        "model.configs.context_mode=future" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_past_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        #all
        srun python3 neighbors_all.py \
        "model='$model'" \
        "model.configs.context_mode=future" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_all_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


settings="1344_336;672_168;168_24"
file_name=mean_results.json
model=chronos


prefix="eucl"
model_names="${model}_baseline;${model}_neighbors_eucl_2;${model}_neighbors_eucl_3;${model}_neighbors_eucl_4;${model}_neighbors_eucl_5;${model}_neighbors_eucl_10"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

prefix="cos"
model_names="${model}_baseline;${model}_neighbors_cos_2;${model}_neighbors_cos_3;${model}_neighbors_cos_4;${model}_neighbors_cos_5;${model}_neighbors_cos_10"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

prefix="windows"
model_names="${model}_baseline;${model}_neighbors_windows_2;${model}_neighbors_windows_3;${model}_neighbors_windows_4;${model}_neighbors_windows_5;${model}_neighbors_windows_10"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

prefix="past"
model_names="${model}_baseline;${model}_past_2;${model}_past_3;${model}_past_4;${model}_past_5;${model}_past_10"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

prefix="all"
model_names="${model}_baseline;${model}_all_2;${model}_all_3;${model}_all_4;${model}_all_5;${model}_all_10"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
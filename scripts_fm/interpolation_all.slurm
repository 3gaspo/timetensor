#!/bin/bash
#SBATCH --job-name=all_interpolation
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/all_interpolation.out
#SBATCH --error=script_outputs/all_interpolation.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/all_interpolation/"
model=chronos
date_splits='0.6;0.4'

lags=1344
horizon=336
stride=512
for data in electricity traffic solar
do
    # #bs1
    # srun python3 neighbors_distance.py \
    # "model='$model'" \
    # "task.lags=$lags" \
    # "task.horizon=$horizon" \
    # "normalization=instance" \
    # "data=$data" \
    # "data.sampling.train_stride=$stride" \
    # "data.sampling.eval_stride=$stride" \
    # "training.bs=1" \
    # "training.eval_runs=1" \
    # "misc.seed=$seed" \
    # "misc.device='cpu'" \
    # "misc.save_name='${model}_baseline'" \
    # "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    for bs in 10
    do
        metric=cosine
        space=raw
        distance="${space}_${metric}"
        srun python3 interpolation_all.py \
        "extra.distance='$distance'" \
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
        "misc.save_name='${model}_interpolation_${distance}_${bs}'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=672
horizon=168
stride=512
for data in electricity traffic solar
do
    # #bs1
    # srun python3 neighbors_distance.py \
    # "model='$model'" \
    # "task.lags=$lags" \
    # "task.horizon=$horizon" \
    # "normalization=instance" \
    # "data=$data" \
    # "data.sampling.train_stride=$stride" \
    # "data.sampling.eval_stride=$stride" \
    # "training.bs=1" \
    # "training.eval_runs=1" \
    # "misc.seed=$seed" \
    # "misc.device='cpu'" \
    # "misc.save_name='${model}_baseline'" \
    # "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    for bs in 10
    do
        metric=cosine
        space=raw
        distance="${space}_${metric}"
        srun python3 interpolation_all.py \
        "extra.distance='$distance'" \
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
        "misc.save_name='${model}_interpolation_${distance}_${bs}'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=168
horizon=24
stride=512
for data in electricity traffic solar
do
    # #bs1
    # srun python3 neighbors_distance.py \
    # "model='$model'" \
    # "task.lags=$lags" \
    # "task.horizon=$horizon" \
    # "normalization=instance" \
    # "data=$data" \
    # "data.sampling.train_stride=$stride" \
    # "data.sampling.eval_stride=$stride" \
    # "training.bs=1" \
    # "training.eval_runs=1" \
    # "misc.seed=$seed" \
    # "misc.device='cpu'" \
    # "misc.save_name='${model}_baseline'" \
    # "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    for bs in 10
    do
        metric=cosine
        space=raw
        distance="${space}_${metric}"
        srun python3 interpolation_all.py \
        "extra.distance='$distance'" \
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
        "misc.save_name='${model}_interpolation_${distance}_${bs}'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=24
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

    for bs in 10
    do
        metric=cosine
        space=raw
        distance="${space}_${metric}"
        srun python3 interpolation_all.py \
        "extra.distance='$distance'" \
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
        "misc.save_name='${model}_interpolation_${distance}_${bs}'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

settings="1344_336;672_168;168_24;24_24"
file_name=mean_results.json
model=chronos
model_names="${model}_baseline;${model}_interpolation_${distance}_${bs}"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', model_names='$model_names', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', settings='$settings', decimals=4)"
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

lags=672
horizon=168
stride=512
for data in electricity #traffic solar
do
    for bs in 1 2 #1 2 3 4 5 10
    do
        srun python3 neighbors_test_cos.py \
        "model=chronos" \
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

        srun python3 neighbors_test_eucl.py \
        "model=chronos" \
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

        srun python3 neighbors_test_lookback.py \
        "model=chronos" \
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
        "misc.save_name='${model}_neighbors_lookback_$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

        srun python3 lookback_test.py \
        "model=chronos" \
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

    done
done


settings="1344_336;672_168;168_24;24_24"
file_name=mean_results.json
model=chronos

for metric in nMSE 'w10 nMSE' 'train time (min)' 'eval time (min)'
do
    output_path="${experiment_dir}neighbors_chronos_${metric}_results.tex"
    model_names="${model}_none;${model}_sign;${model}_neighbors2;${model}_neighbors3;${model}_neighbors5;${model}_neighbors10;${model}_neighbors100;${model}_neighbors1000"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
done
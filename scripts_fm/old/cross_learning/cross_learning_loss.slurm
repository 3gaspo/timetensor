#!/bin/bash
#SBATCH --job-name=cl_loss
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/cl_loss.out
#SBATCH --error=script_outputs/cl_loss.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

experiment_dir="../outputs/fm/cross_learning/loss/"
seed=1

lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2 #1 2
    do
        srun python3 cross_learning_loss.py \
        "model=chronos" \
        "model.configs.cross_learning=True" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=100" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='chronos2_None_bs$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2 #1 2
    do
        srun python3 cross_learning_loss.py \
        "model=chronos" \
        "model.configs.cross_learning=True" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=100" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='chronos2_None_bs$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2 #1 2
    do
        srun python3 cross_learning_loss.py \
        "model=chronos" \
        "model.configs.cross_learning=True" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=100" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='chronos2_None_bs$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

file_name=mean_results.json
settings="672_168;168_24;24_24"
for metric in nMSE 'w10 nMSE' 'training (min)' 'testing (min)'
do
    output_path="${experiment_dir}${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', settings='$settings')"
done
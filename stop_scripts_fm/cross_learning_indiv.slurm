#!/bin/bash
#SBATCH --job-name=cl_indiv
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/cl_indiv.out
#SBATCH --error=script_outputs/cl_indiv.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
source .venv/bin/activate

seed=1
eval_runs=10
experiment_dir="../outputs/fm/cross_learning/indiv/"

lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
    do
    for bs in 1 2 3 4 5 10
    do
        for cross in True False
        do
            srun python3 cross_learning_indiv.py \
            "model=chronos" \
            "model.configs.cross_learning=$cross" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
            "data.sampling.train_stride=$stride" \
            "data.sampling.eval_stride=$stride" \
            "training.bs=$bs" \
            "training.eval_runs=$eval_runs" \
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='chronos2_cross${cross}_bs$bs'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done


lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
    do
    for bs in 1 2 3 4 5 10
    do
        for cross in True False
        do
            srun python3 cross_learning_indiv.py \
            "model=chronos" \
            "model.configs.cross_learning=$cross" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
            "data.sampling.train_stride=$stride" \
            "data.sampling.eval_stride=$stride" \
            "training.bs=$bs" \
            "training.eval_runs=$eval_runs" \
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='chronos2_cross${cross}_bs$bs'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done


lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
    do
    for bs in 1 2 3 4 5 10
    do
        for cross in True False
        do
            srun python3 cross_learning_indiv.py \
            "model=chronos" \
            "model.configs.cross_learning=$cross" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
            "data.sampling.train_stride=$stride" \
            "data.sampling.eval_stride=$stride" \
            "training.bs=$bs" \
            "training.eval_runs=$eval_runs" \
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='chronos2_cross${cross}_bs$bs'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

file_name=mean_results.json
for metric in nMSE 'w10 nMSE' 'compute (min)'
do
    output_path="${experiment_dir}${metric}_results.tex"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', settings='$settings')"
done
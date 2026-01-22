#!/bin/bash
#SBATCH --job-name=cl_indiv
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/cl_indiv.out
#SBATCH --error=script_outputs/cl_indiv.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
eval_runs=10

lags=672
horizon=168
stride=672
for data in electricity traffic solar exchange_rate
    do
    for bs in 1 2 3 4 5 10 100 350
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
            "misc.output_dir='../outputs/fm/cross_indiv/$data/${lags}_$horizon/'"
        done
    done
done


lags=168
horizon=24
stride=24
for data in electricity traffic solar exchange_rate
    do
    for bs in 1 2 3 4 5 10 100 350
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
            "misc.output_dir='../outputs/fm/cross_indiv/$data/${lags}_$horizon/'"
        done
    done
done


lags=24
horizon=24
stride=24
for data in electricity traffic solar exchange_rate
    do
    for bs in 1 2 3 4 5 10 100 350
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
            "misc.output_dir='../outputs/fm/cross_indiv/$data/${lags}_$horizon/'"
        done
    done
done
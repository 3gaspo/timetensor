#!/bin/bash
#SBATCH --job-name=cl_clusters
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/cl_clusters.out
#SBATCH --error=script_outputs/cl_clusters.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
source .venv/bin/activate

seed=1

lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 1 2 3 4 5 10
    do
        use_context=False
        cross_learning=True
        srun python3 cross_learning_clusters.py \
        "model=chronos" \
        "model.configs.cross_learning=$cross_learning" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='chronos2_None_bs$bs'" \
        "misc.output_dir='../outputs/fm/cross_learning/clusters/$data/${lags}_$horizon/'"
    done
done

lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 1 2 3 4 5 10
    do
        use_context=False
        cross_learning=True
        srun python3 cross_learning_clusters.py \
        "model=chronos" \
        "model.configs.cross_learning=$cross_learning" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='chronos2_None_bs$bs'" \
        "misc.output_dir='../outputs/fm/cross_learning/clusters/$data/${lags}_$horizon/'"
    done
done

lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 1 2 3 4 5 10
    do
        use_context=False
        cross_learning=True
        srun python3 cross_learning_clusters.py \
        "model=chronos" \
        "model.configs.cross_learning=$cross_learning" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=$use_context" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "training.bs=$bs" \
        "training.eval_runs=1" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='chronos2_None_bs$bs'" \
        "misc.output_dir='../outputs/fm/cross_learning/clusters/$data/${lags}_$horizon/'"
    done
done
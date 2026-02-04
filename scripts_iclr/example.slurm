#!/bin/bash
#SBATCH --job-name=example
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/example.out
#SBATCH --error=script_outputs/example.err
#SBATCH --time=01:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/example/"
model=tabpfn

lags=1344
horizon=336
for data in electricity
do
    for model in chronos tabpfn
    do
        srun python3 example_model.py \
        "model='$model'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='$model'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


lags=168
horizon=24
for data in electricity
do
    for model in chronos tabpfn
    do
        srun python3 example_model.py \
        "model='$model'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='$model'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done

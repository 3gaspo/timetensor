#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/test.out
#SBATCH --error=script_outputs/test.err
#SBATCH --time=01:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/test/"
model=tabpfn

lags=1344
horizon=336
for data in electricity
do
    for model in tabpfn
    do
        srun python3 test_pfn.py \
        "model='$model'" \
        "model.configs.cross_learning=False" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=None" \
        "data=$data" \
        "data.sampling.use_context=False" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='$model'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done
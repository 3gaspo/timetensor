#!/bin/bash
#SBATCH --job-name=augment
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/augment.out
#SBATCH --error=script_outputs/augment.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/fm/augment/"
model=chronos

# lags=1344
# horizon=336
# stride=512
# for data in electricity traffic solar
# do
#     for augment in 'sign-root' sign square root mirror 
#     do
#         srun python3 eval_users_script.py \
#         "model='$model'" \
#         "model.configs.self_augment='$augment'" \
#         "task.lags=$lags" \
#         "task.horizon=$horizon" \
#         "normalization=instance" \
#         "data=$data" \
#         "data.sampling.train_idx_mode='all'" \
#         "data.sampling.eval_idx_mode='all'" \
#         "data.sampling.train_stride=$stride" \
#         "data.sampling.eval_stride=$stride" \
#         "training.eval_runs=1" \
#         "misc.seed=$seed" \
#         "misc.device='cpu'" \
#         "misc.save_name='${model}_$augment'" \
#         "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     done
# done

# lags=672
# horizon=168
# stride=512
# for data in electricity traffic solar
# do
#     for augment in sign square root mirror 'sign-root'
#     do
#         srun python3 eval_users_script.py \
#         "model='$model'" \
#         "model.configs.self_augment='$augment'" \
#         "task.lags=$lags" \
#         "task.horizon=$horizon" \
#         "normalization=instance" \
#         "data=$data" \
#         "data.sampling.train_idx_mode='all'" \
#         "data.sampling.eval_idx_mode='all'" \
#         "data.sampling.train_stride=$stride" \
#         "data.sampling.eval_stride=$stride" \
#         "training.eval_runs=1" \
#         "misc.seed=$seed" \
#         "misc.device='cpu'" \
#         "misc.save_name='${model}_$augment'" \
#         "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     done
# done

# lags=168
# horizon=24
# stride=512
# for data in electricity traffic solar
# do
#     for augment in none sign square root mirror 'sign-root'
#     do
#         srun python3 eval_users_script.py \
#         "model='$model'" \
#         "model.configs.self_augment='$augment'" \
#         "task.lags=$lags" \
#         "task.horizon=$horizon" \
#         "normalization=instance" \
#         "data=$data" \
#         "data.sampling.train_idx_mode='all'" \
#         "data.sampling.eval_idx_mode='all'" \
#         "data.sampling.train_stride=$stride" \
#         "data.sampling.eval_stride=$stride" \
#         "training.eval_runs=1" \
#         "misc.seed=$seed" \
#         "misc.device='cpu'" \
#         "misc.save_name='${model}_$augment'" \
#         "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     done
# done

file_name=mean_results.json
settings="1344_336;672_168;168_24"
metric=nMSE
model=chronos

model_names="${model}_none;${model}_sign;${model}_square;${model}_root;${model}_mirror;${model}_square;${model}_sign-root"
prefix="${model}_augments"
output_path="${experiment_dir}${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

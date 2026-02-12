#!/bin/bash
#SBATCH --job-name=identity_cr
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/identity_cr.out
#SBATCH --error=script_outputs/identity_cr.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/identity/"

# lags=1344
# horizon=336
# stride=512
# for model in chronos
# do
#     for data in solar #electricity traffic solar exchange_rate
#     do
#         for use_context in False #True 
#         do
#             srun python3 identity_test.py \
#             "model='$model'" \
            # "model.configs.context_mode='future'" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=instance" \
#             "data=$data" \
#             "data.sampling.use_context=$use_context" \
#             "data.splits.date_splits=1" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='all'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=$stride" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=False" \
#             "data.sampling.remove_eval_cte=False" \
#             "training.complete_evaluation=False" \
#             "training.eval_runs=1" \
#             "misc.seed=$seed" \
#             "misc.device='cpu'" \
#             "misc.save_name=${model}_identity$use_context" \
#             "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#         done
#     done
# done



# lags=672
# horizon=168
# stride=512
# for model in chronos
# do
#     for data in solar #electricity traffic solar exchange_rate
#     do
#         for use_context in False #True 
#         do
#             srun python3 identity_test.py \
#             "model='$model'" \
#             "model.configs.context_mode='future'" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=instance" \
#             "data=$data" \
#             "data.sampling.use_context=$use_context" \
#             "data.splits.date_splits=1" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='all'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=$stride" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=False" \
#             "data.sampling.remove_eval_cte=False" \
#             "training.complete_evaluation=False" \
#             "training.eval_runs=1" \
#             "misc.seed=$seed" \
#             "misc.device='cpu'" \
#             "misc.save_name=${model}_identity$use_context" \
#             "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#         done
#     done
# done

# lags=168
# horizon=24
# stride=512
# for model in chronos
# do
#     for data in solar #electricity traffic solar exchange_rate
#     do
#         for use_context in False
#         do
#             srun python3 identity_test.py \
#             "model='$model'" \
#             "model.configs.context_mode='future'" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=instance" \
#             "data=$data" \
#             "data.sampling.use_context=$use_context" \
#             "data.splits.date_splits=1" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='all'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=$stride" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=False" \
#             "data.sampling.remove_eval_cte=False" \
#             "training.complete_evaluation=False" \
#             "training.eval_runs=1" \
#             "misc.seed=$seed" \
#             "misc.device='cpu'" \
#             "misc.save_name=${model}_identity$use_context" \
#             "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#         done
#     done
# done

# lags=24
# horizon=24
# stride=512
# for model in chronos
# do
#     for data in solar #electricity traffic solar exchange_rate
#     do
#         for use_context in False
#         do
#             srun python3 identity_test.py \
#             "model='$model'" \
#             "model.configs.context_mode='future'" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=instance" \
#             "data=$data" \
#             "data.sampling.use_context=$use_context" \
#             "data.splits.date_splits=1" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='all'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=$stride" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=False" \
#             "data.sampling.remove_eval_cte=False" \
#             "training.complete_evaluation=False" \
#             "training.eval_runs=1" \
#             "misc.seed=$seed" \
#             "misc.device='cpu'" \
#             "misc.save_name=${model}_identity$use_context" \
#             "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#         done
#     done
# done

settings="1344_336;672_168;168_24;24_24"
file_name=mean_results.json
metric=nMSE
output_path="${experiment_dir}${metric}_results.tex"

for model in chronos
do
    model_names="${model}_identityFalse;${model}_identityTrue"
    srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
done
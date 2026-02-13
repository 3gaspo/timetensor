#!/bin/bash
#SBATCH --job-name=cov_tab
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/cov_tab.out
#SBATCH --error=script_outputs/cov_tab.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/robust/"
model=tabpfn

lags=1344
horizon=336
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2
    do
        srun python3 covariates_test.py \
        "model='$model'" \
        "model.configs.self_augment='none'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=True" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "training.bs=$bs" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2
    do
        srun python3 covariates_test.py \
        "model='$model'" \
        "model.configs.self_augment='none'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=True" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "training.bs=$bs" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done

done


lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2
    do
        srun python3 covariates_test.py \
        "model='$model'" \
        "model.configs.self_augment='none'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=True" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "training.bs=$bs" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    for bs in 2
    do
        srun python3 covariates_test.py \
        "model='$model'" \
        "model.configs.self_augment='none'" \
        "task.lags=$lags" \
        "task.horizon=$horizon" \
        "normalization=instance" \
        "data=$data" \
        "data.sampling.use_context=True" \
        "data.splits.date_splits=1" \
        "data.splits.indiv_split=1" \
        "data.sampling.train_idx_mode='all'" \
        "data.sampling.eval_idx_mode='all'" \
        "data.sampling.train_stride=$stride" \
        "data.sampling.eval_stride=$stride" \
        "data.sampling.remove_train_cte=False" \
        "data.sampling.remove_eval_cte=False" \
        "training.complete_evaluation=False" \
        "training.eval_runs=1" \
        "training.bs=$bs" \
        "misc.seed=$seed" \
        "misc.device='cpu'" \
        "misc.save_name='${model}_neighbors$bs'" \
        "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    done
done


# settings="1344_336;672_168;168_24;24_24"
# file_name=mean_results.json
# model=chronos
# # metric=nMSE

# for metric in nMSE 'w10 nMSE' 'train time (min)' 'eval time (min)'
# do
#     output_path="${experiment_dir}neighbors_chronos_${metric}_results.tex"
#     model_names="${model}_none;${model}_sign;${model}_neighbors2;${model}_neighbors3;${model}_neighbors5;${model}_neighbors10;${model}_neighbors100;${model}_neighbors1000"
#     srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
# done
#!/bin/bash
#SBATCH --job-name=noise_fut_tab
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/noise_fut_tab.out
#SBATCH --error=script_outputs/noise_fut_tab.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/robust/"

lags=1344
horizon=336
stride=512
for model in tabpfn
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in "noise=1"
        do
            srun python3 eval_users.py \
            "model='$model'" \
            "model.configs.context_mode='future'" \
            "model.configs.augment_mode='future'" \
            "model.configs.self_augment='$augment'" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
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
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='${model}_${augment}_future'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

lags=672
horizon=168
stride=512
for model in tabpfn
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in "noise=1"
        do
            srun python3 eval_users.py \
            "model='$model'" \
            "model.configs.context_mode='future'" \
            "model.configs.augment_mode='future'" \
            "model.configs.self_augment='$augment'" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
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
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='${model}_${augment}_future'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

lags=168
horizon=24
stride=512
for model in tabpfn
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in "noise=1"
        do
            srun python3 eval_users.py \
            "model='$model'" \
            "model.configs.context_mode='future'" \
            "model.configs.augment_mode='future'" \
            "model.configs.self_augment='$augment'" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
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
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='${model}_${augment}_future'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done

lags=24
horizon=24
stride=512
for model in tabpfn
do
    for data in electricity traffic solar exchange_rate
    do
        for augment in "noise=1"
        do
            srun python3 eval_users.py \
            "model='$model'" \
            "model.configs.context_mode='future'" \
            "model.configs.augment_mode='future'" \
            "model.configs.self_augment='$augment'" \
            "task.lags=$lags" \
            "task.horizon=$horizon" \
            "normalization=instance" \
            "data=$data" \
            "data.sampling.use_context=False" \
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
            "misc.seed=$seed" \
            "misc.device='cpu'" \
            "misc.save_name='${model}_${augment}_future'" \
            "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
        done
    done
done


# file_name=test1_mean_results.json
# settings="1344_336;672_168;168_24;24_24"
# metric=nMSE

# for model in chronos
# do
#     #noises
#     model_names="${model}_none_future;${model}_noise=1_future;${model}_2noise=1_future;${model}_3noise=1_future;${model}_5noise=1_future;${model}_10noise=1_future;${model}_100noise=1_future"
#     prefix="${model}_noises_future"
#     output_path="${experiment_dir}${prefix}_${metric}_results.tex"
#     srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

# done
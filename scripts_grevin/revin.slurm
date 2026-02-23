#!/bin/bash
#SBATCH --job-name=in
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/in.out
#SBATCH --error=script_outputs/in.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/grevin/revin/"

# lags=1344
# horizon=336
# stride=512
# for data in electricity traffic solar
# do
#     normalization=standard
#     criterion=MSE
#     srun python3 train_model.py \
#     "model=PatchTST" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=$normalization" \
#     "data=$data" \
#     "data.sampling.use_context=False" \
#     "data.splits.date_splits=0.7" \
#     "data.splits.indiv_split=1" \
#     "data.sampling.train_idx_mode='individuals'" \
#     "data.sampling.eval_idx_mode='all'" \
#     "data.sampling.train_stride=1" \
#     "data.sampling.eval_stride=$stride" \
#     "data.sampling.remove_train_cte=True" \
#     "data.sampling.remove_eval_cte=True" \
#     "training.complete_evaluation=False" \
#     "training.bs=350" \
#     "training.epochs=1500" \
#     "training.eval_runs=1" \
#     "training.loss='$criterion'" \
#     "data.sampling.train_len_multiplier=1" \
#     "data.sampling.eval_len_multiplier=1" \
#     "misc.seed=$seed" \
#     "misc.save_name='PatchTST_${normalization}_$criterion'" \
#     "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"

#     for normalization in instance revin
#     do
#         for criterion in MSE nMSE
#         do
#             srun python3 train_model.py \
#             "model=PatchTST" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=$normalization" \
#             "data=$data" \
#             "data.sampling.use_context=False" \
#             "data.splits.date_splits=0.7" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='individuals'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=1" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=True" \
#             "data.sampling.remove_eval_cte=True" \
#             "training.complete_evaluation=False" \
#             "training.bs=350" \
#             "training.epochs=1500" \
#             "training.eval_runs=1" \
#             "training.loss='$criterion'" \
#             "data.sampling.train_len_multiplier=1" \
#             "data.sampling.eval_len_multiplier=1" \
#             "misc.seed=$seed" \
#             "misc.save_name='PatchTST_${normalization}_$criterion'" \
#             "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"
#         done
#     done
# done

# lags=672
# horizon=168
# stride=512
# for data in electricity traffic solar
# do
#     normalization=standard
#     criterion=MSE
#     srun python3 train_model.py \
#     "model=PatchTST" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=$normalization" \
#     "data=$data" \
#     "data.sampling.use_context=False" \
#     "data.splits.date_splits=0.7" \
#     "data.splits.indiv_split=1" \
#     "data.sampling.train_idx_mode='individuals'" \
#     "data.sampling.eval_idx_mode='all'" \
#     "data.sampling.train_stride=1" \
#     "data.sampling.eval_stride=$stride" \
#     "data.sampling.remove_train_cte=True" \
#     "data.sampling.remove_eval_cte=True" \
#     "training.complete_evaluation=False" \
#     "training.bs=350" \
#     "training.epochs=1500" \
#     "training.eval_runs=1" \
#     "training.loss='$criterion'" \
#     "data.sampling.train_len_multiplier=1" \
#     "data.sampling.eval_len_multiplier=1" \
#     "misc.seed=$seed" \
#     "misc.save_name='PatchTST_${normalization}_$criterion'" \
#     "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"

#     for normalization in instance revin
#     do
#         for criterion in MSE nMSE
#         do
#             srun python3 train_model.py \
#             "model=PatchTST" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=$normalization" \
#             "data=$data" \
#             "data.sampling.use_context=False" \
#             "data.splits.date_splits=0.7" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='individuals'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=1" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=True" \
#             "data.sampling.remove_eval_cte=True" \
#             "training.complete_evaluation=False" \
#             "training.bs=350" \
#             "training.epochs=1500" \
#             "training.eval_runs=1" \
#             "training.loss='$criterion'" \
#             "data.sampling.train_len_multiplier=1" \
#             "data.sampling.eval_len_multiplier=1" \
#             "misc.seed=$seed" \
#             "misc.save_name='PatchTST_${normalization}_$criterion'" \
#             "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"
#         done
#     done
# done

# lags=168
# horizon=24
# stride=512
# for data in electricity traffic solar
# do
#     normalization=standard
#     criterion=MSE
#     srun python3 train_model.py \
#     "model=PatchTST" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=$normalization" \
#     "data=$data" \
#     "data.sampling.use_context=False" \
#     "data.splits.date_splits=0.7" \
#     "data.splits.indiv_split=1" \
#     "data.sampling.train_idx_mode='individuals'" \
#     "data.sampling.eval_idx_mode='all'" \
#     "data.sampling.train_stride=1" \
#     "data.sampling.eval_stride=$stride" \
#     "data.sampling.remove_train_cte=True" \
#     "data.sampling.remove_eval_cte=True" \
#     "training.complete_evaluation=False" \
#     "training.bs=350" \
#     "training.epochs=1500" \
#     "training.eval_runs=1" \
#     "training.loss='$criterion'" \
#     "data.sampling.train_len_multiplier=1" \
#     "data.sampling.eval_len_multiplier=1" \
#     "misc.seed=$seed" \
#     "misc.save_name='PatchTST_${normalization}_$criterion'" \
#     "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"

#     for normalization in instance revin
#     do
#         for criterion in MSE nMSE
#         do
#             srun python3 train_model.py \
#             "model=PatchTST" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=$normalization" \
#             "data=$data" \
#             "data.sampling.use_context=False" \
#             "data.splits.date_splits=0.7" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='individuals'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=1" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=True" \
#             "data.sampling.remove_eval_cte=True" \
#             "training.complete_evaluation=False" \
#             "training.bs=350" \
#             "training.epochs=1500" \
#             "training.eval_runs=1" \
#             "training.loss='$criterion'" \
#             "data.sampling.train_len_multiplier=1" \
#             "data.sampling.eval_len_multiplier=1" \
#             "misc.seed=$seed" \
#             "misc.save_name='PatchTST_${normalization}_$criterion'" \
#             "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"
#         done
#     done
# done

# lags=24
# horizon=24
# stride=512
# for data in electricity traffic solar
# do
#     normalization=standard
#     criterion=MSE
#     srun python3 train_model.py \
#     "model=PatchTST" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=$normalization" \
#     "data=$data" \
#     "data.sampling.use_context=False" \
#     "data.splits.date_splits=0.7" \
#     "data.splits.indiv_split=1" \
#     "data.sampling.train_idx_mode='individuals'" \
#     "data.sampling.eval_idx_mode='all'" \
#     "data.sampling.train_stride=1" \
#     "data.sampling.eval_stride=$stride" \
#     "data.sampling.remove_train_cte=True" \
#     "data.sampling.remove_eval_cte=True" \
#     "training.complete_evaluation=False" \
#     "training.bs=350" \
#     "training.epochs=1500" \
#     "training.eval_runs=1" \
#     "training.loss='$criterion'" \
#     "data.sampling.train_len_multiplier=1" \
#     "data.sampling.eval_len_multiplier=1" \
#     "misc.seed=$seed" \
#     "misc.save_name='PatchTST_${normalization}_$criterion'" \
#     "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"

#     for normalization in instance revin
#     do
#         for criterion in MSE nMSE
#         do
#             srun python3 train_model.py \
#             "model=PatchTST" \
#             "task.lags=$lags" \
#             "task.horizon=$horizon" \
#             "normalization=$normalization" \
#             "data=$data" \
#             "data.sampling.use_context=False" \
#             "data.splits.date_splits=0.7" \
#             "data.splits.indiv_split=1" \
#             "data.sampling.train_idx_mode='individuals'" \
#             "data.sampling.eval_idx_mode='all'" \
#             "data.sampling.train_stride=1" \
#             "data.sampling.eval_stride=$stride" \
#             "data.sampling.remove_train_cte=True" \
#             "data.sampling.remove_eval_cte=True" \
#             "training.complete_evaluation=False" \
#             "training.bs=350" \
#             "training.epochs=1500" \
#             "training.eval_runs=1" \
#             "training.loss='$criterion'" \
#             "data.sampling.train_len_multiplier=1" \
#             "data.sampling.eval_len_multiplier=1" \
#             "misc.seed=$seed" \
#             "misc.save_name='PatchTST_${normalization}_$criterion'" \
#             "misc.output_dir='${experiment_dir}$data/${lags}_$horizon/'"
#         done
#     done
# done

settings="1344_336;672_168;168_24;24_24"
model_names="PatchTST_standard_MSE;PatchTST_instance_MSE;PatchTST_instance_nMSE;PatchTST_revin_MSE;PatchTST_revin_nMSE"

for metric in nMSE 'train time (min)'
do
    for prefix in train test1
    do
        file_name="${prefix}_mean_results.json"
        output_path="${experiment_dir}${prefix}_${metric}_results.tex"
        srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
    done
done


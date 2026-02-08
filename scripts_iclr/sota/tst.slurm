#!/bin/bash
#SBATCH --job-name=tst
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tst.out
#SBATCH --error=script_outputs/tst.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/sota/"

lags=1344
horizon=336
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1500" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "training.init='${experiment_dir}${data}/${lags}_$horizon/PatchTST/trained_model.pt'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='all'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done

lags=672
horizon=168
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1500" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "training.init='${experiment_dir}${data}/${lags}_$horizon/PatchTST/trained_model.pt'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='all'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done


lags=168
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1500" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "training.init='${experiment_dir}${data}/${lags}_$horizon/PatchTST/trained_model.pt'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='all'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done

lags=24
horizon=24
stride=512
for data in electricity traffic solar exchange_rate
do
    srun python3 train_model.py \
    "model=PatchTST" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='individuals'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=1" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1500" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    srun python3 eval_users.py \
    "model=PatchTST" \
    "training.init='${experiment_dir}${data}/${lags}_$horizon/PatchTST/trained_model.pt'" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.use_context=False" \
    "data.splits.date_splits=0.7" \
    "data.splits.indiv_split=1" \
    "data.sampling.train_idx_mode='all'" \
    "data.sampling.eval_idx_mode='all'" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "data.sampling.remove_train_cte=True" \
    "data.sampling.remove_eval_cte=True" \
    "training.complete_evaluation=False" \
    "training.bs=350" \
    "training.epochs=1" \
    "training.eval_runs=1" \
    "data.sampling.train_len_multiplier=1" \
    "data.sampling.eval_len_multiplier=1" \
    "misc.seed=$seed" \
    "misc.save_name='PatchTST'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done

settings="1344_336;672_168;168_24;24_24"
model_names="PatchTST;chronos2"

for metric in nMSE 'w10 nMSE' 'train time (min)' 'eval time (min)'
do
    for prefix in train test1
    do
        file_name="${prefix}_mean_results.json"
        output_path="${experiment_dir}${prefix}_${metric}_results.tex"
        srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
    done
done
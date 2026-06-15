#!/bin/bash
#SBATCH --job-name=conso
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/conso.out
#SBATCH --error=script_outputs/conso.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/conso/"
data=conso

# for model in chronos
# do
#     lags=1344
#     horizon=336
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

#     lags=672
#     horizon=168
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

#     lags=168
#     horizon=24
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    
#     lags=24
#     horizon=24
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
# done


use_time_features=False
for model in tabpfn
do
    lags=1344
    horizon=336
    stride=512
    #context
    bs=2
    srun python3 conso_script.py \
    "model='$model'" \
    "data=$data" \
    "model.configs.context_mode=future" \
    "model.configs.use_time_features=$use_time_features" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data=$data" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=$bs" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_context_notime'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    lags=672
    horizon=168
    stride=512
    #context
    bs=2
    srun python3 conso_script.py \
    "model='$model'" \
    "data=$data" \
    "model.configs.context_mode=future" \
    "model.configs.use_time_features=$use_time_features" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=$bs" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_context_notime'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

    lags=168
    horizon=24
    stride=512
    #context
    bs=2
    srun python3 conso_script.py \
    "model='$model'" \
    "data=$data" \
    "model.configs.context_mode=future" \
    "model.configs.use_time_features=$use_time_features" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=$bs" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_context_notime'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    
    lags=24
    horizon=24
    stride=512
    #context
    bs=2
    srun python3 conso_script.py \
    "model='$model'" \
    "data=$data" \
    "model.configs.context_mode=future" \
    "model.configs.use_time_features=$use_time_features" \
    "task.lags=$lags" \
    "task.horizon=$horizon" \
    "normalization=instance" \
    "data.sampling.train_stride=$stride" \
    "data.sampling.eval_stride=$stride" \
    "training.bs=$bs" \
    "training.eval_runs=1" \
    "misc.seed=$seed" \
    "misc.device='cpu'" \
    "misc.save_name='${model}_context_notime'" \
    "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
done


# use_time_features=True
# for model in tabpfn
# do
#     lags=1344
#     horizon=336
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

#     lags=672
#     horizon=168
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"

#     lags=168
#     horizon=24
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
    
#     lags=24
#     horizon=24
#     stride=512
#     #context
#     bs=2
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "model.configs.context_mode=future" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=$bs" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_context'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
#     #bs1
#     srun python3 conso_script.py \
#     "model='$model'" \
#     "data=$data" \
#     "task.lags=$lags" \
#     "task.horizon=$horizon" \
#     "normalization=instance" \
#     "data=$data" \
#     "data.sampling.train_stride=$stride" \
#     "data.sampling.eval_stride=$stride" \
#     "training.bs=1" \
#     "training.eval_runs=1" \
#     "misc.seed=$seed" \
#     "misc.device='cpu'" \
#     "misc.save_name='${model}_baseline'" \
#     "misc.output_dir='${experiment_dir}${data}/${lags}_$horizon/'"
# done

settings="1344_336;672_168;168_24;24_24"
file_name=mean_results.json
model=chronos

model_names="chronos_baseline;chronos_context;tabpfn_baseline;tabpfn_context;tabpfn_context_notime"
metric=nMSE
output_path="${experiment_dir}neighbors_chronos_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

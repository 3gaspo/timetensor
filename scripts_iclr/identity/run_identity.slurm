#!/bin/bash
#SBATCH --job-name=RUN
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/RUN.out
#SBATCH --error=script_outputs/RUN.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/identity/"

settings="1344_336;672_168;168_24;24_24"
metric=nMSE

model_names="chronos_identityFalse;chronos_identityTrue"
prefix="F_cr"
file_name="mean_results.json"
output_path="${experiment_dir}${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

model_names="tabpfn_identityFalse;tabpfn_features_identityTrue"
prefix="F_tab"
file_name="mean_results.json"
output_path="${experiment_dir}${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"

model_names="chronos_identityTrue;tabpfn_features_identityTrue"
prefix="F_cr_v_tab"
file_name="mean_results.json"
output_path="${experiment_dir}${prefix}_${metric}_results.tex"
srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
#!/bin/bash
#SBATCH --job-name=read
#SBATCH --gres=gpu:1
#SBATCH --output=scripts/read.out
#SBATCH --error=scripts/read.err
#SBATCH --time=01:00:00
#SBATCH --mem=10G
source .venv/bin/activate

expe_name=bench_min_1
names="min_init;min_fixed;min_init+revin;min_fixed+revin;min_init+gamma;min_fixed+gamma"
baseline="min_init"
output_dir="../outputs/$expe_name/"
save_path="../outputs/$expe_name/plots/"
eval_freq=100

multipliers="3 3 4 6"
srun python3 -c "from src.timetensor.visu import get_multiple_expe_results;get_multiple_expe_results('${output_dir}', 'test1_mean_results.json', 5, multipliers='$multipliers', names='$names', baseline='$baseline', save_path='$save_path', save_name='errors1.pdf')"
srun python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$output_dir', 'test1_mean_results.json', 5, 'Test MSE', names='$names', baseline='$baseline', save_path='$save_path', save_name='test1_boxplot.pdf')"

# nohup bash scripts/read.sh > read.log 2>&1 &
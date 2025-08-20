#!/bin/bash
#SBATCH --job-name=bench_min1 -p h100
#SBATCH --gres=gpu:1 -p h100
#SBATCH --output=scripts/bench_min1.out -p h100
#SBATCH --error=scripts/bench_min1.err -p h100
#SBATCH --time=23:00:00 -p h100
#SBATCH --mem=40G -p h100
source .venv/bin/activate

output_dir="../outputs/bench_min1/"

# rm -rf "$output_dir"
# mkdir -p "$output_dir"

# data=sim1
# seed=1

# model_name=PatchTST
# loss=NMSE
# lr=0.00001
# epochs=200

# for normalization in instance revin
# do
#     srun python3 train_model.py \
#         "misc.seed=$seed" \
#         "data.reshuffle=True" \
#         "misc.output_dir=$output_dir" \
#         "misc.benchmark=True" \
#         "model=$model_name" \
#         "normalization=$normalization" \
#         "training.loss=$loss" \
#         "training.bs=10" \
#         "training.lr=$lr" \
#         "training.epochs=$epochs" \
#         "training.eval_freq=100" \
#         "training.print_freq=100" \
#         "data=$data" \
#         "task.lags=100" \
#         "task.horizon=20"
# done

# normalization=mIN

# #min
# #default
# alpha=1.0
# beta=0.0
# use_gamma=False
# inverse_gamma=False
# fixed_beta=False
# fixed_gamma=False
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min"

# #init
# alpha=0.9485
# beta=0.7805
# use_gamma=False
# inverse_gamma=False
# fixed_beta=False
# fixed_gamma=False
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min_init"

# #fixed
# alpha=0.9485
# beta=0.7805
# use_gamma=False
# inverse_gamma=False
# fixed_beta=True
# fixed_gamma=True
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min_fixed"

# #min + revin
# #default
# alpha=1.0
# beta=0.0
# use_gamma=True
# inverse_gamma=True
# fixed_beta=False
# fixed_gamma=False
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min+revin"

# #init
# alpha=0.9485
# beta=0.7805
# use_gamma=True
# inverse_gamma=True
# fixed_beta=False
# fixed_gamma=False
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min_init+revin"

# #fixed
# alpha=0.9485
# beta=0.7805
# use_gamma=True
# inverse_gamma=True
# fixed_beta=True
# fixed_gamma=True
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min_fixed+revin"


# #min + gamma
# #default
# alpha=1.0
# beta=0.0
# use_gamma=True
# inverse_gamma=False
# fixed_beta=False
# fixed_gamma=False
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min+gamma"

# #init
# alpha=0.9485
# beta=0.7805
# use_gamma=True
# inverse_gamma=False
# fixed_beta=False
# fixed_gamma=False
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min_init+gamma"

# #fixed
# alpha=0.9485
# beta=0.7805
# use_gamma=True
# inverse_gamma=False
# fixed_beta=True
# fixed_gamma=True
# srun python3 train_model.py \
#     "misc.seed=$seed" \
#     "data.reshuffle=True" \
#     "misc.output_dir=$output_dir" \
#     "misc.benchmark=True" \
#     "model=$model_name" \
#     "normalization=$normalization" \
#     "training.loss=$loss" \
#     "training.bs=10" \
#     "training.lr=$lr" \
#     "training.epochs=$epochs" \
#     "training.eval_freq=100" \
#     "training.print_freq=100" \
#     "data=$data" \
#     "task.lags=100" \
#     "task.horizon=20" \
#     "normalization.configs.init_beta=$beta" \
#     "normalization.configs.init_alpha=$alpha" \
#     "normalization.configs.fixed_beta=$fixed_beta" \
#     "normalization.configs.fixed_alpha=$fixed_alpha " \
#     "normalization.configs.use_gamma=$use_gamma" \
#     "normalization.configs.inverse_gamma=$use_gamma" \
#     "misc.save_name=min_fixed+gamma"

# srun python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"


multipliers="3 3 4 6"
srun python3 -c "from src.timetensor.visu import get_expe_results;get_expe_results('${output_dir}', 'test1_mean_results.json', multipliers='$multipliers')"

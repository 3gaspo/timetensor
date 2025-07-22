source .venv/bin/activate

main_output_dir="../outputs/benchmark_min_1/"

# rm -rf "$main_output_dir"
# mkdir -p "$main_output_dir"

# data=sim1

# model_name=PatchTST
# loss=MSE
# lr=0.0001
# epochs=200

# for seed in 1 2 3
# do
#     output_dir="${main_output_dir}seed_$seed/"
#     for normalization in instance revin
#     do
#         python3 train_model.py \
#             "misc.seed=$seed" \
#             "model=$model_name" \
#             "misc.output_dir=$output_dir" \
#             "normalization=$normalization" \
#             "training.loss=$loss" \
#             "misc.benchmark=True" \
#             "training.bs=10" \
#             "training.lr=$lr" \
#             "training.epochs=$epochs" \
#             "training.eval_freq=100" \
#             "training.print_freq=100" \
#             "data=$data" \
#             "task.lags=100" \
#             "task.horizon=20"
#     done

#     normalization=mIN
#     alpha=0.95
#     beta=0.78


#     for use_gamma in False True
#     do
#         #default
#         python3 train_model.py \
#             "misc.seed=$seed" \
#             "model=$model_name" \
#             "misc.output_dir=$output_dir" \
#             "normalization=$normalization" \
#             "training.loss=$loss" \
#             "misc.benchmark=True" \
#             "training.bs=10" \
#             "training.lr=$lr" \
#             "training.epochs=$epochs" \
#             "training.eval_freq=50" \
#             "training.print_freq=100" \
#             "data=$data" \
#             "task.lags=100" \
#             "task.horizon=20" \
#             "normalization.configs.use_gamma=$use_gamma" \
#             "misc.save_name=default_gamma$use_gamma"

#         #init
#         python3 train_model.py \
#             "misc.seed=$seed" \
#             "model=$model_name" \
#             "misc.output_dir=$output_dir" \
#             "normalization=$normalization" \
#             "training.loss=$loss" \
#             "misc.benchmark=True" \
#             "training.bs=10" \
#             "training.lr=$lr" \
#             "training.epochs=$epochs" \
#             "training.eval_freq=50" \
#             "training.print_freq=100" \
#             "data=$data" \
#             "task.lags=100" \
#             "task.horizon=20" \
#             "normalization.configs.init_beta=$beta" \
#             "normalization.configs.init_alpha=$alpha" \
#             "normalization.configs.use_gamma=$use_gamma" \
#             "misc.save_name=init_gamma$use_gamma"

#         #fixed
#         python3 train_model.py \
#             "misc.seed=$seed" \
#             "model=$model_name" \
#             "misc.output_dir=$output_dir" \
#             "normalization=$normalization" \
#             "training.loss=$loss" \
#             "misc.benchmark=True" \
#             "training.bs=10" \
#             "training.lr=$lr" \
#             "training.epochs=$epochs" \
#             "training.eval_freq=50" \
#             "training.print_freq=100" \
#             "data=$data" \
#             "task.lags=100" \
#             "task.horizon=20" \
#             "normalization.configs.init_beta=$beta" \
#             "normalization.configs.init_alpha=$alpha" \
#             "normalization.configs.fixed_beta=True" \
#             "normalization.configs.fixed_alpha=True " \
#             "normalization.configs.use_gamma=$use_gamma" \
#             "misc.save_name=fixed_gamma$use_gamma"
#     done
    
#     python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

# done


multipliers="1 2 2 5"
#python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test1_mean_results.json', 3, multipliers='$multipliers')"

multipliers="3 3 4 6"
python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test1_mean_results.json', 3, multipliers='$multipliers', baseline='PatchTST_revin_MSE')"
python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 3, col='Test MSE', names=None, baseline=None, save_path='$main_output_dir')"



#python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test2_mean_results.json', 3, multipliers='$multipliers')"

# nohup bash scripts/benchmark_min_1.sh > scripts/benchmark_min_1.log 2>&1 &
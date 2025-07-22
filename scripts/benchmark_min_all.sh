source .venv/bin/activate

main_output_dir="../outputs/benchmark_min_all/"

# rm -rf "$main_output_dir"
# mkdir -p "$main_output_dir"

data=sim

model_name=PatchTST
loss=MSE
lr=0.0001
epochs=200

for seed in 1 2 3
do
    output_dir="${main_output_dir}seed_$seed/"
    
    # for normalization in instance revin
    # do
    #     python3 train_model.py \
    #         "misc.seed=$seed" \
    #         "model=$model_name" \
    #         "misc.output_dir=$output_dir" \
    #         "normalization=$normalization" \
    #         "training.loss=$loss" \
    #         "misc.benchmark=True" \
    #         "training.bs=10" \
    #         "training.lr=$lr" \
    #         "training.epochs=$epochs" \
    #         "training.eval_freq=50" \
    #         "training.print_freq=100" \
    #         "data=$data" \
    #         "task.lags=100" \
    #         "task.horizon=20"
    # done

    #default
    normalization=cmIN
    python3 train_model.py \
        "misc.seed=$seed" \
        "model=$model_name" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=50" \
        "training.print_freq=100" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.init_alphas=2" \
        "normalization.configs.init_betas=2" \
        "misc.save_name=cmin_default"

    #init
    normalization=cmIN
    python3 train_model.py \
        "misc.seed=$seed" \
        "model=$model_name" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=50" \
        "training.print_freq=100" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.init_alphas='0.95;0.95'" \
        "normalization.configs.init_betas='0.78;-0.78'" \
        "misc.save_name=cmin_init"

    #fixed
    normalization=cmIN
    python3 train_model.py \
        "misc.seed=$seed" \
        "model=$model_name" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=50" \
        "training.print_freq=100" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "normalization.configs.init_alphas='0.95;0.95'" \
        "normalization.configs.init_betas='0.78;-0.78'" \
        "normalization.configs.fixed_alpha=True" \
        "normalization.configs.fixed_beta=True" \
        "misc.save_name=cmin_fixed"

    python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"
done

multipliers="3 3 4 6"
python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test1_mean_results.json', 3, multipliers='$multipliers', baseline='PatchTST_revin_MSE')"
python3 -c "from src.timetensor.visu import print_nice_tables;print_nice_tables('$main_output_dir', 'test2_mean_results.json', 3, multipliers='$multipliers', baseline='PatchTST_revin_MSE')"
python3 -c "from src.timetensor.visu import get_boxplots;get_boxplots('$main_output_dir', 'test1_mean_results.json', 3, col='Test MSE', names=None, baseline=None, save_path='$main_output_dir')"

# nohup bash scripts/benchmark_min_all.sh > scripts/benchmark_min_all.log 2>&1 &
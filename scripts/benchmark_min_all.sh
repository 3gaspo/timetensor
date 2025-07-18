source .venv/bin/activate

output_dir="../outputs/benchmark_min_all/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

data=sim

model_name=PatchTST
loss=MSE
lr=0.0001
epochs=200

for normalization in instance revin
do
    python3 train_model.py \
        "model=${model_name}" \
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
        "task.horizon=20"
done

normalization=mIN
#fixed
python3 train_model.py \
    "model=${model_name}" \
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
    "normalization.configs.fixed_beta=True" \
    "normalization.configs.fixed_alpha=True " \
    "misc.save_name=fixed_centralized_min"

normalization=mIN
#fixed
python3 train_model.py \
    "model=${model_name}" \
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
    "normalization.configs.fixed_beta=True" \
    "normalization.configs.fixed_alpha=True" \
    "normalization.configs.use_gamma=True" \
    "misc.save_name=gamma_fixed_centralized_min"

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="2 1 2 5"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_min_all.sh > benchmark_min_all.log 2>&1 &
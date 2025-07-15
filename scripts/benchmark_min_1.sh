source .venv/bin/activate

output_dir="../outputs/benchmark_min_1/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

data=sim1

model_name=expected
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "misc.benchmark=True" \
    "data=$data"

model_name=PatchTST
loss=MSE
lr=0.0001
epochs=200
eval_freq=50
print_freq=100

for normalization in instance revin
do
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "model.normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=$print_freq" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20"
done

normalization=mIN
for use_gamma in False True
do
    #default
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "model.normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=$print_freq" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "model.configs.use_gamma=$use_gamma" \
        "misc.save_name=default_gamma$use_gamma"

    #init
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "model.normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=$print_freq" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "model.configs.init_beta=1.07" \
        "model.configs.init_alpha=0.88" \
        "model.configs.use_gamma=$use_gamma" \
        "misc.save_name=init_gamma$use_gamma"


    #fixed
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "model.normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=$eval_freq" \
        "training.print_freq=$print_freq" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "model.configs.init_beta=1.07" \
        "model.configs.init_alpha=0.88" \
        "model.configs.fixed_beta=True" \
        "model.configs.fixed_alpha=True " \
        "model.configs.use_gamma=$use_gamma" \
        "misc.save_name=fixed_gamma$use_gamma"
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="2 1 1 5"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_min_1.sh > benchmark_min_1.log 2>&1 &
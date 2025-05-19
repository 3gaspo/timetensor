source .venv/bin/activate
output_dir="../outputs/federated_benchmark_all/"

rm -rf "$output_dir"
mkdir -p "$output_dir"

python3 train_fedavg.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=3" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.loss=MMSE" \
    "fed=fed_all" \
    "fed.reset_revin=True" \
    "misc.save_name=DLinear_revin_resetTrue"

python3 train_fedavg.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=3" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.loss=MMSE" \
    "fed=fed_all" \
    "fed.reset_revin=False" \
    "misc.save_name=DLinear_revin_resetFalse"

python3 train_fedavg.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=2" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.loss=MMSE" \
    "misc.save_name=DLinear_inst" \
    "fed=fed_all"


multipliers="0 1 2 2 1 2 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}MMSE_mean_results.json', multipliers='$multipliers')"
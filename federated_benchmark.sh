source .venv/bin/activate

output_dir="../outputs/federated_benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"


python3 train_fedavg.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=2" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.loss=NMSE" \
    "misc.save_name=DLinear_inst"

python3 train_fedavg.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=3" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.loss=NMSE" \
    "fed.reset_revin=False" \
    "misc.save_name=DLinear_revin_resetFalse"

python3 train_fedavg.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=3" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.loss=NMSE" \
    "fed.reset_revin=True" \
    "misc.save_name=DLinear_revin_resetTrue"

python3 print_table.py "misc.output_dir=${output_dir}MMSE_" "misc.table_coeffs=3 3 3"
python3 print_table.py "misc.output_dir=${output_dir}NMSE_" "misc.table_coeffs=1 1 1"
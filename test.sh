source .venv/bin/activate

output_dir="../outputs/federated_benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for reset_revin in False True
do
    python3 train_fedrevin.py \
        "model.name=DLinear" \
        "model_configs=DLinear" \
        "misc.output_dir=$output_dir" \
        "model.normalization=3" \
        "misc.benchmark=True" \
        "training.retrain=True" \
        "training.loss=NMSE" \
        "fed.reset_revin=$reset_revin" \
        "misc.save_name=DLinear_revin_reset$reset_revin" \
        "fed=fed_subset"
done

python3 print_table.py "misc.output_dir=${output_dir}MMSE_" "misc.table_coeffs=3 3 3"
python3 print_table.py "misc.output_dir=${output_dir}NMSE_" "misc.table_coeffs=1 1 1"
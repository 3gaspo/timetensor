source .venv/bin/activate

output_dir="../outputs/federated_benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for model_name in DLinear
do
    for reset_revin in False True
    do
        python3 train_fedavg.py \
            "model.name=$model_name" \
            "model_configs=$model_name" \
            "misc.output_dir=$output_dir" \
            "model.normalization=3" \
            "misc.benchmark=True" \
            "training.retrain=True" \
            "training.loss=NMSE" \
            "fed.reset_revin=$reset_revin" \
            "misc.save_name=revin$reset_revin"
    done
done

python3 print_table.py "misc.output_dir=${output_dir}MMSE_" "misc.table_coeffs=3 3 3"
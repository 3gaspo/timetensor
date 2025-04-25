source .venv/bin/activate

output_dir="../outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"


for model_name in DLinear
do
    python3 train_model.py \
        "model.name=$model_name" \
        "model_configs=$model_name" \
        "misc.output_dir=$output_dir" \
        "model.normalization=3" \
        "misc.benchmark=True" \
        "training.retrain=True"
done

python3 losses.py "misc.output_dir=$output_dir"
python3 print_table.py "misc.output_dir=$output_dir"
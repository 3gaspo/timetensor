source .venv/bin/activate

output_dir="outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for model_name in persistence repeat lookback linear
do
    for revin in 0 1
    do
        python3 train_model.py \
            "model.name=$model_name" \
            "model_configs=$model_name" \
            "misc.output_dir=$output_dir" \
            "model.revin=$revin"
    done
done
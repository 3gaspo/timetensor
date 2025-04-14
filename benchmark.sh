source .venv/bin/activate

output_dir="../outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for model_name in linear
do
    for normalization in 0 1 2 3
    do
        python3 train_model.py \
            "model.name=$model_name" \
            "model_configs=$model_name" \
            "misc.output_dir=$output_dir" \
            "model.normalization=$normalization" \
            "misc.benchmark=True" \
            "training.retrain=True"
    done
done

python3 losses.py "misc.output_dir=$output_dir"

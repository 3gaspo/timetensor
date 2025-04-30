source .venv/bin/activate

output_dir="../outputs/benchmark/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

for model_name in persistence lookback sklinear
do
    python3 train_model.py \
        "model.name=$model_name" \
        "model_configs=$model_name" \
        "misc.output_dir=$output_dir" \
        "model.normalization=0" \
        "misc.benchmark=True" \
        "training.retrain=True" \
        "data.subset.train=1" \
        "data.subset.valid=1" \
        "data.subset.valid2=1" \
        "data.subset.test=1"
done

for loss in MSE NMSE
do
    for normalization in 1 2 3
    do
        python3 train_model.py \
            "model.name=linear" \
            "model_configs=linear" \
            "misc.output_dir=$output_dir" \
            "model.normalization=3" \
            "misc.benchmark=True" \
            "training.retrain=True" \
            "training.epochs=2" \
            "training.loss=$loss" \
            "data.subset.train=1" \
            "data.subset.valid=1" \
            "data.subset.valid2=1" \
            "data.subset.test=1"
    done
done

python3 train_model.py \
    "model.name=DLinear" \
    "model_configs=DLinear" \
    "misc.output_dir=$output_dir" \
    "model.normalization=3" \
    "misc.benchmark=True" \
    "training.retrain=True" \
    "training.epochs=2" \
    "training.loss=NMSE" \
    "data.subset.train=1" \
    "data.subset.valid=1" \
    "data.subset.valid2=1" \
    "data.subset.test=1"

python3 losses.py "misc.output_dir=$output_dir"
python3 print_table.py "misc.output_dir=$output_dir"
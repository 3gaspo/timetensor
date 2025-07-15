source .venv/bin/activate

output_dir="../outputs/benchmark_cluster_full/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

by_idx=individuals
model_name=PatchTST
lr=0.0001
cluster=synthetic_6way_cluster

normalization=instance
loss=normalize_y
python3 train_model.py \
    "model=${model_name}" \
    "data=$cluster" \
    "misc.output_dir=$output_dir" \
    "model.normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.bs=28" \
    "training.lr=$lr" \
    "training.epochs=1000" \
    "training.eval_freq=50" \
    "training.print_freq=100" \
    "data.by_idx=$by_idx" \
    "model.configs.latent=True"

normalization=mIN
loss=normalize_y
python3 train_model.py \
    "model=${model_name}" \
    "data=$cluster" \
    "misc.output_dir=$output_dir" \
    "model.normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.bs=28" \
    "training.lr=$lr" \
    "training.epochs=1000" \
    "training.eval_freq=50" \
    "training.print_freq=100" \
    "data.by_idx=$by_idx" \
    "model.configs.latent=True"


normalization=instance
loss=MSE
python3 train_model.py \
    "model=${model_name}" \
    "data=$cluster" \
    "misc.output_dir=$output_dir" \
    "model.normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.bs=28" \
    "training.lr=$lr" \
    "training.epochs=1000" \
    "training.eval_freq=50" \
    "training.print_freq=100" \
    "data.by_idx=$by_idx" \
    "model.configs.latent=False"

normalization=mIN
loss=MSE
python3 train_model.py \
    "model=${model_name}" \
    "data=$cluster" \
    "misc.output_dir=$output_dir" \
    "model.normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.bs=28" \
    "training.lr=$lr" \
    "training.epochs=1000" \
    "training.eval_freq=50" \
    "training.print_freq=100" \
    "data.by_idx=$by_idx" \
    "model.configs.latent=False"


python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir')"

multipliers="-3 0 2"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_cluster.sh > benchmark_cluster_full.log 2>&1 &
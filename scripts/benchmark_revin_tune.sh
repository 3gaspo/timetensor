source .venv/bin/activate

output_dir="../outputs/benchmark_revin_tune/"
rm -rf "$output_dir"
mkdir -p "$output_dir"

data=sim

model_name=PatchTST
loss=MSE
lr=0.0001
epochs=200

normalization=revin
python3 train_model.py \
    "model=${model_name}" \
    "misc.output_dir=$output_dir" \
    "normalization=$normalization" \
    "training.loss=$loss" \
    "misc.benchmark=True" \
    "training.bs=10" \
    "training.lr=$lr" \
    "training.epochs=$epochs" \
    "training.eval_freq=50" \
    "training.print_freq=100" \
    "data=$data" \
    "task.lags=100" \
    "task.horizon=20"


epochs=100
init_path="../outputs/benchmark_revin_tune/PatchTST_revin_MSE/trained_model.pt"
normalization=revin

for data in sim1 sim2
do

    #fine tune gammas
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=50" \
        "training.print_freq=100" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "training.init=$init_path" \
        "training.freeze_core=True" \
        "misc.save_name=finetune_$data"

    #no fine tuning
    python3 train_model.py \
        "model=${model_name}" \
        "misc.output_dir=$output_dir" \
        "normalization=$normalization" \
        "training.loss=$loss" \
        "misc.benchmark=True" \
        "training.bs=10" \
        "training.lr=$lr" \
        "training.epochs=$epochs" \
        "training.eval_freq=50" \
        "training.print_freq=100" \
        "data=$data" \
        "task.lags=100" \
        "task.horizon=20" \
        "training.init=$init_path" \
        "training.freeze_core=True" \
        "training.retrain=False" \
        "misc.save_name=nofinetune_$data"
done

python3 -c "from src.timetensor.visu import plot_expe;plot_expe('$output_dir', $eval_freq)"

multipliers="2 1 2 5"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test1_mean_results.json', multipliers='$multipliers')"
python3 -c "from src.timetensor.visu import print_nice_table;print_nice_table('${output_dir}test2_mean_results.json', multipliers='$multipliers')"

# nohup bash scripts/benchmark_revin_tune.sh > benchmark_revin_tune.log 2>&1 &
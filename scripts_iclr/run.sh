source .venv/bin/activate

seed=1
experiment_dir="../outputs/iclr/sota/"

settings="1344_336;672_168;168_24;24_24"
model_names="PatchTST;chronos2;tabpfn"

for metric in nMSE 'w10 nMSE' 'eval time (min)'
do
    for prefix in train test1
    do
        file_name="${prefix}_mean_results.json"
        output_path="${experiment_dir}${prefix}_${metric}_results.tex"
        srun python3 -c "from src.timetensor.visu import generate_results_table;generate_results_table('$experiment_dir', json_filename='$file_name', metric_key='$metric', output_tex_path='$output_path', model_names='$model_names', settings='$settings', decimals=4)"
    done
done
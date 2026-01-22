for job in chronos tst users self_augment
do
    sbatch scripts_fm/${job}.slurm
done

for job in clusters indiv loss
do
    sbatch scripts_fm/cross_learning_${job}.slurm
done
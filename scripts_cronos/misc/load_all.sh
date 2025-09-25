for data in ecl solar traffic
do
    sbatch scripts/${data}/load_${data}.slurm
done
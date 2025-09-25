# for data in ecl
# do
#     sbatch scripts/${data}/${data}_168_24.slurm
#     sbatch scripts/${data}/${data}_504_24.slurm
#     sbatch scripts/${data}/${data}_504_168.slurm
#     sbatch scripts/${data}/${data}_504_504.slurm
# done

# for data in solar traffic
# do
#     sbatch scripts/${data}/${data}_168_24.slurm
#     sbatch scripts/${data}/${data}_504_24.slurm
#     sbatch scripts/${data}/${data}_504_168.slurm
#     sbatch scripts/${data}/${data}_504_504.slurm
# done

for data in synthetic
do
    sbatch scripts/${data}/${data}_40_10.slurm
    sbatch scripts/${data}/${data}_100_20.slurm
    sbatch scripts/${data}/${data}_100_100.slurm
done
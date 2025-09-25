git add src
git add configs
git add load_dataset.py
git add train_model.py
git commit -m"latest"

git push
git push gitlab

scp -r src h61084@cronos.hpc.edf.fr:/home/h61084/
scp -r configs h61084@cronos.hpc.edf.fr:/home/h61084/
scp train_model.py h61084@cronos.hpc.edf.fr:/home/h61084/
scp load_dataset.py h61084@cronos.hpc.edf.fr:/home/h61084/
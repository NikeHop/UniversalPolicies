#! /bin/bash 

#SBATCH -p gpu_h100
#SBATCH -t 60:00:00
#SBATCH --mem=160G
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nhopner@gmail.com


source /projects/0/prjs1044/miniconda3/bin/activate
conda activate diffusion_nl 


python train.py --config ./configs/state/bosslevel.yaml  --experiment_name "standard_bosslevel" --data_path ../../data/BOSSLEVEL/standard_50000_4_0_False_demos/dataset_50000.pkl  --seed 1 --device 0
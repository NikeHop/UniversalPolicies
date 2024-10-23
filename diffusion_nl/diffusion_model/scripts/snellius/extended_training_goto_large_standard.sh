#! /bin/bash 

#SBATCH -p gpu_h100
#SBATCH -t 24:00:00
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


python train.py --config ./configs/state/go_to_large_edm.yaml --experiment_name "mixed_go_to_large_25000_extended_${1}" --data_path ../../data/GOTO_LARGE/mixed_25000_4_7_False_demos/dataset_25000.pkl  --checkpoint $2 --seed $1 --device 0
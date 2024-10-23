#! /bin/bash 

#SBATCH -p gpu_mig
#SBATCH -t 24:00:00
#SBATCH --mem=60G
#SBATCH -c 9
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nhopner@gmail.com


source /projects/0/prjs1044/miniconda3/bin/activate
conda activate diffusion_nl 


python train.py --config ./configs/state/distractors_edm.yaml --experiment_name "mixed_extended_${1}" --data_path ../../data/GOTO/mixed_5000_4_3_False_demos/dataset_5000.pkl  --checkpoint $2 --seed $1 --device 0
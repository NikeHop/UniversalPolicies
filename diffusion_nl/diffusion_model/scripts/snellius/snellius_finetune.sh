#! /bin/bash 

#SBATCH -p gpu_mig
#SBATCH -t 10:00:00
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


python train.py --config ./configs/state/distractors_edm.yaml --experiment_name "finetune_mixed_extended_${2}_${1}" --action_space $2 --data_path $4  --checkpoint $3 --seed $1 --device 0
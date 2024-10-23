#! /bin/bash 

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mem=120G
#SBATCH -c 18
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nhopner@gmail.com


source /projects/0/prjs1044/miniconda3/bin/activate
conda activate diffusion_nl 


bash ./scripts/instruction_imitation.sh
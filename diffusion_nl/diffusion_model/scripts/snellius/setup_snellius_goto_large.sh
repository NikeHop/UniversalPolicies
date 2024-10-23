#! /bin/bash

set -e 

sbatch ./scripts/snellius/extended_training_goto_large_standard.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_25000_go_to_large/DiffusionMultiAgent/4ncfdl08/epoch=1818-step=400000.ckpt
sleep 5
sbatch ./scripts/snellius/extended_training_goto_large_standard.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_25000_go_to_large/DiffusionMultiAgent/43gp8pny/epoch=1818-step=400000.ckpt
sleep 5
sbatch ./scripts/snellius/extended_training_goto_large_standard.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_25000_go_to_large/DiffusionMultiAgent/045c0gfg/epoch=1818-step=400000.ckpt


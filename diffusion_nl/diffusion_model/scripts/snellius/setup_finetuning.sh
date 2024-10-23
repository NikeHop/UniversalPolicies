#! /bin/bash 

for pair in "1 threlixw" "2 agvpx547" "3 wrri2q5o" "4 b5y7fdc3" 
do
    set -- $pair
    seed=$1
    model_id=$2
    
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 0 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/standard_5000_4_3_False_demos/dataset_5000.pkl "finetune_standard_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/no_left_5000_4_3_False_demos/dataset_5000.pkl "finetune_no_left_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/no_right_5000_4_3_False_demos/dataset_5000.pkl "finetune_no_right_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/diagonal_5000_4_3_False_demos/dataset_5000.pkl "finetune_diagonal_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/wsad_5000_4_3_False_demos/dataset_5000.pkl "finetune_wsad_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 5 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/dir8_5000_4_3_False_demos/dataset_5000.pkl "finetune_dir8_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 6 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/left_right_5000_4_3_False_demos/dataset_5000.pkl "finetune_left_right_5000_edm_${seed}"
    sbatch ./scripts/snellius/snellius_finetune.sh $seed 7 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_extended_${seed}/DiffusionMultiAgent/${model_id}/epoch=94-step=300000.ckpt   ../../data/GOTO/all_diagonal_5000_4_3_False_demos/dataset_5000.pkl "finetune_all_diagonal_5000_edm_${seed}"

done
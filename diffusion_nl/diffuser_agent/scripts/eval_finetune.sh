#! /bin/bash

set -e 

# Seed 4 

python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "finetune_mixed_extended_0_1/DiffusionMultiAgent" --dm_model_name "pzvdk8la" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_standard_eval_distractor_1  --device "cuda:0" 

python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "finetune_mixed_extended_1_1/DiffusionMultiAgent" --dm_model_name "uhaa28yc" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_no_left_eval_distractor_1  --device "cuda:0"  

python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "finetune_mixed_extended_2_1/DiffusionMultiAgent" --dm_model_name "zo33xzzv" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_no_right_eval_distractor_1  --device "cuda:0" 

python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "finetune_mixed_extended_3_1/DiffusionMultiAgent" --dm_model_name "b8lhhbjm" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_diagonal_eval_distractor_1  --device "cuda:0"  

python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "finetune_mixed_extended_4_1/DiffusionMultiAgent" --dm_model_name "oiwnsokn" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_wsad_eval_distractor_1  --device "cuda:0" 

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "finetune_mixed_extended_5_1/DiffusionMultiAgent" --dm_model_name "x9ssfp1r" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_dir8_eval_distractor_1  --device "cuda:0"  

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "finetune_mixed_extended_6_1/DiffusionMultiAgent" --dm_model_name "fi4a8fic" --dm_model_checkpoint "epoch=378-step=200000.ckpt"  --experiment_name finetune_left_right_eval_distractor_1  --device "cuda:0" 

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "finetune_mixed_extended_7_1/DiffusionMultiAgent" --dm_model_name "fdq73os1" --dm_model_checkpoint "epoch=378-step=200000.ckpt"  --experiment_name finetune_all_diagonal_eval_distractor_1  --device "cuda:0" 


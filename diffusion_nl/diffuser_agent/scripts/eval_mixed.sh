#! /bin/bash

set -e 

for pair in "1 threlixw" "2 agvpx547" "3 wrri2q5o" "4 b5y7fdc3"
do

set -- $pair 
seed=$1
model_id=$2

python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name standard_distractor_plain_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name  no_left_distractor_plain_mixed_${seed}  --device "cuda:0"  

python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name no_right_distractor_plain_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name diagonal_distractor_plain_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name wsad_distractor_plain_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name dir8_distractor_plain_mixed_${seed} --device "cuda:0"

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name  left_right_distractor_plain_mixed_${seed}  --device "cuda:0" 

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name all_diagonal_distractor_plain_mixed_${seed}  --device "cuda:0" 

done



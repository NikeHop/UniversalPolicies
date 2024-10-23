#! /bin/bash

set -e 

for pair in  "3 c5zvdqmo" "4 rnzia2m1"
do

set -- $pair 
seed=$1
model_id=$2

python ./eval.py --config ./configs/standard/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt" --experiment_name standard_distractor_agent_type_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/no_left/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt" --experiment_name  no_left_distractor_agent_type_mixed_${seed}  --device "cuda:0"  

python ./eval.py --config ./configs/no_right/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt" --experiment_name no_right_distractor_agent_type_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/diagonal/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt" --experiment_name diagonal_distractor_agent_type_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/wsad/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt" --experiment_name wsad_distractor_agent_type_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/dir8/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt" --experiment_name dir8_distractor_agent_type_mixed_${seed} --device "cuda:0"

python ./eval.py --config ./configs/left_right/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt"  --experiment_name  left_right_distractor_agent_type_mixed_${seed}  --device "cuda:0" 

python ./eval.py --config ./configs/all_diagonal/distractors_agent_type.yaml --dm_model_path "mixed_agent_type_5000_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=315-step=1000000.ckpt"  --experiment_name all_diagonal_distractor_agent_type_mixed_${seed}  --device "cuda:0" 

done

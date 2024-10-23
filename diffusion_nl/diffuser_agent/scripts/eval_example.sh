#! /bin/bash

set -e 

for pair in  "3 e4vc1jew" "4 99ndp8a2"
do

set -- $pair 
seed=$1
model_id=$2

python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl" --experiment_name standard_distractor_example_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl" --experiment_name  no_left_distractor_example_mixed_${seed}  --device "cuda:0"  

python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl" --experiment_name no_right_distractor_example_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl" --experiment_name diagonal_distractor_example_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl" --experiment_name wsad_distractor_example_mixed_${seed}   --device "cuda:0" 

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl" --experiment_name dir8_distractor_example_mixed_${seed} --device "cuda:0"

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl"  --experiment_name  left_right_distractor_example_mixed_${seed}  --device "cuda:0" 

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_example_extended_${seed}/DiffusionMultiAgent" --dm_model_name $model_id --dm_model_checkpoint "epoch=157-step=500000.ckpt" --example_path "../../data/GOTO/GOTO_1_3_full_action_space_random.pkl"  --experiment_name all_diagonal_distractor_example_mixed_${seed}  --device "cuda:0" 

done
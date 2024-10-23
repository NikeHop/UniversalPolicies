#! /bin/bash 

set -e 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_25000_action_space_go_to_large/DiffusionMultiAgent" --dm_model_name "hcfdswzy" --dm_model_checkpoint "epoch=2272-step=500000.ckpt" --experiment_name mixed_go_to_large_action_space_standard_1   --device "cuda:0" 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_2/DiffusionMultiAgent" --dm_model_name "2ubc2k5u" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name mixed_go_to_large_action_space_standard_2   --device "cuda:0" 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name mixed_go_to_large_action_space_standard_3   --device "cuda:0" 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_4/DiffusionMultiAgent" --dm_model_name "tulihdrn" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name mixed_go_to_large_action_space_standard_4   --device "cuda:0" 
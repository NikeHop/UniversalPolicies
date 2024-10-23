#! /bin/bash

set -e 


#python ./eval.py --config ./configs/standard/bosslevel.yaml --dm_model_path "standard_bosslevel/DiffusionMultiAgent" --dm_model_name "s5t0ecax" --dm_model_checkpoint "epoch=341-step=300000.ckpt" --experiment_name bosslevel_test   --device "cuda:0" 


python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_25000_action_space_go_to_large/DiffusionMultiAgent" --dm_model_name "hcfdswzy" --dm_model_checkpoint "epoch=2272-step=500000.ckpt" --experiment_name mixed_action_space_go_to_large_test   --device "cuda:0" 
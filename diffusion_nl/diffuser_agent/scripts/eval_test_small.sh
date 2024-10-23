#! /bin/bash

set -e 

# Seed 1
python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_go_to_large/DiffusionMultiAgent" --dm_model_name "bq3gnvfs" --dm_model_checkpoint "epoch=3278-step=400000.ckpt" --experiment_name go_to_large_standard_test   --device "cuda:0"




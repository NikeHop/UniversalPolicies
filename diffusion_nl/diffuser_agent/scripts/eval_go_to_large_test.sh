#! /bin/bash 

set -e 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_25000_go_to_large/DiffusionMultiAgent" --dm_model_name "6hzvy08u" --dm_model_checkpoint "epoch=2272-step=500000.ckpt" --experiment_name mixed_go_to_large_standard_1   --device "cuda:0" 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_go_to_large_25000_extended_2/DiffusionMultiAgent" --dm_model_name "sp9dpkpt" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name mixed_go_to_large_standard_2   --device "cuda:0" 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_go_to_large_25000_extended_3/DiffusionMultiAgent" --dm_model_name "76dpogt0" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name mixed_go_to_large_standard_3   --device "cuda:0" 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_go_to_large_25000_extended_4/DiffusionMultiAgent" --dm_model_name "jaxkq4yr" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name mixed_go_to_large_standard_4   --device "cuda:0" 
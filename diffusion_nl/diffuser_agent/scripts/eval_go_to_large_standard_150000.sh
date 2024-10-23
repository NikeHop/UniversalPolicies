#! /bin/bash 

set -e 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_go_to_large/DiffusionMultiAgent" --dm_model_name "818kxilw" --dm_model_checkpoint "epoch=13513-step=500000.ckpt" --experiment_name standard_25000_eval_1  --device "cuda:0" 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_extended_2/DiffusionMultiAgent" --dm_model_name "xi20kc5n" --dm_model_checkpoint "epoch=2702-step=100000.ckpt" --experiment_name standard_25000_eval_2   --device "cuda:0" 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_extended_3/DiffusionMultiAgent" --dm_model_name "eyjw0xxf" --dm_model_checkpoint "epoch=2702-step=100000.ckpt" --experiment_name standard_25000_eval_3   --device "cuda:0" 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_extended_4/DiffusionMultiAgent" --dm_model_name "fqdeb0nq" --dm_model_checkpoint "epoch=2702-step=100000.ckpt" --experiment_name standard_25000_eval_4   --device "cuda:0" 


#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_go_to_large_150000/DiffusionMultiAgent" --dm_model_name "wt0nxgsc" --dm_model_checkpoint "epoch=2272-step=500000.ckpt" --experiment_name standard_150000_eval_1  --device "cuda:0" 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_extended_2/DiffusionMultiAgent" --dm_model_name "11s4f67h" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name standard_150000_eval_2   --device "cuda:0" 

#python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_extended_3/DiffusionMultiAgent" --dm_model_name "9gwyblzx" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name standard_150000_eval_3   --device "cuda:0" 

python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "standard_extended_4/DiffusionMultiAgent" --dm_model_name "m719wg5l" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name standard_150000_eval_4   --device "cuda:0" 
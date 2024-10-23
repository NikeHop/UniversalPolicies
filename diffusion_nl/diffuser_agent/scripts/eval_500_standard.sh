#! /bin/bash

python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "500_standard/DiffusionMultiAgent" --dm_model_name "z35ytg5h" --dm_model_checkpoint "epoch=943-step=100000.ckpt"  --device "cuda:1" 

python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "500_standard/DiffusionMultiAgent" --dm_model_name "z35ytg5h" --dm_model_checkpoint "epoch=1886-step=200000.ckpt"  --device "cuda:1" 

python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "500_standard/DiffusionMultiAgent" --dm_model_name "z35ytg5h" --dm_model_checkpoint "epoch=2830-step=300000.ckpt"  --device "cuda:1" 

python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "500_standard/DiffusionMultiAgent" --dm_model_name "z35ytg5h" --dm_model_checkpoint "epoch=3773-step=400000.ckpt"  --device "cuda:1"
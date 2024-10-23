#! /bin/bash

python ./eval.py --config ./configs/no_right/basic.yaml --dm_model_path "500_no_right/DiffusionMultiAgent" --dm_model_name "izf47l96" --dm_model_checkpoint "epoch=4716-step=500000.ckpt"  --device "cuda:3"  

python ./eval.py --config ./configs/no_left/basic.yaml --dm_model_path "500_no_left/DiffusionMultiAgent" --dm_model_name "cmooyfav" --dm_model_checkpoint "epoch=4716-step=500000.ckpt"  --device "cuda:3" 

python ./eval.py --config ./configs/diagonal/basic.yaml --dm_model_path "500_diagonal/DiffusionMultiAgent" --dm_model_name "iewhrj3w" --dm_model_checkpoint "epoch=4716-step=500000.ckpt"  --device "cuda:3"

python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "500_standard/DiffusionMultiAgent" --dm_model_name "4omttmxx" --dm_model_checkpoint "epoch=4716-step=500000.ckpt"  --device "cuda:3" 



#! /bin/bash

python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "125_mixed/DiffusionMultiAgent" --dm_model_name "av5sr358" --dm_model_checkpoint "checkpoints/epoch=1879-step=199280.ckpt"  --device "cuda:0" &
pid1=$!

python ./eval.py --config ./configs/no_left/basic.yaml --dm_model_path "125_mixed/DiffusionMultiAgent" --dm_model_name "av5sr358" --dm_model_checkpoint "checkpoints/epoch=1879-step=199280.ckpt"  --device "cuda:1" & 
pid2=$!

python ./eval.py --config ./configs/no_right/basic.yaml --dm_model_path "125_mixed/DiffusionMultiAgent" --dm_model_name "av5sr358" --dm_model_checkpoint "checkpoints/epoch=1879-step=199280.ckpt"  --device "cuda:2" & 
pid3=$!

python ./eval.py --config ./configs/diagonal/basic.yaml --dm_model_path "125_mixed/DiffusionMultiAgent" --dm_model_name "av5sr358" --dm_model_checkpoint "checkpoints/epoch=1879-step=199280.ckpt"  --device "cuda:3" & 
pid4=$!

wait $pid1 $pid2 $pid3 $pid4 
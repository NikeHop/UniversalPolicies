#! /bin/bash

python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "sample_efficiency_diffusion_planner_50000_distractors_3/DiffusionMultiAgent" --dm_model_name "9qklkn9p" --dm_model_checkpoint "epoch=189-step=500000.ckpt"  --device "cuda:0"  
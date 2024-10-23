#! /bin/bash

set -e 


python ./eval.py --config ./configs/standard/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt" --experiment_name mixed_500_agent_with_type_standard_distractor   --device "cuda:0" &
pid1=$!
sleep 5

python ./eval.py --config ./configs/no_left/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt" --experiment_name mixed_500_agent_with_type_no_left_distractor   --device "cuda:1"  &
pid2=$!
sleep 5

python ./eval.py --config ./configs/no_right/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt" --experiment_name mixed_500_agent_with_type_no_right_distractor   --device "cuda:2" &
pid3=$!
sleep 5

python ./eval.py --config ./configs/diagonal/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt" --experiment_name mixed_500_agent_with_type_diagonal_distractor   --device "cuda:3" & 
pid4=$!

wait $pid1 $pid2 $pid3 $pid4


python ./eval.py --config ./configs/wsad/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt" --experiment_name mixed_500_agent_with_type_wsad_distractor   --device "cuda:0" &
pid1=$!
sleep 5

python ./eval.py --config ./configs/dir8/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt" --experiment_name mixed_500_agent_with_type_dir8_distractor   --device "cuda:1"  &
pid2=$!
sleep 5

python ./eval.py --config ./configs/left_right/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt"  --experiment_name mixed_500_agent_with_type_left_right_distractor  --device "cuda:2" &
pid3=$!
sleep 5

python ./eval.py --config ./configs/all_diagonal/basic.yaml --dm_model_path "mixed_500_with_agent_type/DiffusionMultiAgent" --dm_model_name "0onbcz01" --dm_model_checkpoint "epoch=1582-step=500000.ckpt"  --experiment_name mixed_500_agent_with_type_all_diagonal_distractor  --device "cuda:3" &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4
#! /bin/bash

set -e 


python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "finetune_with_agent_id_standard/DiffusionMultiAgent" --dm_model_name "tei6fus1" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_standard_eval_agent_id  --device "cuda:0" &
pid1=$!
sleep 5

python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "finetune_with_agent_id_no_left/DiffusionMultiAgent" --dm_model_name "jerh658a" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_no_left_eval_agent_id  --device "cuda:1"  &
pid2=$!
sleep 5

python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "finetune_with_agent_id_no_right/DiffusionMultiAgent" --dm_model_name "zf60uwsk" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_no_right_eval_agent_id  --device "cuda:2" &
pid3=$!
sleep 5

python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "finetune_with_agent_id_diagonal/DiffusionMultiAgent" --dm_model_name "d3gcxagl" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_diagonal_eval_agent_id  --device "cuda:3" & 
pid4=$!

wait $pid1 $pid2 $pid3 $pid4


python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "finetune_with_agent_id_wsad/DiffusionMultiAgent" --dm_model_name "ehpm6dov" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_wsad_eval_agent_id  --device "cuda:0" &
pid1=$!
sleep 5

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "finetune_with_agent_id_dir8/DiffusionMultiAgent" --dm_model_name "euh7nzs9" --dm_model_checkpoint "epoch=378-step=200000.ckpt" --experiment_name finetune_dir8_eval_agent_id  --device "cuda:1"  &
pid2=$!
sleep 5

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "finetune_with_agent_id_left_right/DiffusionMultiAgent" --dm_model_name "e2dqz6t7" --dm_model_checkpoint "epoch=378-step=200000.ckpt"  --experiment_name finetune_left_right_eval_agent_id  --device "cuda:2" &
pid3=$!
sleep 5

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "finetune_with_agent_id_all_diagonal/DiffusionMultiAgent" --dm_model_name "jofdki1o" --dm_model_checkpoint "epoch=378-step=200000.ckpt"  --experiment_name finetune_all_diagonal_eval_agent_id  --device "cuda:3" &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

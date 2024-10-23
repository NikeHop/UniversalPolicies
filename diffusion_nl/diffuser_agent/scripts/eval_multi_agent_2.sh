#! /bin/bash

set -e 


python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt" --experiment_name examples_1000_standard_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl  --device "cuda:0" &
pid1=$!
sleep 5

python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt" --experiment_name examples_1000_no_left_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl  --device "cuda:1"  &
pid2=$!
sleep 5

python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt" --experiment_name examples_1000_no_right_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl  --device "cuda:2" &
pid3=$!
sleep 5

python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt" --experiment_name examples_1000_diagonal_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl  --device "cuda:3" & 
pid4=$!

wait $pid1 $pid2 $pid3 $pid4


python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt" --experiment_name examples_1000_wsad_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl  --device "cuda:0" &
pid1=$!
sleep 5

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt" --experiment_name examples_1000_dir8_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl  --device "cuda:1"  &
pid2=$!
sleep 5

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt"  --experiment_name examples_1000_left_right_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl --device "cuda:2" &
pid3=$!
sleep 5

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_agent_example_1000/DiffusionMultiAgent" --dm_model_name "qyf6pjgz" --dm_model_checkpoint "epoch=303-step=800000.ckpt"  --experiment_name examples_1000_all_diagonal_distractor --example_path /var/scratch/nrhopner/experiments/diffusion_nl/data/GOTO/GOTO_1000_3_full_action_space_random.pkl --device "cuda:3" &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4

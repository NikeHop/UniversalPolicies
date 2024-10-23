#! /bin/bash

for action_space in 0 1 2 3 4 5 6 7
do 

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/0s58dkmr/checkpoints/epoch=29-step=101250.ckpt --device 0 --experiment_name "eval_distractor_instruction_imitation_complete_as_new_${action_space}_1" &
pid1=$!
sleep 1

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/c5icfv6z/checkpoints/epoch=29-step=101250.ckpt   --device 0 --experiment_name "eval_distractor_instruction_imitation_complete_as_new_${action_space}_2" &
pid2=$!
sleep 1

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/6tw0ndzf/checkpoints/epoch=29-step=101250.ckpt --device 0 --experiment_name "eval_distractor_instruction_imitation_complete_as_new_${action_space}_3" &
pid3=$!
sleep 1

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/dr2uhs29/checkpoints/epoch=29-step=101250.ckpt  --device 0 --experiment_name "eval_distractor_instruction_imitation_complete_as_new_${action_space}_4" &
pid4=$!
sleep 1

wait $pid1 $pid2 $pid3 $pid4

done

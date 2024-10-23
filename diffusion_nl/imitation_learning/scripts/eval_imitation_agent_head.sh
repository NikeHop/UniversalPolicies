#! /bin/bash

for action_space in 0 1 2 3 4 5 6 7
do 

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt --device 0 --experiment_name "eval_distractor_instruction_imitation_agent_head_new_${action_space}_1" &
pid1=$!
sleep 1

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --device 0 --experiment_name "eval_distractor_instruction_imitation_agent_head_new_${action_space}_2" &
pid2=$!
sleep 1

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/l0erdpch/checkpoints/epoch=19-step=67500.ckpt --device 0 --experiment_name "eval_distractor_instruction_imitation_agent_head_new_${action_space}_3" &
pid3=$!
sleep 1

python eval.py --config ./configs/eval_instruction_imitation.yaml --action_space $action_space --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/8s1b564p/checkpoints/epoch=19-step=67500.ckpt  --device 0 --experiment_name "eval_distractor_instruction_imitation_agent_head_new_${action_space}_4" &
pid4=$!
sleep 1

wait $pid1 $pid2 $pid3 $pid4

done

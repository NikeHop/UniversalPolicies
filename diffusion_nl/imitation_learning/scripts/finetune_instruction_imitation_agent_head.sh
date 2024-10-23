set -e

seed=2

python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --action_space 0 --datadirectory "../../data/GOTO/standard_5000_4_3_False_demos" --experiment_name "finetune_agent_head_standard_5000_${seed}" --seed $seed --device 0  &
pid1=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt    --action_space 1 --datadirectory "../../data/GOTO/no_left_5000_4_3_False_demos" --experiment_name "finetune_agent_head_no_left_5000_${seed}" --seed $seed --device 0 &
pid2=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint  /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt  --action_space 2 --datadirectory "../../data/GOTO/no_right_5000_4_3_False_demos" --experiment_name "finetune_agent_head_no_right_5000_${seed}" --seed $seed --device 0 &
pid3=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --action_space 3 --datadirectory "../../data/GOTO/diagonal_5000_4_3_False_demos" --experiment_name "finetune_agent_head_diagonal_5000_${seed}" --seed $seed --device 0 &
pid4=$!
sleep 5

wait $pid1 $pid2 $pid3 $pid4

python train.py --config ./configs/agent_heads.yaml --checkpoint  /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --action_space 4 --datadirectory "../../data/GOTO/wsad_5000_4_3_False_demos" --experiment_name "finetune_agent_head_wsad_5000_${seed}" --seed $seed --device 0 &
pid1=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint  /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --action_space 5 --datadirectory "../../data/GOTO/dir8_5000_4_3_False_demos" --experiment_name "finetune_agent_head_dir8_5000_${seed}" --seed $seed --device 0 &
pid2=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --action_space 6 --datadirectory "../../data/GOTO/left_right_5000_4_3_False_demos" --experiment_name "finetune_agent_head_left_right_5000_${seed}" --seed $seed --device 0 &
pid3=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/mp68ur0u/checkpoints/epoch=18-step=64125.ckpt   --action_space 7 --datadirectory "../../data/GOTO/all_diagonal_5000_4_3_False_demos" --experiment_name "finetune_agent_head_all_diagonal_5000_${seed}" --seed $seed --device 0  &
pid4=$!
sleep 5

wait $pid1 $pid2 $pid3 $pid4



seed=1

python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt   --action_space 0 --datadirectory "../../data/GOTO/standard_5000_4_3_False_demos" --experiment_name "finetune_agent_head_standard_5000_${seed}" --seed $seed --device 0  &
pid1=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt    --action_space 1 --datadirectory "../../data/GOTO/no_left_5000_4_3_False_demos" --experiment_name "finetune_agent_head_no_left_5000_${seed}" --seed $seed --device 0 &
pid2=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint  /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt  --action_space 2 --datadirectory "../../data/GOTO/no_right_5000_4_3_False_demos" --experiment_name "finetune_agent_head_no_right_5000_${seed}" --seed $seed --device 0 &
pid3=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt   --action_space 3 --datadirectory "../../data/GOTO/diagonal_5000_4_3_False_demos" --experiment_name "finetune_agent_head_diagonal_5000_${seed}" --seed $seed --device 0 &
pid4=$!
sleep 5

wait $pid1 $pid2 $pid3 $pid4

python train.py --config ./configs/agent_heads.yaml --checkpoint  /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt   --action_space 4 --datadirectory "../../data/GOTO/wsad_5000_4_3_False_demos" --experiment_name "finetune_agent_head_wsad_5000_${seed}" --seed $seed --device 0 &
pid1=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint  /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt   --action_space 5 --datadirectory "../../data/GOTO/dir8_5000_4_3_False_demos" --experiment_name "finetune_agent_head_dir8_5000_${seed}" --seed $seed --device 0 &
pid2=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt   --action_space 6 --datadirectory "../../data/GOTO/left_right_5000_4_3_False_demos" --experiment_name "finetune_agent_head_left_right_5000_${seed}" --seed $seed --device 0 &
pid3=$!
sleep 5
python train.py --config ./configs/agent_heads.yaml --checkpoint /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/imitation_learning_goal_obs/DiffusionMultiAgent/hy4vvmhw/checkpoints/epoch=18-step=64125.ckpt   --action_space 7 --datadirectory "../../data/GOTO/all_diagonal_5000_4_3_False_demos" --experiment_name "finetune_agent_head_all_diagonal_5000_${seed}" --seed $seed --device 0  &
pid4=$!
sleep 5

wait $pid1 $pid2 $pid3 $pid4
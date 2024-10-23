#! /bin/bash

set -e 

# Seed 1
python ./eval.py --config ./configs/standard/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name standard_go_to_large_mixed_agent_id_1   --device "cuda:0" 

python ./eval.py --config ./configs/no_left/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name  no_left_go_to_large_mixed_agent_id_1  --device "cuda:0"  

python ./eval.py --config ./configs/no_right/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name no_right_go_to_large_mixed_agent_id_1   --device "cuda:0" 

python ./eval.py --config ./configs/diagonal/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name diagonal_go_to_large_mixed_agent_id_1   --device "cuda:0" 

python ./eval.py --config ./configs/wsad/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name wsad_go_to_large_mixed_agent_id_1   --device "cuda:0" 

python ./eval.py --config ./configs/dir8/go_to_large.yaml --dm_model_path "mixed_action_space_goto_large_extended_3/DiffusionMultiAgent" --dm_model_name "aeon8dcu" --dm_model_checkpoint "epoch=454-step=100000.ckpt" --experiment_name dir8_go_to_large_mixed_agent_id_1 --device "cuda:0"

#python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_agentid_extended_1/DiffusionMultiAgent" --dm_model_name "bmt7vfe3" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name  left_right_distractor_mixed_agent_id_1  --device "cuda:0" 

#python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_1/DiffusionMultiAgent" --dm_model_name "bmt7vfe3" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name all_diagonal_distractor_mixed_agent_id_1  --device "cuda:0" 

# # Seed 2

# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name standard_distractor_mixed_agent_id_2  --device "cuda:0" 

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name  no_left_distractor_mixed_agent_id_2 --device "cuda:0"  

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name no_right_distractor_mixed_agent_id_2  --device "cuda:0" 

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name diagonal_distractor_mixed_agent_id_2  --device "cuda:0" 

# python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name wsad_distractor_mixed_agent_id_2  --device "cuda:0" 

# python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name dir8_distractor_mixed_agent_id_2 --device "cuda:0"

# python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name  left_right_distractor_mixed_agent_id_2 --device "cuda:0" 

# python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_2/DiffusionMultiAgent" --dm_model_name "tepfz3f4" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name all_diagonal_distractor_mixed_agent_id_2 --device "cuda:0" 

# # Seed 3

# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name standard_distractor_mixed_agent_id_3   --device "cuda:0" 

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name  no_left_distractor_mixed_agent_id_3  --device "cuda:0"  

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name no_right_distractor_mixed_agent_id_3   --device "cuda:0" 

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name diagonal_distractor_mixed_agent_id_3   --device "cuda:0" 

# python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name wsad_distractor_mixed_agent_id_3   --device "cuda:0" 

# python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name dir8_distractor_mixed_agent_id_3 --device "cuda:0"

# python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name  left_right_distractor_mixed_agent_id_3  --device "cuda:0" 

# python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_3/DiffusionMultiAgent" --dm_model_name "911pfpq7" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name all_diagonal_distractor_mixed_agent_id_3  --device "cuda:0" 

# # Seed 4

# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name standard_distractor_mixed_agent_id_4   --device "cuda:0" 

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name  no_left_distractor_mixed_agent_id_4  --device "cuda:0"  

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name no_right_distractor_mixed_agent_id_4   --device "cuda:0" 

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name diagonal_distractor_mixed_agent_id_4   --device "cuda:0" 

# python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name wsad_distractor_mixed_agent_id_4   --device "cuda:0" 

# python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name dir8_distractor_mixed_agent_id_4 --device "cuda:0"

# python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name  left_right_distractor_mixed_agent_id_4  --device "cuda:0" 

# python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "mixed_agentid_extended_4/DiffusionMultiAgent" --dm_model_name "cchk7jmj" --dm_model_checkpoint "epoch=157-step=500000.ckpt"  --experiment_name all_diagonal_distractor_mixed_agent_id_4  --device "cuda:0" 
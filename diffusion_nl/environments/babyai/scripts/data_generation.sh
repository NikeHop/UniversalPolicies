#! /bin/bash

set -e 

python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 0
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 1
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 2
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 3
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 4
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 5
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 6
python generate_demos.py --config ./configs/goto_distractor.yaml  --action_space 7

python merge.py --data_dir "../../../data/GOTO_DISTRACTOR/" --n_episodes 5000 --minimum_length 4 --num_distractors 3 
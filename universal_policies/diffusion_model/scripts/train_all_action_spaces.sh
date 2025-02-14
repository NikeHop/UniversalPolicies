#! /bin/bash 

set -e 

python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/standard_83_4_0_False_demos/dataset_83.pkl --action_space 0
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/no_left_83_4_0_False_demos/dataset_83.pkl --action_space 1
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/no_right_83_4_0_False_demos/dataset_83.pkl --action_space 2
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/diagonal_83_4_0_False_demos/dataset_83.pkl --action_space 3
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/wsad_83_4_0_False_demos/dataset_83.pkl --action_space 4
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/dir8_83_4_0_False_demos/dataset_83.pkl --action_space 5
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/left_right_83_4_0_False_demos/dataset_83.pkl --action_space 6
python train.py --config ./configs/goto.yaml --datapath ../../data/GOTO_DISTRACTORS/all_diagonal_83_4_0_False_demos/dataset_83.pkl --action_space 7
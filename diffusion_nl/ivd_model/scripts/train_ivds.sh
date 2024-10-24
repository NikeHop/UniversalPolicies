#! /bin/bash 

set -e

python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/standard_5000_4_3_True_demos/dataset_5000.pkl --action_space 0
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/no_left_5000_4_3_True_demos/dataset_5000.pkl --action_space 1
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/no_right_5000_4_3_True_demos/dataset_5000.pkl --action_space 2
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/diagonal_5000_4_3_True_demos/dataset_5000.pkl --action_space 3
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/wsad_5000_4_3_True_demos/dataset_5000.pkl --action_space 4
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/dir8_5000_4_3_True_demos/dataset_5000.pkl --action_space 5
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/left_right_5000_4_3_True_demos/dataset_5000.pkl --action_space 6
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO_DISTRACTORS/all_diagonal_5000_4_3_True_demos/dataset_5000.pkl --action_space 7
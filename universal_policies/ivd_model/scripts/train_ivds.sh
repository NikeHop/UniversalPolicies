#! /bin/bash 

set -e

python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/standard_83_4_0_False_demos/dataset_83.pkl --action_space 0
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/no_left_83_4_0_False_demos/dataset_83.pkl --action_space 1
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/no_right_83_4_0_False_demos/dataset_83.pkl --action_space 2
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/diagonal_83_4_0_False_demos/dataset_83.pkl --action_space 3
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/wsad_83_4_0_False_demos/dataset_83.pkl --action_space 4
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/dir8_83_4_0_False_demos/dataset_83.pkl --action_space 5
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/left_right_83_4_0_False_demos/dataset_83.pkl --action_space 6
python train.py --config ./configs/ivd.yaml --datapath ../../data/GOTO/all_diagonal_83_4_0_False_demos/dataset_83.pkl --action_space 7
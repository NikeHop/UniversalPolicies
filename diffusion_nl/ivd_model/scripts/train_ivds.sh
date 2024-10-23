#! /bin/bash 

python train.py --config ./configs/ivd_boss.yaml --datapath ../../data/GOTO_LARGE/standard_83000_4_7_False_demos --action_space 0
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/no_left_83000_4_7_False_demos --action_space 1
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/no_right_83000_4_7_False_demos --action_space 2
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/diagonal_83000_4_7_False_demos --action_space 3
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/wsad_83000_4_7_False_demos --action_space 4
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/dir8_83000_4_7_False_demos --action_space 5
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/left_right_83000_4_7_False_demos --action_space 6
# python train.py --config ./configs/ivd_babyai_goto_large.yaml --datapath ../../data/GOTO_LARGE/all_diagonal_83000_4_7_False_demos --action_space 7
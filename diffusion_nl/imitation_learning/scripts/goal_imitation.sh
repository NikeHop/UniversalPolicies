#! /bin/bash 

python train.py --config ./configs/goal_imitation.yaml --max_gap 1
python train.py --config ./configs/goal_imitation.yaml --max_gap 2
python train.py --config ./configs/goal_imitation.yaml --max_gap 3
python train.py --config ./configs/goal_imitation.yaml --max_gap 4
python train.py --config ./configs/goal_imitation.yaml --max_gap 5
python train.py --config ./configs/goal_imitation.yaml --max_gap 100
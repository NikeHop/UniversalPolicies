#! /bin/bash 


source /projects/0/prjs1044/miniconda3/bin/activate
conda activate diffusion_nl 


python train.py --config ./configs/complete_action_space.yaml --device 0 --seed 0 &
pid1=$!
sleep 3
python train.py --config ./configs/complete_action_space.yaml --device 1 --seed 1 &
pid2=$!
sleep 3
python train.py --config ./configs/complete_action_space.yaml --device 2 --seed 2 &
pid3=$!
sleep 3
python train.py --config ./configs/complete_action_space.yaml --device 3 --seed 3 &
pid4=$!
sleep 3

wait $pid1 $pid2 $pid3 $pid4
#! /bin/bash 


source /projects/0/prjs1044/miniconda3/bin/activate
conda activate diffusion_nl 


python train.py --config ./configs/agent_heads.yaml --experiment_name "agent_heads_1" --device 0 --seed 0 &
pid1=$!
sleep 2
python train.py --config ./configs/agent_heads.yaml --experiment_name "agent_heads_2"  --device 1 --seed 1 &
pid2=$!
sleep 2
python train.py --config ./configs/agent_heads.yaml --experiment_name "agent_heads_3" --device 2 --seed 2 &
pid3=$!
sleep 2
python train.py --config ./configs/agent_heads.yaml --experiment_name "agent_heads_4"  --device 3 --seed 3 &
pid4=$!
sleep 2

wait $pid1 $pid2 $pid3 $pid4
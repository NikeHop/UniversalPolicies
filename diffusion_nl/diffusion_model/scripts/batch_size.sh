#! /bin/bash 

python train.py --config ./configs/state/standard/basic.yaml  --experiment_name "125_256" --data_path ../../data/baby_ai/standard_125_4_0_demos/dataset_125.pkl --device 0   & 
pid1=$!
python train.py --config ./configs/state/standard/basic.yaml --experiment_name "125_256" --data_path ../../data/baby_ai/standard_125_4_0_demos/dataset_125.pkl --lr 0.00005  --device 1   &
pid2=$!
python train.py --config ./configs/state/standard/basic.yaml --experiment_name "500_256" --device 2   &
pid3=$!
python train.py --config ./configs/state/standard/basic.yaml --experiment_name "500_256" --lr 0.00005 --device 3   &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4
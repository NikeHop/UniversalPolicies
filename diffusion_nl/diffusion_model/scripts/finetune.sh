#! /bin/bash 

set -e 

# python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/no_left_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_no_left --device 0 &
# pid1=$!

# python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/no_right_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_no_right --device 1 & 
# pid2=$!

# python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/diagonal_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_diagonal --device 2 &
# pid3=$!

# python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/standard_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_standard --device 3 &
# pid4=$!

# wait $pid1 $pid2 $pid3 $pid4

python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/wsad_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_wsad --device 0 &
pid1=$!

python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/dir8_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_dir8 --device 1 & 
pid2=$!

python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/left_right_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_left_right --device 2 &
pid3=$!

python train.py --config ./configs/state/distractors_3.yaml --data_path ../../data/GOTO/all_diagonal_5000_4_3_demos/dataset_5000.pkl --experiment_name finetune_all_diagonal --device 3 &
pid4=$!

wait $pid1 $pid2 $pid3 $pid4
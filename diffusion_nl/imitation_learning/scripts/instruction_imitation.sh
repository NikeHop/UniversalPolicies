#! /bin/bash 

set -e 

for seed in 1 2 3 4 5
do

python train.py --config ./configs/instruction_imitation.yaml --action_space 0 --datadirectory "../../data/GOTO/standard_30000_4_3_False_demos" --experiment_name "standard_30000_${seed}" --seed $seed --device 0  


python train.py --config ./configs/instruction_imitation.yaml --action_space 1 --datadirectory "../../data/GOTO/no_left_30000_4_3_False_demos" --experiment_name "no_left_30000_${seed}" --seed $seed --device 0 


python train.py --config ./configs/instruction_imitation.yaml --action_space 2 --datadirectory "../../data/GOTO/no_right_30000_4_3_False_demos" --experiment_name "no_right_30000_${seed}" --seed $seed --device 0 


python train.py --config ./configs/instruction_imitation.yaml --action_space 3 --datadirectory "../../data/GOTO/diagonal_30000_4_3_False_demos" --experiment_name "diagonal_30000_${seed}" --seed $seed --device 0 

done

for seed in 1 2 3 4 5 
do 

python train.py --config ./configs/instruction_imitation.yaml --action_space 4 --datadirectory "../../data/GOTO/wsad_30000_4_3_False_demos" --experiment_name "wsad_30000_${seed}" --seed $seed --device 0 


python train.py --config ./configs/instruction_imitation.yaml --action_space 5 --datadirectory "../../data/GOTO/dir8_30000_4_3_False_demos" --experiment_name "dir8_30000_${seed}" --seed $seed --device 0 


python train.py --config ./configs/instruction_imitation.yaml --action_space 6 --datadirectory "../../data/GOTO/left_right_30000_4_3_False_demos" --experiment_name "left_right_30000_${seed}" --seed $seed --device 0 


python train.py --config ./configs/instruction_imitation.yaml --action_space 7 --datadirectory "../../data/GOTO/all_diagonal_30000_4_3_False_demos" --experiment_name "all_diagonal_30000_${seed}" --seed $seed --device 0 


done 


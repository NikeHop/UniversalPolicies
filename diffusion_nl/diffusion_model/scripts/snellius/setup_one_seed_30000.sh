#! /bin/bash 

seed=4

sbatch ./scripts/snellius/snellius_multi_seed_distractors_small_1_4.sh $seed 3 ../../data/GOTO/diagonal_30000_4_3_False_demos/dataset_30000.pkl "diagonal_30000_edm_${seed}"
sbatch ./scripts/snellius/snellius_multi_seed_distractors_small_1_4.sh $seed 7 ../../data/GOTO/all_diagonal_30000_4_3_False_demos/dataset_30000.pkl "all_diagonal_30000_edm_${seed}"
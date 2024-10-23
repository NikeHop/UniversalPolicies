#! /bin/bash 

seed=1

#sbatch ./scripts/snellius/snellius_mixed.sh $seed
#sbatch ./scripts/snellius/snellius_mixed_example_1.sh $seed
#sbatch ./scripts/snellius/snellius_mixed_example_1000.sh $seed
#sbatch ./scripts/snellius/snellius_mixed_action_space.sh $seed
# sbatch ./scripts/snellius/snellius_mixed_agentid.sh 2
# sbatch ./scripts/snellius/snellius_mixed_agentid.sh 3
# sbatch ./scripts/snellius/snellius_mixed_agentid.sh 4
# sbatch ./scripts/snellius/snellius_mixed.sh 2
# sbatch ./scripts/snellius/snellius_mixed.sh 3
# sbatch ./scripts/snellius/snellius_mixed.sh 4

# sbatch ./scripts/snellius/snellius_mixed_action_space.sh 1
# sbatch ./scripts/snellius/snellius_mixed_action_space.sh 2
# sbatch ./scripts/snellius/snellius_mixed_action_space.sh 3
# sbatch ./scripts/snellius/snellius_mixed_action_space.sh 4

sbatch ./scripts/snellius/snellius_mixed_agent_type.sh 1
sbatch ./scripts/snellius/snellius_mixed_agent_type.sh 2
sbatch ./scripts/snellius/snellius_mixed_agent_type.sh 3
sbatch ./scripts/snellius/snellius_mixed_agent_type.sh 4

# sbatch ./scripts/snellius/snellius_mixed_example_1.sh 2
# sbatch ./scripts/snellius/snellius_mixed_example_1.sh 3
# sbatch ./scripts/snellius/snellius_mixed_example_1.sh 4
# Get the results of imitation learning 
import numpy as np 
import wandb

api = wandb.Api()

entity = "niklas_hop"
project = "DiffusionMultiAgent"

runs = api.runs(f"{entity}/{project}")


action_spaces = [0] # ["standard"] # ["standard","no_left","no_right","diagonal","wsad","dir8","left_right","all_diagonal"]
seeds = [1,2,3,4,5]

for action_space in action_spaces:
    runs_of_condition = [run for run in runs if f"evaluation_diffusion_agentmixed_go_to_large_standard_" in run.name]
    print([run.name for run in runs_of_condition])
    assert len(runs_of_condition) == 4
    average_rewards = []
    completion_rates = []
    for run in runs_of_condition:
        summary = run.summary
        average_rewards.append(summary["average_reward"])
        completion_rates.append(summary["completion_rate"])

    average_rewards = np.array(average_rewards)
    completion_rates = np.array(completion_rates)

    print(f"Action space: {action_space}")
    print(f"Average reward: {average_rewards.mean()} {average_rewards.std()}")
    print(f"Completion rate: {completion_rates.mean()} {completion_rates.std()}")
    
    

    
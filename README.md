# Making Universal Policies Universal

![method overview](./assets/overview_method.png)

## Prerequisites & Dependencies


All of the logging is done via [WandB](https://wandb.ai/site/), but needs to be enabled in the config files.

## Run Experiments 

### Generate Demonstrations 

The code to generate demonstrations can be found in the `./diffusion_nl/environments/babyai` folder all following commands should be run from there. To generate the demonstration datasets for the different environments and different actions spaces, run:

`python generate_demos.py --config ./configs/CONFIG_FILE --action_space 0`.

Here `CONFIG_FILE` should be one of:
- `goto.yaml`: agent needs to go to the object; see [here](https://minigrid.farama.org/environments/babyai/GoToObj/)
- `goto_distractor.yaml`: agent needs to go to an object with distractors present; see [here](https://minigrid.farama.org/environments/babyai/GoToLocal/)
- `goto_distractor_large.yaml`, agent needs to go to an object with distractors present navigating through nine rooms; see [here](https://minigrid.farama.org/environments/babyai/GoTo/)

The available action spaces are:

- 0: standard 
- 1: no left-turns
- 2: no right-turns
- 3: diagonal, additional to the standard actions move forward diagonally
- 4: wsad, move to the left, right, up, down and if pointing to another direction turn at the same time
- 5: dir8, move to any diagonal fields and turn right
- 6: left-right, move left and right and turn right
- 7: all-diagonal, all diagonal cells + turn right

If the agent should be coloured differntly depending on the action space used, set the `use_agent_type` argument in the config-file to True. 

The resulting pickle file contains a list of tuples and will be stored in a folder with the following naming convention: `{action_space}_{n_episodes}_{min_length}_{num_distractors}_{use_agent_type}_demos`. Each tuple corresponds to an episode. The tuple contains the following values:

- instruction `str`
- environment name: `str`
- obs_sequence: `blosc.array`
- directions: `list[int]`
- actions: `list[int]`
- rewards: `list[floats]`
- action_space: `int`

To generate the dataset for all action spaces and pool all datasets from the in-distribution agents (0-5) to create the mixture dataset run:

`bash ./scripts/data_generation.sh`

### Train Diffusion Planner

### Train Inverse Dynamics Models 

The code for the inverse dynamics models can be found in `./diffusion_nl/ivd_model`. All the following commands should be run from there. To train an inverse dynamics model for a given action space on a specific dataset run:

`python train.py ./configs/ivd.yaml --datapath ../../data/GOTO/standard_83_4_0_False_demos/dataset_83.pkl --action_space 0`

To train an ivd for all available action spaces in an instance of the BabyAI environment, add the corresponding datapaths to the script `./scripts/train_ivds.sh` and run:

`bash ./scripts/train_ivds.sh`

### Evaluate via the Diffusion Agent

### Train Imitation Learning Policy

The code for the inverse dynamics models can be found in `./diffusion_nl/imitation_learning`. All the following commands should be run from there. To train the imitation learning baselines on a specific dataset for a specific action space run 

`python train.py --config ./configs/instruction_imitation_goto.yaml --datapath ./data/GOTO/`


After training the model is evaluate over 512 evaluation episodes. 

## Trained Models 

We make the trained inverse dynamics models, imitation learning baselines and diffusion planners for a single random seed available here.



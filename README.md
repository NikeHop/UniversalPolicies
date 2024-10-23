# Making Universal Policies Universal

![method overview](./assets/overview_method.png)

## Prerequisites & Dependencies

## Run Experiments 

### Generate Demonstrations 

To generate the demonstration datasets for the different environments and different actions spaces, run:

`python generate_demos.py --config CONFIG_FILE`

The 

The resulting pickle file contains a list of tuples. Each tuple corresponds to an episode. The tuple contains the following values:

- instruction `str`
- environment name: `str`
- obs_sequence: `blosc.array`
- directions: `list[int]`
- actions: `list[int]`
- rewards: `list[floats]`
- action_space: `int`

### Train Diffusion Planner

### Train Inverse Dynamics Models 

To train an inverse dynamics model for a given action space in a specific environment modify the action space and datapath argument in `./configs/ivd.yaml` and run 

`python train.py ./configs/ivd.yaml`

To train an ivd for all available action spaces in an instance of the BabyAI environment, add the corresponding datapaths to the script `./scripts/train_ivds.sh`.

`bash ./scripts/train_ivds.sh`

### Evaluate via Diffusion Agent

### Train Imitation Learning Policy

## Pretrained Models 

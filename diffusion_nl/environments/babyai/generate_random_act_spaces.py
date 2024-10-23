import argparse
from collections import defaultdict
import logging
import multiprocessing as mp
import os
import pickle
import random
import time
import sys

print(sys.path)

from minigrid.core.actions import ActionSpace
from generate import generate_demos
import yaml

from diffusion_nl.utils.environments import (
    seeding,
    ENVS,
    TEST_ENV,
    CLASS2ENV,
    FIXINSTGOTO_ENVS,
)
from goto_specific import register_envs

def generate_random_act_space_trajs(config):
    if config["env"] == "GOTO":
        envs = FIXINSTGOTO_ENVS
    elif config["env"] == "BOSSLEVEL":
        envs = ["BabyAI-BossLevel-v0"]
    else:
        raise NotImplementedError(f"This environment has not been implemented {config['env']}")
    seeding(config["seed"])
    act_spaces = ActionSpace.get_all_action_spaces_with_n_sample_actions(n_sample_actions=config["n_sample_actions"])
    random.shuffle(act_spaces)

    successful_act_spaces = 0
    current_act_space_i = -1
    while current_act_space_i + 1 < len(act_spaces) and successful_act_spaces < config["n_action_spaces"]:
        current_act_space_i += 1
        print(f"Processing act space nr. {current_act_space_i}")
        act_space = act_spaces[current_act_space_i]
        if act_space.is_named_action_space():
            continue
        all_demos = []
        try:
            for e in envs:
                demos = generate_demos(
                    n_episodes=config["n_episodes"],
                    num_workers=config["num_workers"],
                    data_dir=config["data_directory"],
                    env=config["env"],
                    env_name=e,
                    action_space=act_space,
                    minimum_length=config["minimum_length"],
                    max_steps=config["max_steps"],
                    seed=config["seed"],
                    img_size=config["img_size"],
                    use_img_obs=config["use_img_obs"],
                    num_distractors=config["num_distractors"],
                    debug=False,
                    resample_on_fail=False,
                    save_data=False
                )

                all_demos += demos
        except ValueError as e:
            continue
        successful_act_spaces += 1

        dataset_name = f"{config['n_episodes']}_{config['minimum_length']}_{config['num_distractors']}_{config['n_sample_actions']}actions_demos"
        dir_path = os.path.join(
            config["data_directory"],
            config["env"],
            dataset_name,
        )
        filepath = f"{dir_path}/{successful_act_spaces}.pkl"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(filepath, "wb") as file:
            pickle.dump(all_demos, file)
    print("Processed all action spaces")

if __name__ == "__main__":
    register_envs()  # registers fixed instruction envs with gym

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, help="Path to the config file", required=True
    )

    parser.add_argument(
        "--test", action="store_true", help="Generates one episode for each environment"
    )

    parser.add_argument("--img_obs", action="store_true", help="Use image observations")
    parser.add_argument(
        "--n_episodes", type=int, help="How many episodes per environment to create"
    )
    parser.add_argument(
        "--n_dists",
        type=int,
        default=None,
        help="Number of distractors in the environment",
    )

    parser.add_argument(
        "--n_sample_actions", type=int,
        help="How many episodes actions should be in each action space (+ 4 fixed actions)"
    )
    parser.add_argument(
        "--n_action_spaces", type=int,
        help="How many actions spaces should be sampled"
    )
    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.ERROR)

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if args.n_episodes != None:
        config["n_episodes"] = args.n_episodes

    if args.n_dists != None:
        config["num_distractors"] = args.n_dists
    if args.n_sample_actions != None:
        config["n_sample_actions"] = args.n_sample_actions
    if args.n_action_spaces != None:
        config["n_action_spaces"] = args.n_action_spaces

    # Pick an environment
    if config["env"]=="GOTO":
        envs = FIXINSTGOTO_ENVS
    elif config["env"]=="BOSSLEVEL":
        envs = ["BabyAI-BossLevel-v0"]
    else:
        raise NotImplementedError(f"This environment has not been implemented {config['env']}")
    generate_random_act_space_trajs(config=config)


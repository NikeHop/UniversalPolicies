from collections import defaultdict


import gymnasium as gym
import numpy as np
import random
import torch
from minigrid.core.constants import COLOR_ENV_NAMES


# Seed everything
def seeding(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# All environment names BabyAI
LEVELS2NAME = [
    "GoToObj",
    "GoToRedBallGrey",
    "GoToRedBall",
    "GoToLocal",
    "PutNextLocal",
    "PickupLoc",
    "GoToObjMaze",
    "GoTo",
    "Pickup",
    "UnblockPickup",
    "Open",
    "Unlock",
    "PutNext",
    "Synth",
    "SynthLoc",
    "GoToSeq",
    "GoToImpUnlock",
    "BossLevel",
]

LEVELS2NAME = {i:name for i,name in enumerate(LEVELS2NAME)}
NAME2LEVELS = {name:i for i,name in LEVELS2NAME.items()}


ENVS = [
    env_name
    for env_name in list(gym.envs.registry.keys())
    if "BabyAI" in env_name and "v0" in env_name
]

LEVEL2ENV = defaultdict(list)
for env in ENVS:
    env_level = env.split("-")[1]
    if env_level in NAME2LEVELS:
        level = NAME2LEVELS[env_level]
        LEVEL2ENV[level].append(env)

LEVEL_CLASSES = [(0,5),(6,11),(12,18)]

CLASS2ENV = defaultdict(list)

for k,cl in enumerate(LEVEL_CLASSES):
    for i in range(cl[0],cl[1]+1):
        if i in LEVEL2ENV:
            CLASS2ENV[k].append(LEVEL2ENV[i][0])

print(LEVEL2ENV)

"""
"GoToObj"
"PutNextLocal",
"PickupLoc",
"GoToObjMaze",
"GoTo",
"Pickup",
"UnblockPickup",
"Open",
"Unlock",
"Synth",
"SynthLoc",
"GoToSeq",

"""

TEST_ENV = ["BabyAI-PutNextLocal-v0"]

FIXINSTGOTO_ENVS = []
for color in COLOR_ENV_NAMES:
    print(f"color: {color}")
    for obj in ['ball', 'box', 'key']:
        FIXINSTGOTO_ENVS.append(
            f'BabyAI-FixInstGoTo{color.capitalize()}{obj.capitalize()}-v0'
        )

print(f"FIXINSTGOTO_ENVS: {FIXINSTGOTO_ENVS}")

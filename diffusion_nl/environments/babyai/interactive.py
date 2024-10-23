import argparse
import os

import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper
from minigrid.core.actions import Actions, NamedActionSpace, ActionSpace
import torch
from torchvision.io import write_video

from diffusion_nl.utils.environments import seeding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_mode",
        choices=["human", "rgb_array"],
        help="Rendering mode of the environment",
    )
    parser.add_argument(
        "--action_space",
        default=0,
        type=int,
        help="The type of action space"
    )

    args = parser.parse_args()

    print("The game begins")
    seeding(42)
    print(NamedActionSpace(args.action_space))
    env = gym.make("BabyAI-GoToObj-v0", action_space=NamedActionSpace(args.action_space), render_mode=args.render_mode, action_space_agent_color=True)
    env = RGBImgObsWrapper(env, tile_size=224 // env.width)

    obs = env.reset()
    env.render()
    done = False
    images = []

    while not done:
        images.append(obs[0]["image"])
        action = int(input(f"Choose an action between 0-{len(Actions)}:"))
        if action == len(Actions):
            break

        obs = env.step(action)
        env.render()
        

    filename = os.path.join("./", f"test.mp4")
    video = torch.tensor(images)
    with open(filename, "wb+") as file:
        write_video(file, video, fps=1)

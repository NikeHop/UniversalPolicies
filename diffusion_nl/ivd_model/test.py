import argparse
import yaml 

import matplotlib.pyplot as plt

from diffusion_nl.ivd_model.model import IVDBabyAI
from diffusion_nl.ivd_model.data import get_data_calvin

def example_predictions(config):
    # Load the model
    model = IVDBabyAI.load_from_checkpoint(config["checkpoint"],device=config["device"])

    # Load the data
    train_dataloader, validation_dataloader = get_data_calvin(config["data"])

    for batch in train_dataloader:
        obs1, obs2, gt_actions = batch
        print(obs1.shape)
        print(obs2.shape)
        plt.imshow(obs1)
        plt.savefig("obs1.png")
        plt.clf()
        plt.imshow(obs2)
        plt.savefig("obs2.png")
        pred_actions = model.predict(obs1.to(config["device"]),obs2.to(config["device"]))
        print(gt_actions)
        print(pred_actions)
        break

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.config,"rb") as file:
        config = yaml.safe_load(file)

    example_predictions(config)


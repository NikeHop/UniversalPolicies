"""
Merge samples from individual files
"""

import os

import glob
import pickle


def merge(config):

    all_files = []
    for environment in config["environments"]:
        files = glob.glob(
            os.path.join(config["directory"], environment, f"dataset*.pkl")
        )
        all_files += files

    print(all_files)
    samples = []
    for file in all_files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            samples += data

    if not os.path.exists(config["goal_directory"]):
        os.makedirs(config["goal_directory"])

    filepath = os.path.join(config["goal_directory"], config["dataset_name"])
    with open(filepath, "wb+") as file:
        pickle.dump(samples, file)

def subsample():
    filepath = "../../../data/BOSSLEVEL/standard_1000000_4_0_False_demos/dataset_1000000.pkl"

    with open(filepath, "rb") as f:
        data = pickle.load(f)
        data = data[:50000]

    if not os.path.exists("../../../data/BOSSLEVEL/standard_50000_4_0_False_demos"):
        os.makedirs("../../../data/BOSSLEVEL/standard_50000_4_0_False_demos")

    goal_filepath = "../../../data/BOSSLEVEL/standard_50000_4_0_False_demos/dataset_50000.pkl"
    with open(goal_filepath, "wb+") as file:
        pickle.dump(data, file)


    
if __name__ == "__main__":

    config = {
        "environments": [
            "standard_500000_4_0_False_demos",
            "standard_500000_4_7_False_demos_part_2",
        ],
        "directory": "../../../data/BOSSLEVEL",
        "goal_directory": "../../../data/BOSSLEVEL/standard_1000000_4_0_False_demos",
        "dataset_name": "dataset_1000000.pkl",
    }

    """
    [
            "standard_83000_4_7_False_demos",
            "no_left_83000_4_7_False_demos",
            "no_right_83000_4_7_False_demos",
            "diagonal_83000_4_7_False_demos",
            "wsad_83000_4_7_False_demos",
            "dir8_83000_4_7_False_demos",
    ]
    """
    
    #merge(config)
    subsample()
"""
Remove any unnecessary data from the files to improve data loading speed
"""
import argparse 
import glob 
import numpy as np 
import os 
import yaml
import tqdm 

def transform_data(config):
    for split in ["training","validation"]:
        directory = os.path.join(config["directory"],split)
        print(directory+"/*.npz")
        files = glob.glob(directory+"/*.npz")
        for file in tqdm.tqdm(files):
            filename = file.split("/")[-1]        
            data = np.load(file,allow_pickle=True).items()
            new_data = {}
            for key, value in data:
                if key in ["rgb_static","rel_actions"]:
                    new_data[key] = value
            
            new_filename = os.path.join(config["directory"],split,filename)
            np.savez(new_filename,**new_data)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config,"r") as file:
        config = yaml.safe_load(file)
        transform_data(config)





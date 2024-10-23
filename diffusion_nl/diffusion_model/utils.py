import glob
import os 
import pickle 

import blosc
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np 
import torch 

from minigrid.core.grid import Grid
from minigrid.core.constants import COLORS, IDX_TO_COLOR
def extract_agent_pos_and_direction(obs):
    agent_pos = None
    agent_dir = None
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            if obs[i][j][0] == 10:
                print(obs[i][j])
                agent_pos = (i,j)
                agent_dir = obs[i][j][2]
                break
    return agent_pos, agent_dir

def merge_datasets(directory:str, dataset_name:str):
    files = glob.glob(os.path.join(directory,f"*.pkl"))
    total_data = []
    for file in files:
        with open(file,"rb") as f:
            data = pickle.load(f)[0]
            total_data += data
    print(total_data)
    with open(os.path.join(directory,dataset_name),"wb") as file:
        pickle.dump(total_data,file)

def load_dataset(directory:str, dataset_name:str):
    with open(os.path.join(directory,dataset_name),"rb") as file:
        data = pickle.load(file)
    return data

def state2img(state:np.array):
    agent_pos, agent_dir = extract_agent_pos_and_direction(state)
    grid, _ = Grid.decode(state)
    if agent_pos is not None:
        agent_type = state[agent_pos[0]][agent_pos[1]][1]
    else:
        agent_type = 0
    agent_color = COLORS[IDX_TO_COLOR[agent_type]]
    img = grid.render(32,agent_pos,agent_dir,agent_color=agent_color)
    return img 

def transform_sample(sample:torch.Tensor):

    sample = torch.round(sample)

    sample[:,:,:,0] = torch.clamp(sample[:,:,:,0],0,10)
    sample[:,:,:,1] = torch.clamp(sample[:,:,:,1],0,5)
    sample[:,:,:,2] = torch.clamp(sample[:,:,:,2],0,4)

    sample = sample.cpu().numpy()

    images = []
    for state in sample:
        img = state2img(state)
        images.append(img)

    return images 

def transform_video(video):
    """
    Takes in a video of states and returns a video of RGB observation
    """
    images = []
    for state in video:
        print("calling state2img")
        img = state2img(state)
        images.append(img)
    print(images[0].shape)
    video = torch.stack([torch.tensor(image).long() for image in images],dim=0)

    return video 

def visualize_state(data):
    for sample in data:
        traj = blosc.unpack_array(sample[2])
        state = traj[0]
        agent_pos, agent_dir = extract_agent_pos_and_direction(state)
        print(agent_pos,agent_dir)
        grid, _  = Grid.decode(state)
        img = grid.render(32,agent_pos,agent_dir)
        plt.imshow(img)
        plt.axis("off")
        plt.savefig("./sample.png")
        break

if __name__=="__main__":
    #merge_datasets("../../data/baby_ai/sample_efficiency_data","babyai_goto_obj_10000.pkl")
    data = load_dataset("../../data/baby_ai/sample_efficiency_data","babyai_goto_obj_10000.pkl")
    
    visualize_state(data)

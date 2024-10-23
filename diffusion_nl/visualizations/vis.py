### Code for visualizations 

import pickle 

import blosc
import matplotlib.pyplot as plt

from diffusion_nl.diffusion_model.utils import transform_video

def environment_examples():
    env_type="GOTO"
    datapath = f"../../data/{env_type}/standard_5000_4_3_False_demos/dataset_5000.pkl"
    
    with open(datapath,"rb") as f:
        data = pickle.load(f)
    
    sample= data[0]
    trajectory = sample[2]

    trajectory = blosc.unpack_array(trajectory)
    video = transform_video(trajectory)
    print(trajectory.shape)
    start_obs = video[0]

    plt.imshow(start_obs)
    plt.axis("off")
    plt.savefig(f"example_distractor_{env_type}.png")
    plt.savefig(f"example_distractor_{env_type}.pdf")




if __name__=="__main__":
    environment_examples()

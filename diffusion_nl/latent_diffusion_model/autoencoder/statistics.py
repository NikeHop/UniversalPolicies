"""
Methods to collect statistics about the different datasets used for training
"""

import matplotlib.pyplot as plt
import numpy as np 
import torch 

from torchvision.transforms import Resize, InterpolationMode

def effect_of_downsizing(config):
    if config["dataset"]=="calvin":
        example = get_example_image(config["dataset"]) 
        downsized_example = downsize_image(example,config["algorithm"])
        plt.clf()
        plt.imshow(example.numpy())
        plt.savefig("original.png")
        plt.clf()
        plt.imshow(downsized_example.numpy())
        plt.savefig("downsized.png")    

    else:
        raise NotImplementedError("Unknown dataset")
    

def downsize_image(example,algorithm):
    if algorithm=="bilinear":
        interpolation_mode = InterpolationMode.BILINEAR
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")
    resize_transform = Resize(size=(128,128),interpolation=interpolation_mode)
    print(example.shape)
    example = resize_transform(example.permute(2,0,1)).permute(1,2,0)
    print(example.shape)

    return example

def get_example_image(dataset):
    if dataset=="calvin":
        return get_example_image_calvin()
    else:
        raise NotImplementedError("Unknown dataset")

def get_example_image_calvin():
    filepath = "../../../data/calvin/calvin_debug_dataset/validation/episode_0553567.npz"
    calvin_data = np.load(filepath,allow_pickle=True)
    for key,value in calvin_data.items():
        if key=="rgb_static":
            return torch.from_numpy(value)
        
    
if  __name__=="__main__":

    config = {"dataset":"calvin","algorithm":"bilinear"}
    effect_of_downsizing(config)



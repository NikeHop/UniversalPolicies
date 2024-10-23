import math
import os

import matplotlib.pyplot as plt 
import numpy as np 
import torch 
from torch import Tensor
from torchvision.io import write_video


def to_tensor(samples: list, long=True):
    """
    samples: list of lists of tensors
    """
    if isinstance(samples[0],list):
        cpu_samples = []
        for sample in samples:
            cpu_sample = []
            for elem in sample:
                elem = elem.cpu()
                if long:
                    cpu_sample.append(elem.long())
                else:
                    cpu_sample.append(elem.float())
            cpu_samples.append(cpu_sample)

    elif isinstance(samples[0],Tensor):
        cpu_samples = []
        for elem in samples:
            elem = elem.cpu()
            if long:
                cpu_samples.append(elem.long())
            else:
                cpu_samples.append(elem.float())
    else:
        raise NotImplemented("Samples must either be a list of tensors or a list of list of tensors")

    return cpu_samples

def normalize(sample:Tensor,mean:Tensor,var:Tensor,reverse=False):
    if not reverse:
        return (sample-mean)/var
    else:
        return sample*var + mean

def normalize_samples(samples:list,mean:Tensor,var:Tensor, reverse=False):

    if isinstance(samples[0],list):
        cpu_samples = []
        for sample in samples:
            cpu_sample = []
            for elem in sample:
                elem = normalize(elem,mean,var,reverse)
                cpu_sample.append(elem.long())
            cpu_samples.append(cpu_sample)

    elif isinstance(samples[0],Tensor):
        cpu_samples = []
        for elem in samples:
            elem = normalize(elem,mean,var,reverse)
            cpu_samples.append(elem.long())
    else:
        raise NotImplemented("Samples must either be a list of tensors or a list of list of tensors")

    return cpu_samples

def visualize_action_annotations(traj: torch.Tensor, actions: torch.Tensor, count:int, path:str):
    filename = f"action_annotated_traj_{count}"
    filepath = os.path.join(path,filename)
    T = traj.shape[0]
    fig, axs = plt.subplots(nrows=1,ncols=T)
    for i in range(T):
            img = traj.long()[i].permute(1,2,0).cpu()
            axs[i].imshow(img)
            axs[i].set_axis_off()

            if i<T-1:
                axs[i].set_title(f"{actions[i].cpu().item()}")

    plt.savefig(filepath)

def visualize_samples(samples,path,d2=False):
    if d2:
        visualize_samples_d2(samples,path)
    else:
        visualize_samples_d3(samples,path)

def visualize_samples_d3(samples,path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i,sample in enumerate(samples):
        video = sample[-1][0]
        filename = os.path.join(path,f"trajectory_{i}.png")
        T = video.shape[1]
        fig, axs = plt.subplots(nrows=1,ncols=T)
        for i in range(T):
            img = video.permute(1,0,2,3).long()[i].permute(1,2,0).cpu()
            axs[i].imshow(img)
            axs[i].set_axis_off()

        plt.savefig(filename)

def visualize_samples_d2(samples,path):
    if not os.path.exists(path):
        os.makedirs(path)

    n = len(samples)
    n_rows = math.sqrt(n)
    if n_rows != int(n_rows):
        n_rows = int(n_rows) + 1
    else:
        n_rows = int(n_rows)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows)

    for k, sample in enumerate(samples):
        row = k // n_rows
        col = k % n_rows
   
        image = sample[-1][0].permute(1, 2, 0)
        axs[row, col].imshow(image)
        axs[row, col].set_axis_off()

    plt.tight_layout()
    plt.savefig(f"{path}/samples.png")


def visualize_backward_diffusion(samples, freq=1 ,name=None, path="./", d2=False):
    if d2:
        visualize_backward_diffusion_d2(samples,freq,name,path)
    else:
        visualize_backward_diffusion_d3(samples,freq,name,path)

def visualize_backward_diffusion_d3(samples, freq=1 ,name=None, path="./"):
    # Transforms it into a video which shows the denoising of each frame 
    # and then moves on to the next frame 
    sample = samples[0]
    
    video = []
    for i, s in enumerate(sample):
        if i%freq==0:
            vid = s[0].permute(1,2,3,0)
            for frame in vid:
     
                video.append(frame)
    
    video = torch.stack(video,dim=0)

    filename = os.path.join(path,"backward_process_sample.mp4")
    write_video(filename,video,fps=2)
            

def visualize_backward_diffusion_d2(samples, freq=1, name=None, path="./"):

    if not os.path.exists(path):
        os.makedirs(path)

    sample = samples[0]
    
    ncols = len(sample) / freq
    if ncols != int(ncols):
        ncols = int(ncols) + 1
    else:
        ncols = int(ncols)

    print(ncols)
    fig, axs = plt.subplots(nrows=1, ncols=ncols)

    count = 0
    for i in range(len(sample)-1):
        if i % freq == 0:
            print(i,count)
            image = sample[i][0].permute(1, 2, 0)
            axs[count].imshow(image)
            axs[count].set_axis_off()
            count += 1
    
    axs[count].imshow(sample[-1][0].permute(1, 2, 0))
    axs[count].set_axis_off()

    plt.tight_layout()
    if name==None:
        path = f"{path}/backward_process_samples.png"
    else: 
        path = f"{path}/{name}.png"
    plt.savefig(path)

def visualize_conditional_backward_diffusion(samples, gt, freq=1, name=None, path="./"):

    if not os.path.exists(path):
        os.makedirs(path)

    ncols = len(samples) / freq
    if ncols != int(ncols):
        ncols = int(ncols) + 2
    else:
        ncols = int(ncols) + 1

    fig, axs = plt.subplots(nrows=1, ncols=ncols)

    count = 0
    for i in range(len(samples)):
        if i % freq == 0:
            image = samples[i][0].permute(1, 2, 0)
            axs[count].imshow(image)
            axs[count].set_axis_off()
            count += 1

    axs[-1].imshow(gt)
    axs[-1].set_axis_off()

    plt.tight_layout()
    if name==None:
        path = f"{path}/conditional_backward_process_samples.png"
    else: 
        path = f"{path}/{name}.png"
    plt.savefig(path)
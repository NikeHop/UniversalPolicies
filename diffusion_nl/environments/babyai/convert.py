import os 
import pickle

import blosc 
from tqdm import tqdm 

def convert_traj2images(path:str,envs:set, filename:str = None):
    """
    Convert the collected demos to a dataset of images 
    """
    if filename==None:
        filename = "_".join(envs)
    
    total_images = []
    for env in envs:
        with open(os.path.join(path,f"standard_50000_demos_{env}.pkl"),"rb") as file:
            demos = pickle.load(file)
            for d in tqdm(demos[0]):
                traj = blosc.unpack_array(d[2])
                total_images += [blosc.pack_array(elem) for elem in traj]
    
    with open(os.path.join(path,f"observations_{filename}.pkl"),"wb+") as file:
        pickle.dump(total_images,file)


if __name__=="__main__":

    envs = {"debug"}
    path = "../../data/baby_ai/"
    
    convert_traj2images(path,envs)


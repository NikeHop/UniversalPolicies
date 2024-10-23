import random 

import numpy as np
import torch

def set_seed(seed:int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 

def get_cuda_memory_usage():
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(r/1024**3,a/1024**3)
    r,a = torch.cuda.mem_get_info()
    print(r / 1024**3, a / 1024**3)

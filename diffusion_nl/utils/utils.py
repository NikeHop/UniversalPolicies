import random

import numpy as np
import torch

from torch import Tensor


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cuda_memory_usage():
    """
    Prints the CUDA memory usage.

    This function prints the reserved and allocated CUDA memory usage in GB.

    Returns:
        None
    """
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(r / 1024**3, a / 1024**3)
    r, a = torch.cuda.mem_get_info()
    print(r / 1024**3, a / 1024**3)


def normalize(
    sample: Tensor, mean: Tensor, var: Tensor, reverse: bool = False
) -> Tensor:
    """
    Normalize or reverse normalize a sample using mean and variance.

    Args:
        sample (Tensor): The input sample to be normalized or reverse normalized.
        mean (Tensor): The mean value used for normalization.
        var (Tensor): The variance value used for normalization.
        reverse (bool, optional): If True, performs reverse normalization. Defaults to False.

    Returns:
        Tensor: The normalized or reverse normalized sample.
    """
    if not reverse:
        return (sample - mean) / var
    else:
        return sample * var + mean

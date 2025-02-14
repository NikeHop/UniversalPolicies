import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel


def set_seed(seed: int) -> None:
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


def get_cuda_memory_usage() -> None:
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


def get_embeddings(data: list[tuple], config: dict) -> dict:
    """
    Get embeddings for instructions.

    Args:
        data (list[tuple]): The data containing instructions.
        config (dict): The configuration dictionary.

    Returns:
        dict: The dictionary mapping instructions to embedding
    """
    instructions = list(set([sample[0] for sample in data]))

    if config["embeddings"]["type"] == "random":
        inst2embed = get_random_embeddings(instructions, config)

    elif config["embeddings"]["type"] == "t5":
        inst2embed = get_t5_embeddings(instructions, config)

    else:
        raise NotImplementedError("Embedding type is not implemented")

    return inst2embed


def get_random_embeddings(instructions: list[str], config: dict) -> dict:
    """
    Creates random embeddings for instructions.

    Args:
        instructions (list[tuple]): Instructions to embed.
        config (dict): The configuration dictionary.

    Returns:
        dict: The dictionary mapping instructions to embedding
    """

    inst2embed = {}
    embeddings = nn.Embedding(len(instructions), config["embeddings"]["size"])
    embeddings.requires_grad = False
    for i, inst in enumerate(sorted(instructions)):
        inst2embed[inst] = embeddings(torch.tensor(i))

    return inst2embed


def get_t5_embeddings(instructions: list[str], config: dict) -> dict:
    """
    Embed instruction via the T5-Language Model.

    Args:
        instructions (list[tuple]): Instructions to embed.
        config (dict): The configuration dictionary.

    Returns:
        dict: The dictionary mapping instructions to embedding
    """

    inst2embed = {}

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["embeddings"]["model"])
    encoder_model = T5EncoderModel.from_pretrained(config["embeddings"]["model"]).to(
        config["embeddings"]["device"]
    )

    # Tokenize
    inputs = tokenizer(
        instructions, return_tensors="pt", padding=True, truncation=True
    ).to(config["embeddings"]["device"])

    # Embed Instructions
    encoded_embeddings = []
    n_instructions = len(instructions)
    n_encoded_instructions = 0
    B = config["embeddings"]["batch_size"]
    pbar = tqdm.tqdm(total=n_instructions)
    with torch.no_grad():
        # Embed B instructions at a time
        while n_encoded_instructions < n_instructions:
            model_output = encoder_model(
                input_ids=inputs["input_ids"][
                    n_encoded_instructions : n_encoded_instructions + B
                ],
                attention_mask=inputs["attention_mask"][
                    n_encoded_instructions : n_encoded_instructions + B
                ],
            )
            encoded_embeddings.append(
                model_output.last_hidden_state.mean(dim=1).detach().cpu()
            )
            n_encoded_instructions += B
            pbar.update(B)

    embeddings = torch.cat(encoded_embeddings, dim=0)

    for elem, instruction in zip(embeddings, instructions):
        inst2embed[instruction] = elem

    return inst2embed

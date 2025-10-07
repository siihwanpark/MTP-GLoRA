import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from .distributed import print_with_rank


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        all_params += param.numel()
    print_with_rank(f"All parameters: {all_params} | Trainable parameters: {trainable_params} ({trainable_params / all_params * 100:.2f}%)")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
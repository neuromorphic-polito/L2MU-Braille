from importlib import import_module
from torch import Tensor
import torch
from torch.nn import init
import numpy as np





def initialize_states(tensor: Tensor, shape: tuple, device):
    new_tensor = tensor.to('cpu')
    new_tensor = torch.Tensor(new_tensor)
    new_tensor = torch.zeros(shape, requires_grad=True)
    new_tensor = new_tensor.to(device)
    return new_tensor


def _lazy_import(package_name, module_name, class_name):
    module = import_module(module_name, package=package_name)
    return getattr(module, class_name)

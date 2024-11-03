from torch import nn
from torch.nn import init
from torch import Tensor
import numpy as np

def lecun_uniform(tensor: Tensor):
    fan_in = init._calculate_correct_fan(tensor, mode='fan_in')
    a = np.sqrt(3.0 / fan_in)
    return init.uniform_(tensor, -a, a)

class LCLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.__class__ = nn.Linear

    def reset_parameters(self) -> None:
        lecun_uniform(self.weight)


class CLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.__class__ = nn.Linear

    def reset_parameters(self) -> None:
        init.constant_(self.weight, 0)


class XavierLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.__class__ = nn.Linear

    def reset_parameters(self) -> None:
        init.xavier_normal_(self.weight)

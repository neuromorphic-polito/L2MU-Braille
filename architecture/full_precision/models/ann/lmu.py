import torch
from architecture.full_precision.models.ann.backend.lmu.lmu_cell import LMUCell
import torch.nn as nn

class LMU(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.lmu_cell = LMUCell(input_size=input_size, output=True, output_size=output_size, params=params)

    def forward(self, input):
        hidden, memory = torch.empty(0), torch.empty(0)

        # Record the output
        output_list = []

        for step in range(input.size(0)):
            output, hidden, memory = self.lmu_cell(input[step].flatten(1), spk_hidden=hidden,
                                              spk_memory=memory)
            output_list.append(output)

        return torch.stack(output_list, dim=0)

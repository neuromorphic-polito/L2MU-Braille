import torch
import torch.nn as nn
from architecture.full_precision.models.snn.backend.l2mu.l2mu_cell import L2MUCell


class L2MU(nn.Module):
    def __init__(self, input_size, output_size, params, neuron_type='Leaky'):
        super().__init__()
        self.l2mu_cell = L2MUCell(input_size=input_size, output=True, output_size=output_size, params=params, neuron_type=neuron_type)

    def forward(self, input):
        spk_hidden, spk_memory = self.l2mu_cell.init_cell()

        # Record the output
        spk_output_list = []

        for step in range(input.size(0)):
            spk_output, spk_hidden, spk_memory = self.l2mu_cell(input[step].flatten(1), spk_hidden=spk_hidden,
                                                           spk_memory=spk_memory)
            spk_output_list.append(spk_output)

        return torch.stack(spk_output_list, dim=0)

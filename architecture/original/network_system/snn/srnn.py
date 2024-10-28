import torch
import torch.nn as nn
from architecture.original.network_system.snn.backend_module.srnn import SRNNLeaky, SRNNSynaptic


class SRNNNet(nn.Module):
    def __init__(self, input_size, output_size, params, neuron_type='leaky', encoder='plain'):
        super().__init__()
        # initialize layers
        self.srnn = None
        self.select_neuron_type(input_size, output_size, neuron_type, encoder, params)

    def select_neuron_type(self, input_size, output_size, neuron_type, encoder, params):
        match neuron_type:
            case 'leaky':
                self.srnn = SRNNLeaky(input_size=input_size, output_size=output_size, encoder=encoder,
                                      params=params)
            case 'synaptic':
                self.srnn = SRNNSynaptic(input_size=input_size, output_size=output_size, encoder=encoder,
                                         params=params)

    def forward(self, input):

        # Record the output
        spk_output_list = []

        # Initialize SRNN
        self.srnn.init_network()

        for step in range(input.size(0)):
            spk_output = self.srnn(input[step].flatten(1))
            spk_output_list.append(spk_output)

        return torch.stack(spk_output_list, dim=0)

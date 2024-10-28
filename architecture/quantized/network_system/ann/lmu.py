import torch
from architecture.quantized.network_system.ann.backend_module.lmu.LMU import LMU
import torch.nn as nn
class LMUNet(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.lmu = LMU(input_size=input_size, output=True, output_size=output_size, params=params)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, input):
        input = self.quant(input)
        hidden, memory = torch.empty(0), torch.empty(0)

        # Record the output
        output_list = []

        for step in range(40):
            output, hidden, memory = self.lmu(input[step].flatten(1), spk_hidden=hidden,
                                              spk_memory=memory)
            output_list.append(output)

        x = self.dequant(torch.stack(output_list, dim=0))

        return x

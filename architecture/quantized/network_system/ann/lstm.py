from torch.nn import LSTM
from torch import nn
import torch
from torch.quantization import QuantStub, DeQuantStub


class LSTMNet(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.hidden_size = int(params['hidden_size'])
        self.lstm = LSTM(input_size=input_size, hidden_size=self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, input):

        new_input = torch.zeros(input.shape, device=input.device)
        new_input = self.quant(new_input)
        h0 = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)
        h0 = self.quant(h0)
        c0 = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)
        c0 = self.quant(c0)
        out, (hn, cn) = self.lstm(new_input, (h0, c0))
        output = self.output(out)
        output = self.dequant(output)

        return output

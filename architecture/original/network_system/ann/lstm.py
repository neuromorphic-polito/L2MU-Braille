from torch.nn import LSTM
from torch import nn
import torch


class LSTMNet(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.hidden_size = int(params['hidden_size'])
        self.lstm = LSTM(input_size=input_size, hidden_size=self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input):
        h0 = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)
        c0 = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)
        out, (hn, cn) = self.lstm(input, (h0, c0))
        output = self.output(out)

        return output

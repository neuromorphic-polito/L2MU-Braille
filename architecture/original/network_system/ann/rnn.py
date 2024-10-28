from torch.nn import RNN
from torch import nn
import torch


class RNNNet(nn.Module):
    def __init__(self, input_size, output_size, params):
        super().__init__()
        self.hidden_size = int(params['hidden_size'])
        self.rnn = RNN(input_size=input_size, hidden_size=self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input):
        h0 = torch.zeros(1, input.size(1), self.hidden_size, device=input.device)
        out, hn = self.rnn(input, h0)
        output = self.output(out)

        return output

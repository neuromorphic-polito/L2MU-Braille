import snntorch as snn
from architecture.original.signal_encoding.single_pop_encoder import SinglePopEncoderSynaptic
from architecture.original.signal_encoding import StackedPopEncoderSynaptic
import torch.nn as nn


class SRNN(nn.Module):

    def __init__(self, input_size, output_size, params, encoder='plain'):
        super().__init__()

        self.input_encoder = None
        self.output_enc = input_size
        self.select_input_encoder(input_size, encoder, params)

        self.hidden_size = int(params['hidden_size'])
        # Hidden layer
        self.fc1 = nn.Linear(self.output_enc, self.hidden_size)
        self.spk_hidden = snn.RSynaptic(beta=params['beta_hidden'], all_to_all=True,
                                        threshold=params['threshold_hidden'],
                                        alpha=params['alpha_hidden'], linear_features=params['hidden_size'])

        # Output layer
        self.fc2 = nn.Linear(self.hidden_size, output_size)
        self.spk_output = snn.Synaptic(beta=params['beta_output'], threshold=params['threshold_output'],
                                       alpha=params['alpha_output'])

        self.spk_hn, self.syn_hidden, self.mem_hidden = None, None, None
        self.syn_output, self.mem_output = None, None

    def select_input_encoder(self, input_size, encoder, params):
        match encoder:
            case 'plain':
                pass
            case 'single':
                self.input_encoder = SinglePopEncoderSynaptic(input_size=input_size, params=params)
                self.output_enc = self.input_encoder.output_size
            case 'stacked':
                self.input_encoder = StackedPopEncoderSynaptic(input_size=input_size, params=params)
                self.output_enc = self.input_encoder.output_size

    def init_network(self):
        if self.input_encoder:
            self.input_encoder.init_pop_encoder()
        self.spk_hn, self.syn_hidden, self.mem_hidden = self.spk_hidden.init_rsynaptic()
        self.syn_output, self.mem_output = self.spk_output.init_synaptic()

    def forward(self, input_):

        # Encoding layer if input_encoder is present
        spk_enc = self.input_encoder(input_) if self.input_encoder else input_

        # Hidden layer (recursive neuron)
        curr1 = self.fc1(spk_enc)
        self.spk_hn, self.syn_hidden, self.mem_hidden = self.spk_hidden(curr1, self.spk_hn, self.syn_hidden,
                                                                        self.mem_hidden)

        # Output layer
        curr2 = self.fc2(self.spk_hn)
        spk2, self.syn_output, self.mem_output = self.spk_output(curr2, self.syn_output, self.mem_output)

        return spk2

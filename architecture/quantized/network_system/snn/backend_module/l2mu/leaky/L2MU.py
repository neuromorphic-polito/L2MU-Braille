from snntorch import Leaky
import torch
from architecture.quantized.signal_encoding.single_pop_encoder import SinglePopEncoderLeaky
from architecture.quantized.signal_encoding.stacked_pop_encoder import StackedPopEncoderLeaky
from architecture.quantized.network_system.bare_lmu_network.bare_LMU import BareLMU
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub, DeQuantStub


class L2MU(BareLMU):

    def __init__(
            self,
            input_size,
            params,
            bias=False,
            trainable_theta=False,
            output_size=None,
            output=False,
            encoder='plain'
    ):

        super().__init__(input_size=input_size, hidden_size=int(params['hidden_size']),
                         memory_size=int(params['memory_size']), order=int(params['order']), theta=params['theta'],
                         output=output, output_size=output_size, bias=bias, trainable_theta=trainable_theta)

        self.input_encoder = None
        self.select_input_encoder(input_size, encoder, params)

        # self._gen_AB()
        self.init_parameters()

        self.spk_u = Leaky(beta=params['beta_spk_u'], threshold=params['threshold_spk_u'], learn_beta=True, learn_threshold=True)
        self.spk_h = Leaky(beta=params['beta_spk_h'], threshold=params['threshold_spk_h'], learn_beta=True, learn_threshold=True)
        self.spk_m = Leaky(beta=params['beta_spk_m'], threshold=params['threshold_spk_m'], learn_beta=True, learn_threshold=True)

        self.mem_m = None
        self.mem_h = None
        self.mem_u = None

        if self.output:
            self.spk_output = Leaky(beta=params['beta_spk_output'], threshold=params['threshold_spk_output'], learn_beta=True, learn_threshold=True)
            self.mem_output = None

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.operation = FloatFunctional()

    def select_input_encoder(self, input_size, encoder, params):
        match encoder:
            case 'plain':
                pass
            case 'single':
                self.input_encoder = SinglePopEncoderLeaky(input_size=input_size, params=params)
                self.input_size = self.input_encoder.output_size
            case 'stacked':
                self.input_encoder = StackedPopEncoderLeaky(input_size=input_size, params=params)
                self.input_size = self.input_encoder.output_size

    def init_l2mu(self):
        self.mem_u = self.spk_u.init_leaky()
        self.mem_h = self.spk_h.init_leaky()
        self.mem_m = self.spk_m.init_leaky()
        if self.input_encoder is not None:
            self.input_encoder.init_pop_encoder()

        if self.output:
            self.mem_output = self.spk_output.init_leaky()

        spike_h = torch.empty(0)
        spike_m = torch.empty(0)

        return spike_h, spike_m

    def forward(self, input_, spk_hidden, spk_memory):

        if len(spk_hidden) == 0 or len(spk_memory) == 0:
            # spk_hidden = torch.zeros(input_.size(0), self.hidden_size, device=input_.device, requires_grad=True)
            spk_hidden = torch.zeros((input_.shape[0], self.hidden_size), dtype=torch.float, device=input_.device,
                                     layout=torch.strided)
            spk_hidden.requires_grad_(True)
            # spk_memory = torch.zeros(input_.size(0), self.memory_size * self.order, device=input_.device,
            # requires_grad=True)
            spk_memory = torch.zeros((input_.shape[0], self.memory_size * self.order), dtype=torch.float,
                                     device=input_.device, layout=torch.strided)
            spk_memory.requires_grad_(True)
            spk_hidden = self.quant(spk_hidden)
            spk_memory = self.quant(spk_memory)
        # spk_memory = initialize_states(spk_memory, (input_, self.memory_size * self.order), input_.device)

        spk_input = self.input_encoder(input_) if self.input_encoder is not None else input_

        # Equation (7) of the paper
        curr_u = self.operation.add(self.e_x(spk_input), self.operation.add(self.e_h(spk_hidden), self.e_m(
            spk_memory)))

        # Equation (4) of the paper
        spk_u, self.mem_u = self.spk_u(curr_u, self.mem_u)

        # separate memory/order dimensions
        spk_u = torch.unsqueeze(spk_u, -1)
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size, self.order))

        # if self.discretizer == 'zoh' and self.trainable_theta:
        # self.lin_A.weight, self.lin_B.weight = self._cont2discrete_zoh(self._base_A * self.theta_inv,
        # self._base_B * self.theta_inv)
        curr_m = self.operation.add(self.A(spk_memory), self.B(spk_u))

        if self.discretizer == 'euler' and self.trainable_theta:
            curr_m += curr_m * self.theta_inv

        spk_memory, self.mem_m = self.spk_m(curr_m, self.mem_m)

        # re-combine memory/order dimensions
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size * self.order))

        # Equation (6) of the paper
        curr_h = self.operation.add(self.W_x(spk_input), self.operation.add(self.W_h(spk_hidden), self.W_m(
            spk_memory)))

        spk_hidden, self.mem_h = self.spk_h(curr_h, self.mem_h)  # [batch_size, hidden_size]

        # Output
        if self.output:
            curr_output = self.output_transformation(spk_hidden)
            spk_output, self.mem_output = self.spk_output(curr_output, self.mem_output)
            return spk_output, spk_hidden, spk_memory

        return spk_hidden, spk_hidden, spk_memory

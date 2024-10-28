import torch
from architecture.quantized.network_system.bare_lmu_network.bare_LMU import BareLMU
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import QuantStub, DeQuantStub


class LMU(BareLMU):

    def __init__(
            self,
            input_size,
            params,
            bias=False,
            trainable_theta=False,
            output_size=None,
            output=False,
    ):

        super().__init__(input_size=input_size, hidden_size=int(params['hidden_size']),
                         memory_size=int(params['memory_size']), order=int(params['order']), theta=params['theta'],
                         output=output, output_size=output_size, bias=bias, trainable_theta=trainable_theta)

        # self._gen_AB()
        self.init_parameters()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.operation = FloatFunctional()
        self.f = torch.nn.Tanh()

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

        spk_input = input_

        # Equation (7) of the paper
        spk_u = self.operation.add(self.e_x(spk_input), self.operation.add(self.e_h(spk_hidden), self.e_m(
            spk_memory)))

        # separate memory/order dimensions
        spk_u = torch.unsqueeze(spk_u, -1)
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size, self.order))

        # if self.discretizer == 'zoh' and self.trainable_theta:
        # self.lin_A.weight, self.lin_B.weight = self._cont2discrete_zoh(self._base_A * self.theta_inv,
        # self._base_B * self.theta_inv)
        spk_memory = self.operation.add(self.A(spk_memory), self.B(spk_u))

        if self.discretizer == 'euler' and self.trainable_theta:
            spk_memory += spk_memory * self.theta_inv

        # re-combine memory/order dimensions
        spk_memory = torch.reshape(spk_memory, (-1, self.memory_size * self.order))

        # Equation (6) of the paper
        spk_hidden = self.operation.add(self.W_x(spk_input), self.operation.add(self.W_h(spk_hidden), self.W_m(
            spk_memory)))  # [batch_size, hidden_size]

        spk_hidden = self.f(spk_hidden)

        # Output
        if self.output:
            spk_output = self.output_transformation(spk_hidden)
            return spk_output, spk_hidden, spk_memory

        return spk_hidden, spk_hidden, spk_memory

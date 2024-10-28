import torch
from torch import nn
from torch.nn import init
import numpy as np
from abc import abstractmethod
from architecture.utils import lecun_uniform


class LCLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.__class__ = nn.Linear

    def reset_parameters(self) -> None:
        lecun_uniform(self.weight)


class CLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.__class__ = nn.Linear

    def reset_parameters(self) -> None:
        init.constant_(self.weight, 0)


class XavierLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.__class__ = nn.Linear

    def reset_parameters(self) -> None:
        init.xavier_normal_(self.weight)


class BareLMU(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            memory_size,
            order,
            theta,
            output_size=None,
            output=False,
            bias=False,
            trainable_theta=False,
            discretizer='zoh'

    ):
        super().__init__()

        self.B = None
        self.A = None

        # Parameters passed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.order = order
        self._init_theta = theta
        self.output_size = output_size
        self.bias = bias
        self.output = output
        self.trainable_theta = trainable_theta
        self.discretizer = discretizer

        # Parameters to be learned
        self.W_h = None
        self.W_m = None
        self.W_x = None
        self.bias_m = None
        self.bias_h = None
        self.bias_x = None
        self.e_m = None
        self.e_h = None
        self.e_x = None
        self.theta_inv = None
        self.output_transformation = None

    def init_parameters(self):
        if self.trainable_theta:
            self.theta_inv = nn.Parameter(torch.empty(()))
        else:
            self.theta_inv = 1 / self._init_theta

        self.e_x = LCLinear(self.input_size, self.memory_size, bias=False)
        self.e_h = LCLinear(self.hidden_size, self.memory_size, bias=False)
        self.e_m = CLinear(self.memory_size * self.order, self.memory_size, bias=False)
        # Kernels

        self.A = CLinear(self.order, self.order, bias=False)
        self.B = CLinear(1, self.order, bias=False)
        self._gen_AB()

        self.W_x = XavierLinear(self.input_size, self.hidden_size, bias=False)
        self.W_h = XavierLinear(self.hidden_size, self.hidden_size, bias=False)
        self.W_m = XavierLinear(self.memory_size * self.order, self.hidden_size, bias=False)

        if self.output:
            self.output_transformation = nn.Linear(self.hidden_size, self.output_size)

    @property
    def theta(self):
        if self.trainable_theta:
            return 1 / self.theta_inv
        return self._init_theta

    def _gen_AB(self):
        """Generates A and B matrices."""

        # compute analog A/B matrices
        Q = np.arange(self.order, dtype=np.float64)
        R = (2 * Q + 1)[:, None]
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # discretize matrices
        if self.discretizer == "zoh":
            # save the un-discretized matrices for use in .call
            _base_A = torch.FloatTensor(A.T)
            _base_B = torch.FloatTensor(B.T)

            if self.trainable_theta:
                self._base_A = nn.Parameter(_base_A, requires_grad=False)
                self._base_B = nn.Parameter(_base_B, requires_grad=False)
            else:
                self._base_A = torch.tensor(0)
                self._base_B = torch.tensor(0)

            A, B = self._cont2discrete_zoh(
                _base_A / self._init_theta, _base_B / self._init_theta
            )

            self.A.weight = nn.Parameter(A, requires_grad=False)
            self.B.weight = nn.Parameter(B, requires_grad=False)

        else:
            if not self.trainable_theta:
                A = A.T / self._init_theta + np.eye(self.order)
                B = B.T / self._init_theta

            self.A.weight = nn.Parameter(A, requires_grad=False)
            self.A.weight = nn.Parameter(B, requires_grad=False)

    @staticmethod
    def _cont2discrete_zoh(A, B):
        """
        Function to discretize A and B matrices using Zero Order Hold method.

        Functionally equivalent to
        ``scipy.signal.cont2discrete((A.T, B.T, _, _), method="zoh", dt=1.0)``
        (but implemented in Pytorch so that it is differentiable).

        Note that this accepts and returns matrices that are transposed from the
        standard linear system implementation (as that makes it easier to use in
        `.call`).
        """

        # combine A/B and pad to make square matrix
        em_upper = torch.concat([A, B], dim=0)  # pylint: disable=no-value-for-parameter
        padding = (0, B.shape[0], 0, 0)
        em = torch.nn.functional.pad(em_upper, padding)

        # compute matrix exponential
        ms = torch.matrix_exp(em)

        # slice A/B back out of combined matrix
        discreet_A = ms[: A.shape[0], : A.shape[1]]
        discreet_B = ms[A.shape[0]:, : A.shape[1]]
        discreet_B = discreet_B.reshape(discreet_B.shape[1], discreet_B.shape[0])

        return discreet_A, discreet_B

    @classmethod
    @abstractmethod
    def forward(self, input_, _h, _m):
        pass

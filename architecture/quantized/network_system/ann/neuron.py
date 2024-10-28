import inspect
from warnings import warn
# from snntorch.surrogate import StraightThroughEstimator, atan, straight_through_estimator
import torch
import torch.nn as nn

import torch
import math


@torch.fx.wrap
def check_shape(x, y):
    return 1 if x.size() == y.size() else 0


# Spike-gradient functions

# slope = 25
# """``snntorch.surrogate.slope``
# parameterizes the transition rate of the surrogate gradients."""


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight Through Estimator.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                \\frac{∂S}{∂U}=1


    """

    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def straight_through_estimator():
    """Straight Through Estimator surrogate gradient enclosed
    with a parameterized slope."""

    def inner(x):
        return StraightThroughEstimator.apply(x)

    return inner


class Triangular(torch.autograd.Function):
    """
    Triangular Surrogate Gradient.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of the triangular function.

        .. math::

                \\frac{∂S}{∂U}=\\begin{cases} U_{\\rm thr} &
                \\text{if U < U$_{\\rm thr}$} \\\\
                -U_{\\rm thr}  & \\text{if U ≥ U$_{\\rm thr}$}
                \\end{cases}


    """

    @staticmethod
    def forward(ctx, input_, threshold):
        ctx.save_for_backward(input_)
        ctx.threshold = threshold
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ctx.threshold
        grad[input_ >= 0] = -grad[input_ >= 0]
        return grad, None


def triangular(threshold=1):
    """Triangular surrogate gradient enclosed with
    a parameterized threshold."""
    threshold = threshold

    def inner(x):
        return Triangular.apply(x, threshold)

    return inner


class FastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                S&≈\\frac{U}{1 + k|U|} \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{(1+k|U|)^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.fast_sigmoid(slope=25)``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in
    Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def fast_sigmoid(slope=25):
    """FastSigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return FastSigmoid.apply(x, slope)

    return inner


class ATan(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of shifted arc-tan function.

        .. math::

                S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{π}\\frac{1}{(1+(πU\\frac{α}{2})^2)}


    α defaults to 2, and can be modified by calling \
        ``surrogate.atan(alpha=2)``.

    Adapted from:

    *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang,
    Y. Tian (2021) Incorporating Learnable Membrane Time Constants
    to Enhance Learning of Spiking Neural Networks. Proc. IEEE/CVF
    Int. Conf. Computer Vision (ICCV), pp. 2661-2671.*"""

    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = torch.sign(torch.relu_(input_))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
                ctx.alpha
                / 2
                / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2))
                * grad_input
        )
        return grad, None


def atan(alpha=2.0):
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    alpha = alpha

    def inner(x):
        return ATan.apply(x, alpha)

    return inner


@staticmethod
class Heaviside(torch.autograd.Function):
    """Default spiking function for neuron.

    **Forward pass:** Heaviside step function shifted.

    .. math::

        S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
        0 & \\text{if U < U$_{\\rm thr}$}
        \\end{cases}

    **Backward pass:** Heaviside step function shifted.

    .. math::

        \\frac{∂S}{∂U}=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
        0 & \\text{if U < U$_{\\rm thr}$}
        \\end{cases}

    Although the backward pass is clearly not the analytical
    solution of the forward pass, this assumption holds true
    on the basis that a reset necessarily occurs after a spike
    is generated when :math:`U ≥ U_{\\rm thr}`."""

    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad = grad_output * out
        return grad


def heaviside():
    """Heaviside surrogate gradient wrapper."""

    def inner(x):
        return Heaviside.apply(x)

    return inner


class Sigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of sigmoid function.

        .. math::

                S&≈\\frac{1}{1 + {\\rm exp}(-kU)} \\\\
                \\frac{∂S}{∂U}&=\\frac{k
                {\\rm exp}(-kU)}{[{\\rm exp}(-kU)+1]^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.sigmoid(slope=25)``.


    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning
    in Multilayer Spiking
    Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
                grad_input
                * ctx.slope
                * torch.exp(-ctx.slope * input_)
                / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        return grad, None


def sigmoid(slope=25):
    """Sigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return Sigmoid.apply(x, slope)

    return inner


class SpikeRateEscape(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of Boltzmann Distribution.

        .. math::

                \\frac{∂S}{∂U}=k{\\rm exp}(-β|U-1|)

    :math:`β` defaults to 1, and can be modified by calling \
        ``surrogate.spike_rate_escape(beta=1)``.
    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.spike_rate_escape(slope=25)``.


    Adapted from:

    * Wulfram Gerstner and Werner M. Kistler,
    Spiking neuron models: Single neurons, populations, plasticity.
    Cambridge University Press, 2002.*"""

    @staticmethod
    def forward(ctx, input_, beta, slope):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
                grad_input
                * ctx.slope
                * torch.exp(-ctx.beta * torch.abs(input_ - 1))
        )
        return grad, None, None


def spike_rate_escape(beta=1, slope=25):
    """SpikeRateEscape surrogate gradient
    enclosed with a parameterized slope."""
    beta = beta
    slope = slope

    def inner(x):
        return SpikeRateEscape.apply(x, beta, slope)

    return inner


class StochasticSpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U}
            & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator,
    where the subthreshold gradient is sampled from uniformly
    distributed noise on the interval :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`,
    where :math:`μ` is the mean and :math:`σ^2` is the variance.

        .. math::

            S&≈\\begin{cases} \\frac{U(t)}{U}\\Big{|}_{U(t)→U_{\\rm thr}}
            & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases} \\\\
            \\frac{∂S}{∂U}&=\\begin{cases} 1  & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    :math:`μ` defaults to 0, and can be modified by calling \
        ``surrogate.SSO(mean=0)``.

    :math:`σ^2` defaults to 0.2, and can be modified by calling \
        ``surrogate.SSO(variance=0.5)``.

    The above defaults set the gradient to the following expression:

    .. math::

                \\frac{∂S}{∂U}&=\\begin{cases} 1
                & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    """

    @staticmethod
    def forward(ctx, input_, mean, variance):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.mean = mean
        ctx.variance = variance
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * out + (grad_input * (~out.bool()).float()) * (
                (torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance
        )
        return grad, None, None


def SSO(mean=0, variance=0.2):
    """Stochastic spike operator gradient enclosed with a parameterized mean
    and variance."""
    mean = mean
    variance = variance

    def inner(x):
        return StochasticSpikeOperator.apply(x, mean, variance)

    return inner


class LeakySpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Leaky gradient of spike operator, where
    the subthreshold gradient is treated as a small constant slope.

        .. math::

                S&≈\\begin{cases} \\frac{U(t)}{U}\\Big{|}_{U(t)→U_{\\rm thr}}
                & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                kU & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} 1
                & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                k & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    :math:`k` defaults to 0.1, and can be modified by calling \
        ``surrogate.LSO(slope=0.1)``.

    The gradient is identical to that of a threshold-shifted Leaky ReLU."""

    @staticmethod
    def forward(ctx, input_, slope):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        ctx.slope = slope
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
                grad_input * out + (~out.bool()).float() * ctx.slope * grad_input
        )
        return grad


def LSO(slope=0.1):
    """Leaky spike operator gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return StochasticSpikeOperator.apply(x, slope)

    return inner


class SparseFastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function clipped below B.

        .. math::

                S&≈\\frac{U}{1 + k|U|}H(U-B) \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{(1+k|U|)^2}
                & \\text{\\rm if U > B}
                0 & \\text{\\rm otherwise}
                \\end{cases}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.SFS(slope=25)``.
    :math:`B` defaults to 1, and can be modified by calling \
        ``surrogate.SFS(B=1)``.

    Adapted from:

    *N. Perez-Nieves and D.F.M. Goodman (2021) Sparse Spiking
    Gradient Descent. https://arxiv.org/pdf/2105.08810.pdf.*"""

    @staticmethod
    def forward(ctx, input_, slope, B):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.B = B
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
                grad_input
                / (ctx.slope * torch.abs(input_) + 1.0) ** 2
                * (input_ > ctx.B).float()
        )
        return grad, None, None


def SFS(slope=25, B=1):
    """SparseFastSigmoid surrogate gradient enclosed with a
    parameterized slope and sparsity threshold."""
    slope = slope
    B = B

    def inner(x):
        return SparseFastSigmoid.apply(x, slope, B)

    return inner


class CustomSurrogate(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** User-defined custom surrogate gradient function.

    The user defines the custom surrogate gradient in a separate function.
    It is passed in the forward static method and used in the backward
    static method.

    The arguments of the custom surrogate gradient function are always
    the input of the forward pass (input_), the gradient of the input
    (grad_input) and the output of the forward pass (out).

    ** Important Note: The hyperparameters of the custom surrogate gradient
    function have to be defined inside of the function itself. **

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn
        from snntorch import surrogate

        def custom_fast_sigmoid(input_, grad_input, spikes):
            ## The hyperparameter slope is defined inside the function.
            slope = 25
            grad = grad_input / (slope * torch.abs(input_) + 1.0) ** 2
            return grad

        spike_grad = surrogate.custom_surrogate(custom_fast_sigmoid)

        net_seq = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True,
                            output=True)
                    ).to(device)

    """

    @staticmethod
    def forward(ctx, input_, custom_surrogate_function):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.custom_surrogate_function = custom_surrogate_function
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, out = ctx.saved_tensors
        custom_surrogate_function = ctx.custom_surrogate_function

        grad_input = grad_output.clone()
        grad = custom_surrogate_function(input_, grad_input, out)
        return grad, None


def custom_surrogate(custom_surrogate_function):
    """Custom surrogate gradient enclosed within a wrapper."""
    func = custom_surrogate_function

    def inner(data):
        return CustomSurrogate.apply(data, func)

    return inner


# class InverseSpikeOperator(torch.autograd.Function):
#     """
#     Surrogate gradient of the Heaviside step function.

#     **Forward pass:** Spike operator function.

#         .. math::

#             S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥
#             U$_{\\rm thr}$} \\\\
#             0 & \\text{if U < U$_{\\rm thr}$}
#             \\end{cases}

#     **Backward pass:** Gradient of spike operator.

#         .. math::

#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 0 & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling
#     ``surrogate.spike_operator(threshold=1)``.
#     .. warning:: ``threshold`` should match the threshold of the neuron,
#     which defaults to 1 as well.

#                 """

#     @staticmethod
#     def forward(ctx, input_, threshold=1):
#         out = (input_ > 0).float()
#         ctx.save_for_backward(input_, out)
#         ctx.threshold = threshold
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_, out) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = (grad_input * out) / (input_ + ctx.threshold)
#         return grad, None


# def inverse_spike_operator(threshold=1):
#     """Spike operator gradient enclosed with a parameterized threshold."""
#     threshold = threshold

#     def inner(x):
#         return InverseSpikeOperator.apply(x, threshold)

#     return inner


# class InverseStochasticSpikeOperator(torch.autograd.Function):
#     """
#     Surrogate gradient of the Heaviside step function.

#     **Forward pass:** Spike operator function.

#         .. math::

#             S=\\begin{cases} \\frac{U(t)}{U}
#             & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#             0 & \\text{if U < U$_{\\rm thr}$}
#             \\end{cases}

#     **Backward pass:** Gradient of spike operator,
#     where the subthreshold gradient is sampled from
#     uniformly distributed noise on the interval
#     :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`,
#     where :math:`μ` is the mean and :math:`σ^2` is the variance.

#         .. math::

#                 S&≈\\begin{cases} \\frac{U(t)}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.5, 0.5) + μ) σ^2
#                 & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.5, 0.5) + μ) σ^2
#                 & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling
#     ``surrogate.SSO(threshold=1)``.

#     :math:`μ` defaults to 0, and can be modified by calling
#     ``surrogate.SSO(mean=0)``.

#     :math:`σ^2` defaults to 0.2, and can be modified by calling
#     ``surrogate.SSO(variance=0.5)``.

#     The above defaults set the gradient to the following expression:

#     .. math::

#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     .. warning:: ``threshold`` should match the threshold of the neuron,
#     which defaults to 1 as well.

#     """

#     @staticmethod
#     def forward(ctx, input_, threshold=1, mean=0, variance=0.2):
#         out = (input_ > 0).float()
#         ctx.save_for_backward(input_, out)
#         ctx.threshold = threshold
#         ctx.mean = mean
#         ctx.variance = variance
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_, out) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = (grad_input * out) / (input_ + ctx.threshold) + (
#             grad_input * (~out.bool()).float()
#         ) * ((torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance)

#         return grad, None, None, None


# def ISSO(threshold=1, mean=0, variance=0.2):
#     """Stochastic spike operator gradient enclosed with a parameterized
#     threshold, mean and variance."""
#     threshold = threshold
#     mean = mean
#     variance = variance

#     def inner(x):
#         return InverseStochasticSpikeOperator.
#         apply(x, threshold, mean, variance)

#     return inner


# piecewise linear func
# tanh surrogate func


__all__ = [
    "SpikingNeuron",
    "LIF",
    "_SpikeTensor",
    "_SpikeTorchConv",
    "Leaky"
]

dtype = torch.float


class SpikingNeuron(nn.Module):
    """Parent class for spiking neuron models."""

    instances = []
    """Each :mod:`snntorch.SpikingNeuron` neuron
    (e.g., :mod:`snntorch.Synaptic`) will populate the
    :mod:`snntorch.SpikingNeuron.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the
    argument `init_hidden=True`."""

    reset_dict = {
        "subtract": 0,
        "zero": 1,
        "none": 2,
    }

    def __init__(
            self,
            threshold=1.0,
            spike_grad=None,
            surrogate_disable=False,
            init_hidden=False,
            inhibition=False,
            learn_threshold=False,
            reset_mechanism="subtract",
            state_quant=False,
            output=False,
            graded_spikes_factor=1.0,
            learn_graded_spikes_factor=False,
    ):
        super().__init__()

        SpikingNeuron.instances.append(self)

        if surrogate_disable:
            self.spike_grad = self._surrogate_bypass
        elif spike_grad == None:
            self.spike_grad = atan()
        else:
            self.spike_grad = spike_grad

        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.output = output
        self.surrogate_disable = surrogate_disable

        self._snn_cases(reset_mechanism, inhibition)
        self._snn_register_buffer(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )
        self._reset_mechanism = reset_mechanism

        self.state_quant = state_quant

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        spk = spk * self.graded_spikes_factor

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        # reset = spk.clone().detach()

        return spk

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset

    def _snn_cases(self, reset_mechanism, inhibition):
        self._reset_cases(reset_mechanism)

        if inhibition:
            warn(
                "Inhibition is an unstable feature that has only been tested "
                "for dense (fully-connected) layers. Use with caution!",
                UserWarning,
            )

    def _reset_cases(self, reset_mechanism):
        if (
                reset_mechanism != "subtract"
                and reset_mechanism != "zero"
                and reset_mechanism != "none"
        ):
            raise ValueError(
                "reset_mechanism must be set to either 'subtract', "
                "'zero', or 'none'."
            )

    def _snn_register_buffer(
            self,
            threshold,
            learn_threshold,
            reset_mechanism,
            graded_spikes_factor,
            learn_graded_spikes_factor,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""

        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override
            # reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(SpikingNeuron.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

    def _graded_spikes_buffer(
            self, graded_spikes_factor, learn_graded_spikes_factor
    ):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer("graded_spikes_factor", graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict
        for mapping."""
        reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[reset_mechanism]
        )
        self.register_buffer("reset_mechanism_val", reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered
        to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[new_reset_mechanism]
        )
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances`
        when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()


class LIF(SpikingNeuron):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(
            self,
            beta,
            threshold=1.0,
            spike_grad=None,
            surrogate_disable=False,
            init_hidden=False,
            inhibition=False,
            learn_beta=False,
            learn_threshold=False,
            reset_mechanism="subtract",
            state_quant=False,
            output=False,
            graded_spikes_factor=1.0,
            learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._lif_register_buffer(
            beta,
            learn_beta,
        )
        self._reset_mechanism = reset_mechanism

    def _lif_register_buffer(
            self,
            beta,
            learn_beta,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _V_register_buffer(self, V, learn_V):
        if V is not None:
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @staticmethod
    def init_rleaky():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, mem

    @staticmethod
    def init_rsynaptic():
        """
        Used to initialize spk, syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, syn, mem


class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and
    replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
            self,
            *args,
            init_flag=True,
    ):
        # super().__init__() # optional
        self.init_flag = init_flag


def _SpikeTorchConv(*args, input_):
    """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""

    states = []
    # if len(input_.size()) == 0:
    #     _batch_size = 1  # assume batch_size=1 if 1D input
    # else:
    #     _batch_size = input_.size(0)
    if (
            len(args) == 1 and type(args) is not tuple
    ):  # if only one hidden state, make it iterable
        args = (args,)
    for arg in args:
        arg = arg.to("cpu")
        arg = torch.Tensor(arg)  # wash away the SpikeTensor class
        arg = torch.zeros_like(input_, requires_grad=True)
        states.append(arg)
    if len(states) == 1:  # otherwise, list isn't unpacked
        return states[0]

    return states


class Leaky(LIF):
    """
    First-order leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have
    `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1] - RU_{\\rm thr}

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0`
    whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - R(βU[t] + I_{\\rm in}[t+1])

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise \
        :math:`R = 0`
    * :math:`β` - Membrane potential decay rate

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Leaky(beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Leaky(beta=beta)

            def forward(self, x, mem1, spk1, mem2):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                return mem1, spk1, mem2, spk2


    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron).
    :type beta: float or torch.tensor

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to \
    :math:`mem` each time the threshold is met. Reset-by-subtraction: \
        "subtract", reset-to-zero: "zero", none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden state :math:`mem` is quantized_version
        to a valid state for the forward pass. Defaults to False
    :type state_quant: spike_core_pq function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are
        returned when neuron is called. Defaults to False
    :type output: bool, optional

    :param graded_spikes_factor: output spikes are scaled this value, if specified. Defaults to 1.0
    :type graded_spikes_factor: float or torch.tensor

    :param learn_graded_spikes_factor: Option to enable learnable graded spikes. Defaults to False
    :type learn_graded_spikes_factor: bool, optional

    :param reset_delay: If `True`, a spike is returned with a one-step delay after the threshold is reached.
        Defaults to True
    :type reset_delay: bool, optional

    Inputs: \\input_, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input
            features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the
            initial membrane potential for each element in the batch.

    Outputs: spk, mem_1
        - **spk** of shape `(batch, input_size)`: tensor containing the
            output spikes.
        - **mem_1** of shape `(batch, input_size)`: tensor containing the
            next membrane potential for each element in the batch

    Learnable Parameters:
        - **Leaky.beta** (torch.Tensor) - optional learnable weights must be
            manually passed in, of shape `1` or (input_size).
        - **Leaky.threshold** (torch.Tensor) - optional learnable thresholds
            must be manually passed in, of shape `1` or`` (input_size).

    """

    def __init__(
            self,
            beta,
            threshold=1.0,
            spike_grad=None,
            surrogate_disable=False,
            init_hidden=False,
            inhibition=False,
            learn_beta=False,
            learn_threshold=False,
            reset_mechanism="subtract",
            state_quant=False,
            output=False,
            graded_spikes_factor=1.0,
            learn_graded_spikes_factor=False,
            reset_delay=True,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self.k = None
        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

        if not self.reset_delay and self.init_hidden:
            raise NotImplementedError(
                "`reset_delay=True` is only supported for `init_hidden=False`"
            )

    def _init_mem(self):
        mem = torch.zeros(1)
        self.register_buffer("mem", mem)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)

    def init_leaky(self):
        """Deprecated, use :class:`Leaky.reset_mem` instead"""
        self.reset_mem()
        return self.mem

    def forward(self, input_, mem=None):


        if self.init_hidden and not mem == None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )

        self.reset = self.mem_reset(self.mem)
        mem = self.state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.inhibition:
            spk = self.fire_inhibition(
                self.mem.size(0), self.mem
            )  # batch_size
        else:
            spk = self.fire(mem)

        if not self.reset_delay:
            do_reset = (
                    spk / self.graded_spikes_factor - self.reset
            )  # avoid double reset
            if self.reset_mechanism_val == 0:  # reset by subtraction
                mem = mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:  # reset to zero
                mem = mem - do_reset * self.mem

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:

            return spk, mem

    def _base_state_function(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn

    def _base_sub(self, input_):
        return self._base_state_function(input_) - self.reset * self.threshold

    def _base_zero(self, input_):
        self.mem = (1 - self.reset) * self.mem
        return self._base_state_function(input_)

    def _base_int(self, input_):
        return self._base_state_function(input_)

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Leaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Leaky):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )

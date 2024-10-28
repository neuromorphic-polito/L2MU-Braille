from architecture.utils import _lazy_import


def RNN(*args, **kwargs):
    return _lazy_import("architecture.original.network_system.ann", ".rnn", "RNNNet")(*args, **kwargs)


def LSTM(*args, **kwargs):
    return _lazy_import("architecture.original.network_system.ann", ".lstm", "LSTMNet")(*args, **kwargs)


def LMU(*args, **kwargs):
    return _lazy_import("architecture.original.network_system.ann", ".lmu", "LMUNet")(*args, **kwargs)

def L2MU(*args, **kwargs):
    return _lazy_import("architecture.original.network_system.snn", ".l2mu", "L2MUNet")(*args, **kwargs)


def SRNN(*args, **kwargs):
    return _lazy_import("architecture.original.network_system.snn", ".srnn", "SRNNNet")(*args, **kwargs)

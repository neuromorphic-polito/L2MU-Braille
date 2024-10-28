from architecture.utils import _lazy_import


def SRNNLeaky(*args, **kwargs):
    return _lazy_import("architecture.network_system.snn.backend_module.srnn.leaky", ".SRNN", "SRNN")(*args, **kwargs)


def SRNNSynaptic(*args, **kwargs):
    return _lazy_import("architecture.network_system.snn.backend_module.srnn.synaptic", ".SRNN", "SRNN")(*args,
                                                                                                          **kwargs)

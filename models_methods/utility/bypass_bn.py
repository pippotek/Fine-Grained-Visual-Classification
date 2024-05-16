import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

# This function is used to disable the running statistics of the BatchNorm layers in the model without setting the model to evaluation mode.

def disable_running_stats(model):
    """
    Used in the second forward-backward pass when updating the weighs.
    """
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum 
            module.momentum = 0 # setting the momentum to 0 disable the update of the running statistics

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

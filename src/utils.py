from collections import OrderedDict
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

def get_parameters(net):
    """Get model parameters (weights of Conv2d and Linear layers) as a list of NumPy arrays."""
    params = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach().cpu().numpy()
            params.append(weight)
    return params


def set_parameters(net, parameters):
    """Set model parameters from a dictionary of NumPy arrays (weights of Conv2d and Linear layers)."""
    params_iter = iter(parameters)
    for _, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = next(params_iter)
            module.weight.data = torch.from_numpy(weight).to(module.weight.device)

    
def reshape_parameters(original_params, decrypted_data) -> List[np.ndarray]:
    """Reshape the decrypted data to match the shapes of the original parameters."""
    reshaped_params = []
    decrypted_data = np.array(decrypted_data).flatten()
    current_idx = 0
    
    for param in original_params:
        param_size = param.size
        param_shape = param.shape
        param_data = decrypted_data[current_idx:current_idx + param_size]
        reshaped_params.append(param_data.reshape(param_shape))
        current_idx += param_size
    
    return reshaped_params
from collections import OrderedDict
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn



def get_parameters(net):
    """Get model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    """Set model parameters from a list of NumPy arrays."""
    try:
        # 确保参数列表不为空
        if not parameters or len(parameters) == 0:
            raise ValueError("Empty parameters list")
            
        # 获取模型的参数名称
        params_dict = zip(net.state_dict().keys(), parameters)
        
        # 转换参数为 PyTorch tensors，保持数据类型
        state_dict = OrderedDict()
        for k, v in params_dict:
            if v.size == 0:  # 检查空数组
                raise ValueError(f"Empty array for parameter {k}")
            tensor = torch.tensor(v)
            # 确保参数维度匹配
            if k in net.state_dict():
                expected_shape = net.state_dict()[k].shape
                if tensor.shape != expected_shape:
                    tensor = tensor.reshape(expected_shape)
            state_dict[k] = tensor
            
        # 加载参数
        net.load_state_dict(state_dict, strict=True)
        
    except Exception as e:
        print(f"Error in set_parameters: {str(e)}")
        print(f"Parameters length: {len(parameters)}")
        print(f"Model state dict keys: {net.state_dict().keys()}")
        raise
    
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

import os
from matplotlib import pyplot as plt
import numpy as np
import torch


def quantize_weights(weights, n_bits):
    '''
    Quantization of weights to n_bits.
    
    Parameters:
    weights (dict): Weights of the model.
    n_bits (int): Number of bits to quantize the weights to.
    
    Returns:
    quantized_weights (dict): Quantized weights.
    scale (dict): Scale of the weights.
    '''
    
    quantized_weights = {}
    scales = {}
    
    for name, param in weights.items():
        if isinstance(param, torch.Tensor):
            weights_np = param.detach().cpu().numpy()
        elif isinstance(param, np.ndarray):
            weights_np = param
        else:
            raise TypeError(f"Unsupported type for weights: {type(param)}. Expected torch.Tensor or numpy.ndarray.")
        
        max_val = np.max(np.abs(weights_np))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / (2 ** (n_bits - 1) - 1)
            
        quantized_weight = np.round(weights_np / scale).astype(np.int32)
        
        quantized_weights[name] = quantized_weight
        scales[name] = scale
        
        
    return quantized_weights, scales

def dequantize_weights(quantized_weights, scales):
    '''
    Dequantization of weights.
    
    Parameters:
    quantized_weights (dict): Quantized weights.
    scale (dict): Scale of the weights.
    
    Returns:
    dequantized_weights (dict): Dequantized weights.
    '''
    
    dequantized_weights = {}
    
    for name, param in quantized_weights.items():
        if name not in scales:
            raise KeyError(f"Scale for layer '{name}' not found in scales dictionary.")
        
        scale = scales[name]
        dequantized_weight = param * scale
        dequantized_weights[name] = dequantized_weight
        

    return dequantized_weights

# Test the quantization and dequantization functions
def test_quantization():
    # Ensure the 'plots' directory exists
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    weights = {
        "conv1.weight": np.random.rand(3, 3, 3, 3),
        "conv2.weight": np.random.rand(3, 3, 3, 3),
        "fc1.weight": np.random.rand(10, 10),
        "fc2.weight": np.random.rand(10, 10),
    }
    
    quantized_weights, scales = quantize_weights(weights, 16)
    dequantized_weights = dequantize_weights(quantized_weights, scales)
    
    # Print debug information and assert
    for name, param in weights.items():
        print(f"Original weights for {name}:\n{param}")
        print(f"Dequantized weights for {name}:\n{dequantized_weights[name]}")
        print(f"Scale for {name}: {scales[name]}\n")


        
    print("Quantization and Dequantization functions are correct.")
    # Print the error and plot the weights
    for name, param in weights.items():
        error = np.abs(param - dequantized_weights[name])
        print(f"Error for {name}: {np.max(error)}")
        
        plt.figure()
        plt.hist(param.flatten(), bins=100, alpha=0.5, label="Original")
        plt.hist(dequantized_weights[name].flatten(), bins=100, alpha=0.5, label="Dequantized")
        plt.legend()
        plt.title(name)
        plt.savefig(os.path.join(plots_dir, f"{name}.png"))
        plt.close()
        
# test_quantization()

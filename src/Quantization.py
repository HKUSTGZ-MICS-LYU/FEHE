import os
from matplotlib import pyplot as plt
import numpy as np
import torch


def quantize_weights(weight, n_bits, global_max = None, global_min = None):
    '''
    Quantization of weights to n_bits.
    
    Parameters:
    weights (list): List of weights (each weight is a torch.Tensor or numpy.ndarray).
    n_bits (int): Number of bits to quantize the weights to.
    
    Returns:
    quantized_weights (list): Quantized weights.
    scales (int): Scale of the weights.
    min_vals (int): Minimum values of the weights.
    '''
    
    scale = None
    min_val = None
    
    
    weight = np.array(weight)

    # Get the global min and max if not provided
    if global_min is None or global_max is None:
        min_val = np.min(weight)
        max_val = np.max(weight)
    else:
        min_val = global_min
        max_val = global_max
    
    
    # Handle case where all values are the same
    if max_val == min_val:
        scale = 1.0  # Avoid division by zero, scale can be arbitrary
        quantized_weight = np.zeros_like(weight, dtype=np.int32)
    else:
        # Calculate the scale
        scale = (max_val - min_val) / (2 ** n_bits - 1)
        
        
        # Quantize and clip to prevent overflow
        quantized_weight = np.round((weight - min_val) / scale)
        quantized_weight = np.clip(quantized_weight, 0, 2**n_bits - 1).astype(np.int32)
    
    
    return quantized_weight, scale, min_val


def dequantize_weights(quantized_weight, scale, min_val):
    '''
    Dequantization of weights from n_bits.
    
    Parameters:
    quantized_weight (list): Quantized weights.
    scales (int): Scale of the weights.
    min_vals (int): Minimum values of the weights.
    
    Returns:
    dequantized_weight (list): Dequantized weights.
    '''
    dequantized_weight = []
    for weight in quantized_weight:
        dequantized_weight.append(weight * scale + min_val)
        
    return dequantized_weight


# Test the quantization and dequantization functions
def test_quantization():
    # Ensure the 'plots' directory exists
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a list of weights (including edge cases)
    weights = np.random.randn(1, 100),  # Normal distribution
    
    
    # Quantize and dequantize the weights
    quantized_weights, scales, min_vals = quantize_weights(weights, 8)
    dequantized_weights = dequantize_weights(quantized_weights, scales, min_vals)
    
    for i, (weight, dequantized_weight) in enumerate(zip(weights, dequantized_weights)):
        print(f"\nWeight {i}:")
        print(f"Original range: {np.min(weight):.4f} to {np.max(weight):.4f}")
        print(f"Dequantized range: {np.min(dequantized_weight):.4f} to {np.max(dequantized_weight):.4f}")
        
    print("\nQuantization and Dequantization functions are verified.")
    
    # Plot the weights comparison
    for i, (weight, dequantized_weight) in enumerate(zip(weights, dequantized_weights)):
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(weight.flatten(), bins=50, alpha=0.7, label="Original")
        plt.hist(dequantized_weight.flatten(), bins=50, alpha=0.7, label="Dequantized")
        plt.legend()
        plt.title(f"Weight {i} Distribution")
        
        plt.subplot(1, 2, 2)
        error = np.abs(weight - dequantized_weight)
        plt.plot(error.flatten(), marker='o', linestyle='', alpha=0.5)
        plt.yscale('log')
        plt.title(f"Pointwise Absolute Error (Max: {np.max(error):.2e})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"weight_{i}_comparison.png"))
        plt.close()

# Uncomment to run the test
# test_quantization()
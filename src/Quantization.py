
import numpy as np


def quantize_weights(weights, n_bits):
    """
    仅量化权重，将权重量化为n_bits的整数表示。

    参数:
    weights (np.ndarray): 原始权重数组。
    n_bits (int): 量化位数。

    返回:
    quantized_weights (np.ndarray): 量化后的整数权重。
    scale (float): 权重的缩放因子。
    """
    # 计算权重的最大绝对值
    max_abs = np.max(np.abs(weights))
    
    # 计算缩放因子，避免除以零
    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / (2**(n_bits - 1) - 1)  # 对称量化
    
    # 将权重除以缩放因子并四舍五入
    quantized_weights = np.round(weights / scale).astype(np.int32)
    
    # 定义量化的整数范围
    q_min = -2**(n_bits - 1)
    q_max = 2**(n_bits - 1) - 1
    
    # 剪裁到有效范围
    quantized_weights = np.clip(quantized_weights, q_min, q_max)
    
    return quantized_weights, scale

def dequantize_weights(quantized_weights, scale):
    """
    反量化权重，将整数量化的权重映射回浮点范围。

    参数:
    quantized_weights (np.ndarray): 量化后的整数权重。
    scale (float): 权重的缩放因子。

    返回:
    dequantized_weights (np.ndarray): 反量化后的浮点权重。
    """
    dequantized_weights = quantized_weights * scale
    return dequantized_weights
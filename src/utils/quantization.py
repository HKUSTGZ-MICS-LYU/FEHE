import os
from matplotlib import pyplot as plt
import numpy as np
import torch


class Quantizer:
    def __init__(self):
        pass
    
    def quantize_weights(self, weight, n_bits, global_max=None, global_min=None):
        """
        Naive quantization of weights to n_bits.
        """
        weight = np.array(weight)
        if global_min is None or global_max is None:
            min_val = np.min(weight)
            max_val = np.max(weight)
        else:
            min_val = global_min
            max_val = global_max

        if max_val == min_val:
            scale = 1.0
            quantized_weight = np.zeros_like(weight, dtype=np.int32)
        else:
            scale = (max_val - min_val) / (2 ** n_bits - 1)
            quantized_weight = np.round((weight - min_val) / scale)
            quantized_weight = np.clip(quantized_weight, 0, 2 ** n_bits - 1).astype(np.int32)

        return quantized_weight, scale, min_val
    
    def dequantize_weights(self, quantized_weight, scale, min_val):
        """
        Naive dequantization of weights from n_bits.
        """
        dequantized_weight = []
        for weight in quantized_weight:
            dequantized_weight.append(weight * scale + min_val)
        return dequantized_weight
    
    def quantize_weights_truncate(self, weight, n_bits, truncate_method='quantile', truncate_param=0.01):
        """
        Truncate quantization of weights to n_bits.
        """
        weight = np.array(weight)
        if truncate_method == 'quantile':
            lower = np.quantile(weight, truncate_param)
            upper = np.quantile(weight, 1 - truncate_param)
        elif truncate_method == 'sigma':
            mu, sigma = np.mean(weight), np.std(weight)
            lower = mu - 2 * sigma
            upper = mu + 2 * sigma
        else:
            lower = np.min(weight)
            upper = np.max(weight)
        weight_clipped = np.clip(weight, lower, upper)
        min_val, max_val = lower, upper

        if max_val == min_val:
            scale = 1.0
            quantized_weight = np.zeros_like(weight_clipped, dtype=np.int32)
        else:
            scale = (max_val - min_val) / (2 ** n_bits - 1)
            quantized_weight = np.round((weight_clipped - min_val) / scale)
            quantized_weight = np.clip(quantized_weight, 0, 2 ** n_bits - 1).astype(np.int32)
        return quantized_weight, scale, min_val
    
    def quantize_weights_layerwise(self, weight, n_bits, split_threshold=None):
        """
        Layerwise quantization of weights to n_bits.
        """
        weight = np.array(weight)
        if split_threshold is None:
            split_threshold = np.quantile(np.abs(weight), 0.95)
        small_vals = weight[np.abs(weight) <= split_threshold]
        large_vals = weight[np.abs(weight) > split_threshold]
        bits_small = n_bits - 2
        scale_small = (split_threshold - (-split_threshold)) / (2 ** bits_small - 1)
        quantized_small = np.round((small_vals + split_threshold) / scale_small)
        quantized_small = np.clip(quantized_small, 0, 2 ** bits_small - 1)
        bits_large = n_bits - bits_small
        scale_large = (np.max(large_vals) - np.min(large_vals)) / (2 ** bits_large - 1)
        quantized_large = np.round((large_vals - np.min(large_vals)) / scale_large)
        quantized_large = np.clip(quantized_large, 0, 2 ** bits_large - 1)
        return (quantized_small, quantized_large), (scale_small, scale_large), (-split_threshold, np.min(large_vals))   
    
    def dequantize_weights_layerwise(self, quantized_data, scales, min_vals, split_threshold=1.0):
        """
        Dequantization of layerwise quantized weights.
        """
        quantized_small, quantized_large = quantized_data
        scale_small, scale_large = scales
        min_small, min_large = -split_threshold, min_vals
        dequantized_small = quantized_small * scale_small + min_small
        if quantized_large.size == 0 or scale_large is None:
            dequantized_large = np.array([])
        else:
            dequantized_large = quantized_large * scale_large + min_large
        return np.concatenate([dequantized_small, dequantized_large])

    def log_quantize(self, weight, n_bits, base=2):
        """
        Logarithmic quantization of weights to n_bits.
        """
        weight = np.array(weight)
        sign = np.sign(weight)
        magnitude = np.log(np.abs(weight) + 1e-8) / np.log(base)
        max_mag = np.max(magnitude)
        min_mag = np.min(magnitude)
        scale = (max_mag - min_mag) / (2 ** n_bits - 1)
        quantized_mag = np.round((magnitude - min_mag) / scale)
        quantized_mag = np.clip(quantized_mag, 0, 2 ** n_bits - 1).astype(np.int32)
        return quantized_mag, scale, min_mag

    def log_dequantize(self, quantized_mag, scale, min_mag, base=2):
        """
        Dequantization of log-quantized weights.
        """
        dequantized_mag = quantized_mag * scale + min_mag
        dequantized_mag = np.clip(dequantized_mag, -50, 50)
        dequantized = np.sign(dequantized_mag) * (base ** np.abs(dequantized_mag) - 1e-8)
        return dequantized

    def dynamic_quantize(self, weight, n_bits):
        """
        Dynamic quantization of weights to n_bits.
        """
        weight = np.array(weight)
        max_val = np.max(np.abs(weight))
        scale = (2 ** (n_bits - 1) - 1) / max_val
        quantized_weight = np.round(weight * scale).astype(int)
        return quantized_weight, scale
    
    def dynamic_dequantize(self, quantized_weight, scale):
        """
        Dequantization of dynamic quantized weights.
        """
        dequantized_weight = quantized_weight / scale
        return dequantized_weight
    
    def symmetric_quantize(self, data, q_bit):
        max_val = np.max(np.abs(data))
        scale = (2 ** (q_bit - 1) - 1) / max_val  # 对称范围
        qw = np.round(data * scale).astype(int)   # 量化
        return qw, scale
    
    def symmetric_dequantize(self, qw, scale):
        return qw / scale
    
    def asymmetric_quantize(self, data, q_bit):
        min_val = np.min(data)
        max_val = np.max(data)
        scale = (2 ** q_bit - 1) / (max_val - min_val)
        zero_point = np.round(-min_val * scale).astype(int)
        qw = np.round(data * scale + zero_point).astype(int)
        return qw, scale, zero_point
    
    def asymmetric_dequantize(self, qw, scale, zero_point):
        return (qw - zero_point) / scale
    

    
    def block_quantize(self, data, q_bit, block_size=1024):
        qw = np.zeros_like(data, dtype=int)
        scales = []
        for i in range(0, len(data), block_size):
            block = data[i:i+block_size]
            max_val = np.max(np.abs(block))
            scale = (2 ** (q_bit - 1) - 1) / max_val if max_val != 0 else 1.0
            qw[i:i+block_size] = np.round(block * scale).astype(int)
            scales.append(scale)
        return qw, scales

    def block_dequantize(self, qw, scales, block_size=1024):
        deq = np.zeros_like(qw, dtype=float)
        num_blocks = len(scales)
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, len(qw))
            deq[start:end] = qw[start:end] / scales[i]
        return deq
        
    def sigma_quantize(self, data, sigma_bits, global_mu, global_sigma):
        """
        Sigma-based quantization: Divides data into regions based on sigma intervals (3sigma, 2-3sigma, 1-2sigma, 1sigma),
        and quantizes each region with specified bits.
        
        Args:
            data: Input weight data (numpy array)
            sigma_bits: List of bits for each region (e.g., [8, 6, 4, 2])
            global_mu: Global mean of the data
            global_sigma: Global standard deviation of the data
        
        Returns:
            quantized_parts: List of quantized data for each region
            params: Dictionary containing quantization parameters
        """
        data = np.array(data)
        mu = global_mu
        sigma = global_sigma
        n = len(sigma_bits)
   
        # Define masks for each region
        masks = []
        masks.append(np.abs(data - mu) > (n - 1) * sigma)   
        for i in range(1, n - 1):
            lower = (n - 1 - i) * sigma
            upper = (n - i) * sigma
            masks.append((np.abs(data - mu) > lower) & (np.abs(data - mu) <= upper))
        masks.append(np.abs(data - mu) <= sigma)
     
        flat_data = data.flatten()
        flattened_quantized = np.zeros_like(flat_data, dtype=np.int32)
        scales = []
        min_vals = []
        region_map = np.zeros_like(flat_data, dtype=np.int32)  
        
        for i, mask in enumerate(masks):
            flat_mask = mask.flatten()
            indices = np.where(flat_mask)[0]
            if len(indices) > 0:
                region_data = flat_data[indices]
                quantized_part, scale, min_val = self.quantize_weights(region_data, sigma_bits[i])
                flattened_quantized[indices] = quantized_part
                region_map[indices] = i
            else:
                scale, min_val = 0.0, 0.0
            scales.append(scale)
            min_vals.append(min_val)
            
        params = {
            'sigma_bits': sigma_bits,
            'scales': scales,
            'min_vals': min_vals,
            'region_map': region_map,
            'original_shape': data.shape
        }
        return flattened_quantized, params
    
    def sigma_dequantize(self, flattened_quantized, params):
        """
        Dequantization of flattened sigma-quantized weights
        
        Args:
            flattened_quantized: Flattened quantized data
            params: Dictionary containing region information
        
        Returns:
            dequantized_data: Reshaped dequantized data
        """
        scales = params['scales']
        min_vals = params['min_vals']
        region_map = params['region_map']
        original_shape = params['original_shape']
        
        # 确保输入数据是numpy数组
        flattened_quantized = np.array(flattened_quantized)
        dequantized_flat = np.zeros_like(flattened_quantized, dtype=float)
        
        # 对每个区域进行反量化
        for i in range(len(scales)):
            # 确保indices是整数类型
            region_indices = np.where(region_map == i)[0].astype(int)
            if len(region_indices) > 0:
                q_data = flattened_quantized[region_indices]
                dq_data = self.dequantize_weights(q_data, scales[i], min_vals[i])
                dequantized_flat[region_indices] = dq_data
        
        return dequantized_flat.reshape(original_shape)
            
    
    def quantize_weights_unified(self, weight, n_bits, method='naive', **kwargs):
        """
        Unified quantization of weights to n_bits. Support the motheds below:
        - naive: Naive quantization of weights to n_bits.
        - truncate: Truncate quantization of weights to n_bits.
        - layerwise: Layerwise quantization of weights to n_bits.
        - log: Logarithmic quantization of weights to n_bits.
        - dynamic: Dynamic quantization of weights to n_bits.
        - symmetric: Symmetric quantization of weights to n_bits.
        - asymmetric: Asymmetric quantization of weights to n_bits.
        - block: Block quantization of weights to n_bits.
        - sigma: Sigma-based quantization of weights to n_bits.
        """
        data = np.array(weight)
        params = {'method': method, 'n_bits': n_bits}

        if method == 'naive':
            qw, scale, min_val = self.quantize_weights(data, n_bits, kwargs.get('global_max'), kwargs.get('global_min'))
            params.update({'scale': scale, 'min_val': min_val})
        elif method == 'truncate':
            qw, scale, min_val = self.quantize_weights_truncate(data, n_bits, kwargs.get('truncate_method', 'quantile'), kwargs.get('truncate_param', 0.01))
            params.update({'scale': scale, 'min_val': min_val})
        elif method == 'layerwise':
            (qw_small, qw_large), (scale_small, scale_large), min_vals = self.quantize_weights_layerwise(data, n_bits, kwargs.get('split_threshold', None))
            qw = (qw_small, qw_large)
            params.update({'scales': (scale_small, scale_large), 'min_vals': min_vals})
        elif method == 'log':
            qw, scale, min_mag = self.log_quantize(data, n_bits, kwargs.get('base', 2))
            params.update({'scale': scale, 'min_mag': min_mag, 'base': kwargs.get('base', 2)})
        elif method == 'dynamic':
            qw, scale = self.dynamic_quantize(data, n_bits)
            params.update({'scale': scale})
        elif method == 'symmetric':
            qw, scale = self.symmetric_quantize(data, n_bits)
            params.update({'scale': scale})
        elif method == 'asymmetric':
            qw, scale, zero_point = self.asymmetric_quantize(data, n_bits)
            params.update({'scale': scale, 'zero_point': zero_point})
        elif method == 'block':
            qw, scales = self.block_quantize(data, n_bits, kwargs.get('block_size', 1024))
            params.update({'scales': scales, 'block_size': kwargs.get('block_size', 1024)})
        elif method == 'sigma':
            sigma_bits = kwargs.get('sigma_bits', [8, 6, 4, 2])
            global_mu = kwargs.get('global_mu', np.mean(data))
            global_sigma = kwargs.get('global_sigma', np.std(data))
            quantized_parts, sigma_params = self.sigma_quantize(data, sigma_bits, global_mu, global_sigma)
            params.update(sigma_params)
            return quantized_parts, params
        else:
            raise ValueError(f"Unsupported quantization method: {method}")
        
        return qw, params
    
    def dequantize_weights_unified(self, quantized_data, params):
        """
        统一反量化接口，对应 quantize_weights_unified 返回的数据和参数。
        """
        method = params['method']
        if method in ['naive', 'truncate']:
            scale = params['scale']
            min_val = params['min_val']
            return self.dequantize_weights(quantized_data, scale, min_val)
        elif method == 'layerwise':
            scales = params['scales']
            min_vals = params['min_vals']
            return self.dequantize_weights_layerwise(quantized_data, scales, min_vals)
        elif method == 'log':
            scale = params['scale']
            min_mag = params['min_mag']
            base = params.get('base', 2)
            return self.log_dequantize(quantized_data, scale, min_mag, base)
        elif method == 'dynamic':
            scale = params['scale']
            return self.dynamic_dequantize(quantized_data, scale)
        elif method == 'symmetric':
            scale = params['scale']
            return self.symmetric_dequantize(quantized_data, scale)
        elif method == 'asymmetric':
            scale = params['scale']
            zero_point = params['zero_point']
            return self.asymmetric_dequantize(quantized_data, scale, zero_point)
        elif method == 'block':
            scales = params['scales']
            block_size = params['block_size']
            return self.block_dequantize(quantized_data, scales, block_size)
        elif method == 'sigma':
            return self.sigma_dequantize(quantized_data, params)
        else:
            raise ValueError(f"Unsupported dequantization method: {method}")
    
        
        
    
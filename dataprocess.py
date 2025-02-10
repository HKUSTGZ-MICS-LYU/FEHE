import numpy as np
from sympy import nextprime
import tenseal as ts
from src.Quantization import Quantizer
import sys
import matplotlib.pyplot as plt


file_path = "src/encrypted/unencrypt_params_0.txt"
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(float(line.strip()))
    f.close()
print("============SIGMA QUANTIZATION============")
# Choose quantization method: e.g., 'naive', 'truncate', 'layerwise', 'log', 'dynamic', 'symmetric', 'asymmetric', 'block'
quant_method = 'sigma'
q_bit = 8
quantizer = Quantizer()
qw, params = quantizer.quantize_weights_unified(data, q_bit, method=quant_method, sigma_bits = [14,14,14,14])

if hasattr(qw, 'nbytes'):  # 如果是numpy数组
    size_bytes = qw.nbytes
else:  # 如果是列表或其他类型
    size_bytes = sys.getsizeof(qw)
print(f"Size of qw: {size_bytes} bytes ({size_bytes/1024:.2f} KB)")
# Dequantize the decrypted parameters
dequantized_params = quantizer.dequantize_weights_unified(qw, params)

# Compare dequantized data with original data
mse_data = np.mean((np.array(data) - dequantized_params)**2)
print(f"MSE (original vs dequantized): {mse_data}")
mae_data = np.mean(np.abs(np.array(data) - dequantized_params))
print(f"MAE (original vs dequantized): {mae_data}")
print("Original data (first 10):", data[:10])
print("Dequantized data (first 10):", dequantized_params[:10])

# print('\n')
# print("============BLOCK QUANTIZATION============")
# # Use block method to quantize the data
# quantizer = Quantizer()

# quantized_data, params = quantizer.quantize_weights_unified(data, 12, 'block', block_size=32)
# if hasattr(quantized_data, 'nbytes'):  # 如果是numpy数组
#     size_bytes = quantized_data.nbytes
# else:  # 如果是列表或其他类型
#     size_bytes = sys.getsizeof(quantized_data)
# print(f"Size of quantized_data: {size_bytes} bytes ({size_bytes/1024:.2f} KB)")
# # Dequantize the data
# dequantized_params = quantizer.dequantize_weights_unified(quantized_data, params)


# # Compare dequantized data with original data
# mse_data = np.mean((np.array(data) - dequantized_params)**2)
# print(f"MSE (original vs dequantized): {mse_data}")
# mae_data = np.mean(np.abs(np.array(data) - dequantized_params))
# print(f"MAE (original vs dequantized): {mae_data}")
# print("Original data (first 10):", data[:10])
# print("Dequantized data (first 10):", dequantized_params[:10])


    


# # Create a more comprehensive visualization
# plt.figure(figsize=(15, 10))

# # 1. Original vs Dequantized comparison
# plt.subplot(2, 2, 1)
# plt.plot(data[:100], 'b-', label='Original', alpha=0.7)
# plt.plot(dequantized_params[:100], 'r--', label='Decrypted', alpha=0.7)
# plt.title('Original vs Decrypted (First 100 samples)')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 2. Error distribution histogram
# plt.subplot(2, 2, 2)
# errors = np.array(data) - dequantized_params
# plt.hist(errors, bins=50, color='skyblue', alpha=0.7)
# plt.title('Error Distribution')
# plt.xlabel('Error Value')
# plt.ylabel('Frequency')
# plt.grid(True, alpha=0.3)

# # 3. Scatter plot of Original vs Dequantized
# plt.subplot(2, 2, 3)
# plt.scatter(data, dequantized_params, alpha=0.5, s=1)
# plt.plot([min(data), max(data)], [min(data), max(data)], 'r--', alpha=0.7)
# plt.title('Original vs Decrypted Scatter')
# plt.xlabel('Original Values')
# plt.ylabel('Decrypted Values')
# plt.grid(True, alpha=0.3)

# # 4. Relative error percentage
# plt.subplot(2, 2, 4)
# relative_error = np.abs(errors / (np.array(data) + 1e-10)) * 100
# plt.plot(relative_error[:100], 'g-', alpha=0.7)
# plt.title('Relative Error % (First 100 samples)')
# plt.xlabel('Sample Index')
# plt.ylabel('Relative Error %')
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

# # Print additional statistical information
# print(f"Standard deviation of errors: {np.std(errors):.6f}")
# print(f"Mean relative error (%): {np.mean(relative_error):.6f}")
# print(f"Max relative error (%): {np.max(relative_error):.6f}")

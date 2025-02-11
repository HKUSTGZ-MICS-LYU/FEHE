import numpy as np
from sympy import nextprime
import tenseal as ts
from src.quantization import Quantizer
import matplotlib.pyplot as plt

quanti_file = "src/client_0_quantized_params.txt"
dequanti_file = "src/decrypted_params_0.txt"
quanti_data = []
dequanti_data = []
with open(quanti_file, 'r') as f:
    for line in f:
        quanti_data.append(float(line.strip()))
f.close()
with open(dequanti_file, 'r') as f:
    for line in f:
        dequanti_data.append(float(line.strip()))
f.close()

# Compare dequantized data with original data
mse_data = np.mean((np.array(quanti_data) - dequanti_data)**2)
print(f"MSE (original vs dequantized): {mse_data}")
mae_data = np.mean(np.abs(np.array(quanti_data) - dequanti_data))
print(f"MAE (original vs dequantized): {mae_data}")
print("Quantized data (first 10):", quanti_data[:10])
print("Dequantized data (first 10):", dequanti_data[:10])

# Read data from file /hpc/home/connect.xmeng027/Work/FEHE/src/encrypted/unencrypt_params_0.txt
original_file = "src/client_0_params.txt"
original_data = []
dequanti_file = "src/dequantized_params_0.txt"
dequanti_data = []
with open(original_file, 'r') as f:
    for line in f:
        original_data.append(float(line.strip()))
f.close()
with open(dequanti_file, 'r') as f:
    for line in f:
        dequanti_data.append(float(line.strip()))
f.close()

# Compare dequantized data with original data
mse_data = np.mean((np.array(original_data) - dequanti_data)**2)
print(f"MSE (original vs dequantized): {mse_data}")
mae_data = np.mean(np.abs(np.array(original_data) - dequanti_data))
print(f"MAE (original vs dequantized): {mae_data}")
print("Original data (first 10):", original_data[:10])
print("Dequantized data (first 10):", dequanti_data[:10])


# data = []
# with open(file_path, 'r') as f:
#     for line in f:
#         data.append(float(line.strip()))
#     f.close()

# # Statistics of the data
# data = np.array(data)
# print("Data shape:", data.shape)
# print("Max:", np.max(data))
# print("Min:", np.min(data))
# print("Mean:", np.mean(data))
# print("Number of elements larger than 1:", np.sum(data > 1))
# print("Number of elements smaller than -1:", np.sum(data < -1))
# max_idx = np.argmax(data)
# min_idx = np.argmin(data)
# print("Max index:", max_idx)
# print("Min index:", min_idx)
# # 得到大于0但是最接近0的数和它的索引
# print("Closest to 0 but positive:", data[data > 0].min())
# print("Find the index of the closest to 0 but positive:", np.argmin(np.abs(data[data > 0])))
# # 得到小于0但是最接近0的数和它的索引
# print("Closest to 0 but negative:", data[data < 0].max())
# print("Find the index of the closest to 0 but negative:", np.argmax(np.abs(data[data < 0])))    




# # Use BFV Scheme
# context = ts.context(
#     ts.SCHEME_TYPE.BFV,
#     poly_modulus_degree=4096,
#     plain_modulus=1032193
# )
# context.generate_galois_keys()
# context.global_scale = 2 ** 40

# # Choose quantization method: e.g., 'naive', 'truncate', 'layerwise', 'log', 'dynamic', 'symmetric', 'asymmetric', 'block'


# quantizer = Quantizer()
# qw, params = quantizer.quantize_weights_unified(data, 
#                                                 n_bits = 8, 
#                                                 method = 'sigma', 
#                                                 global_max = 1, 
#                                                 global_min = -1, 
#                                                 global_sigma = np.std(data),
#                                                 global_mu = np.mean(data),
#                                                 sigma_bits = [14,14,14,14])
# print("global sigma", np.std(data))
# print("global mu", np.mean(data))
# with open('quantized_params.txt', 'w') as f:
#     for param in qw:
#         f.write(f"{param}\n")
# f.close()

# chunk_size = 4096
# num_chunks = (len(qw) + chunk_size - 1) // chunk_size

# Encrypted_params = []
# for i in range(num_chunks):
#     chunk = qw[i * chunk_size:(i + 1) * chunk_size]
#     encrypted_chunk = ts.bfv_vector(context, chunk)  # 使用 bfv_vector 而不是 ckks_vector
#     Encrypted_params.append(encrypted_chunk)

    
# Decrypted_params = []
# for i in range(num_chunks):
#     decrypted_chunk = Encrypted_params[i].decrypt()
#     Decrypted_params.extend(decrypted_chunk)
# qw = Decrypted_params

# # Dequantize the decrypted parameters
# dequantized_params = quantizer.dequantize_weights_unified(qw, params)

# with open("/hpc/home/connect.xmeng027/Work/FEHE/src/encrypted/q_decrypted_params_0.txt", 'w') as f:
#     for param in dequantized_params:
#         f.write(f"{param}\n")
# f.close()


# # Compare dequantized data with original data
# print("Quantization Method:", quant_method)
# mse_data = np.mean((np.array(data) - dequantized_params)**2)
# print(f"MSE (original vs dequantized): {mse_data}")
# mae_data = np.mean(np.abs(np.array(data) - dequantized_params))
# print(f"MAE (original vs dequantized): {mae_data}")
# print("Original data (first 10):", data[:10])
# print("Quantized data (first 10):", qw[:10])
# print("Decrypted data (first 10):", Decrypted_params[:10])
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

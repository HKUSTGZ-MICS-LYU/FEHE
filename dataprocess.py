import numpy as np
from sympy import nextprime
import tenseal as ts
from src.Quantization import Quantizer
import matplotlib.pyplot as plt

# Read data from file /hpc/home/connect.xmeng027/Work/FEHE/src/encrypted/unencrypt_params_0.txt
file_path = "/hpc/home/connect.xmeng027/Work/FEHE/src/encrypted/unencrypt_params_0.txt"
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(float(line.strip()))
    f.close()

# Choose quantization method: e.g., 'naive', 'truncate', 'layerwise', 'log', 'dynamic', 'symmetric', 'asymmetric', 'block'
quant_method = 'sigma'
q_bit = 8
quantizer = Quantizer()
qw, params = quantizer.quantize_weights_unified(data, q_bit, method=quant_method, sigma_bits = [16,16,16,16])

# Set BFV encryption parameters
poly_modulus_degree = 4096
plain_modulus = nextprime(2**16)
context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree, plain_modulus)
context.generate_galois_keys()
public_key = context.public_key()
secret_key = context.secret_key()

# Encrypt the each region of the quantized weights
encrypted_params = []
print(len(qw))
for region_data in qw:
    chunks = [region_data[i:i+poly_modulus_degree] for i in range(0, len(region_data), poly_modulus_degree)]
    encrypted_chunks = [ts.bfv_vector(context, chunk.tolist()) for chunk in chunks]
    encrypted_params.append(encrypted_chunks)
    
# Decrypt the encrypted parameters
decrypted_params = []
for region in encrypted_params:
    decrypted_chunks = [chunk.decrypt(secret_key) for chunk in region]
    decrypted_region = np.concatenate(decrypted_chunks)
    decrypted_params.append(decrypted_region)
    
# Dequantize the decrypted parameters
dequantized_params = quantizer.dequantize_weights_unified(decrypted_params, params)

with open("/hpc/home/connect.xmeng027/Work/FEHE/src/encrypted/q_decrypted_params_0.txt", 'w') as f:
    for param in dequantized_params:
        f.write(f"{param}\n")
f.close()


# Compare dequantized data with original data
mse_data = np.mean((np.array(data) - dequantized_params)**2)
print(f"MSE (original vs dequantized): {mse_data}")
mae_data = np.mean(np.abs(np.array(data) - dequantized_params))
print(f"MAE (original vs dequantized): {mae_data}")
print("Original data (first 10):", data[:10])
print("Dequantized data (first 10):", dequantized_params[:10])

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

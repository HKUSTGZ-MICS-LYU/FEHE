import warnings
import numpy as np
import matplotlib.pyplot as plt
from src.models import Net
import base64
import src.filedata as fd
import tenseal as ts

# 禁用警告
warnings.filterwarnings("ignore", category=UserWarning)

# 加载加密上下文
secret_ctx_bytes = fd.read_data("secret.txt")
secret_ctx = ts.context_from(secret_ctx_bytes[0])  # 私钥上下文

public_ctx_bytes = fd.read_data("public.txt")
public_ctx = ts.context_from(public_ctx_bytes[0])  # 公钥上下文

# 初始化模型并获取参数
model = Net()
params = model.state_dict()

# 展平模型参数
flattened_params = []
for param_name, param_tensor in params.items():
    flat_tensor = param_tensor.flatten()
    for val in flat_tensor:
        flattened_params.append(val.item())

print(f"Flattened params length: {len(flattened_params)}")

# 保存展平参数到文件（可选）
with open("flattened_params.txt", "w") as f:
    for val in flattened_params:
        f.write(str(val) + "\n")

# 分块参数
chunk_size = 4096
num_chunks = (len(flattened_params) + chunk_size - 1) // chunk_size  # 向上取整
print(f"Num chunks: {num_chunks}")

chunked_params = []
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(flattened_params))
    chunked_params.append(flattened_params[start_idx:end_idx])

# 加密每个块
encrypted_params = []
for chunk in chunked_params:
    ct = ts.ckks_vector(public_ctx, chunk)
    encrypted_params.append(ct.serialize())

# 修改后的 write_data 函数：Base64 编码并写入每行
def write_data(file_name: str, data):
    with open(file_name, 'wb') as f:
        for d in data:
            encoded = base64.b64encode(d)  # Base64 编码
            f.write(encoded + b'\n')       # 写入并添加换行符

write_data("encrypted_params.txt", encrypted_params)

# 解密参数
def read_data(file_name: str) -> list:
    data_list = []
    with open(file_name, 'rb') as f:
        for line in f:
            data = base64.b64decode(line.strip())  # 去除换行符并解码
            data_list.append(data)
    return data_list

ct_list = read_data("encrypted_params.txt")
decrypted_params = []
for ct in ct_list:
    encrypted_tensor = ts.lazy_ckks_vector_from(ct)
    encrypted_tensor.link_context(secret_ctx)
    decrypted_chunk = encrypted_tensor.decrypt()
    decrypted_params.extend(decrypted_chunk)  # 解密并展平
print(len(decrypted_params))

with open("decrypted_params.txt", "w") as f:
    for val in decrypted_params:
        f.write(str(val) + "\n")
f.close()
# 验证解密后的参数长度是否与原始参数一致
print(f"Original flattened params length: {len(flattened_params)}")
print(f"Decrypted params length: {len(decrypted_params)}")

if len(flattened_params) != len(decrypted_params):
    print("Error: The number of decrypted parameters does not match the original parameters.")
else:
    print("Success: The number of decrypted parameters matches the original parameters.")

# 计算误差指标
original_params = np.array(flattened_params)
decrypted_params = np.array(decrypted_params)

mse = np.mean((original_params - decrypted_params) ** 2)
mae = np.mean(np.abs(original_params - decrypted_params))
max_ae = np.max(np.abs(original_params - decrypted_params))

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Max Absolute Error: {max_ae}")

# 检查是否在容忍范围内
tolerance = 1e-3  # 根据需要调整
if np.allclose(original_params, decrypted_params, atol=tolerance):
    print("All parameters are similar within the specified tolerance.")
else:
    differences = np.abs(original_params - decrypted_params)
    max_diff = np.max(differences)
    print(f"Parameters differ by up to {max_diff}, which is above the tolerance of {tolerance}.")

# 可视化差异
plt.figure(figsize=(8, 8))
plt.scatter(original_params, decrypted_params, alpha=0.5)
plt.xlabel("Original Parameters")
plt.ylabel("Decrypted Parameters")
plt.title("Original vs Decrypted Parameters")
plt.plot([original_params.min(), original_params.max()],
         [original_params.min(), decrypted_params.max()], 'r--')  # 45度参考线
plt.savefig("parameter_comparison.png")

plt.figure(figsize=(10, 6))
differences = original_params - decrypted_params
plt.hist(differences, bins=100, alpha=0.75, color='blue')
plt.xlabel("Absolute Difference")
plt.ylabel("Frequency")
plt.title("Histogram of Parameter Differences")
plt.savefig("parameter_differences.png")

# 重建模型参数
model_reconstructed = Net()
params_reconstructed = model_reconstructed.state_dict()
idx = 0

for param_name, param_tensor in params_reconstructed.items():
    flat_tensor = param_tensor.flatten()
    for i in range(len(flat_tensor)):
        if idx < len(decrypted_params):
            flat_tensor[i] = decrypted_params[idx]
            idx += 1
        else:
            raise ValueError(f"Insufficient decrypted params for {param_name}")
    params_reconstructed[param_name] = flat_tensor.reshape(param_tensor.shape)

model_reconstructed.load_state_dict(params_reconstructed)
model_reconstructed.eval()


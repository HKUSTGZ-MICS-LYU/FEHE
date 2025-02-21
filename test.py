import tenseal as ts
import time
import numpy as np

# 创建TenSEAL上下文
context = ts.Context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=4096,  # 多项式模数次数
    plain_modulus=1032193,
    coeff_mod_bit_sizes=[32,32,32]
)
context.generate_galois_keys()

# 测试次数
num_tests = 100
encrypt_times = []
decrypt_times = []

for i in range(num_tests):
    # 生成随机向量
    random_vector = np.random.randint(0, 65536, 4096)
    
    # 测试加密时间
    start_time = time.time()
    encrypted_vector = ts.bfv_vector(context, random_vector)
    encrypt_times.append(time.time() - start_time)
    
    # 测试解密时间
    start_time = time.time()
    decrypted_vector = encrypted_vector.decrypt()
    decrypt_times.append(time.time() - start_time)
    
    # 验证结果正确性
    if not np.array_equal(random_vector, decrypted_vector):
        print(f"测试 {i+1} 验证失败!")

# 计算平均时间
avg_encrypt_time = np.mean(encrypt_times)
avg_decrypt_time = np.mean(decrypt_times)

print(f"\n{num_tests}次测试的平均加密时间: {avg_encrypt_time:.4f} 秒")
print(f"{num_tests}次测试的平均解密时间: {avg_decrypt_time:.4f} 秒")


# 计算总时间
print(f"\n{num_tests}次测试的总加密时间: {np.sum(encrypt_times):.4f} 秒")
print(f"{num_tests}次测试的总解密时间: {np.sum(decrypt_times):.4f} 秒")

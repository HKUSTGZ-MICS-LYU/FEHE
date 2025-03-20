import os
import sys
from sympy import nextprime
import tenseal as ts
import time
import numpy as np

# 创建四种不同参数的TenSEAL上下文
contexts = {
    4096: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096),
    # 8192: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192),
    # 16384: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384),
    # 32768: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768)
}

# 为每个上下文生成Galois密钥并设置全局scale
for context in contexts.values():
    context.generate_galois_keys()
    context.global_scale = 2**40

# 测试次数
num_tests = 1

# 存储每种框架的加密和解密时间
results = {size: {'encrypt_times': [], 'decrypt_times': []} for size in contexts.keys()}

for _ in range(num_tests):
    # 生成随机向量
    random_vector = np.random.uniform(0, 30, 4096)
    
    # 对每种框架进行测试
    for size, context in contexts.items():
        num_splits = 4096 // size
        encry_list = []
        
        # 加密
        start_time = time.time()
        for i in range(num_splits):
            vector_slice = random_vector[i*size:(i+1)*size]
            encrypted = ts.ckks_vector(context, vector_slice)
            encry_list.append(encrypted)
        encrypt_time = time.time() - start_time
        results[size]['encrypt_times'].append(encrypt_time)
        
        # 打印加密向量大小
        print(f"Size of Encryption Vector with slot {size}: {sys.getsizeof(encry_list[0])}")
        
        # 解密
        decry_list = []
        start_time = time.time()
        for encrypted in encry_list:
            decrypted = encrypted.decrypt()
            decry_list.append(decrypted)
        decrypt_time = time.time() - start_time
        results[size]['decrypt_times'].append(decrypt_time)
        
        # 合并解密结果
        decrypted_full = np.concatenate(decry_list)
        
        # 验证结果
        if not np.allclose(random_vector, decrypted_full, rtol=1e-1, atol=1e-1):
            print(f"Decryption failed for size {size}!")

# 打印平均时间
for size in contexts.keys():
    print(f"\nResults for {size} slots:")
    print(f"Average encryption time: {sum(results[size]['encrypt_times'])/num_tests:.4f} seconds")
    print(f"Average decryption time: {sum(results[size]['decrypt_times'])/num_tests:.4f} seconds")

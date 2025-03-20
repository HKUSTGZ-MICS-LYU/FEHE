import os
import sys
from sympy import nextprime
import tenseal as ts
import time
import numpy as np

# 创建四种不同参数的TenSEAL上下文
ckks_contexts = {
    4096: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=4096),
    8192: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192),
    16384: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384),
    32768: ts.Context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768)
}

# 为每个上下文生成Galois密钥并设置全局scale
for context in ckks_contexts.values():
    context.generate_galois_keys()
    context.global_scale = 2**40



# 存储每种框架的加密和解密时间
results = {size: {'encrypt_times': [], 'decrypt_times': []} for size in ckks_contexts.keys()}


# 生成随机向量
random_vector = np.random.uniform(0, 30, 16384*512)
for size, context in ckks_contexts.items():
    # check the size of the context and may be it need to split the vector
    for i in range(0, len(random_vector), size):
        # 加密
        start = time.time()
        enc = ts.ckks_vector(context, random_vector[i:i+size])
        end = time.time()
        results[size]['encrypt_times'].append(end - start)
        
        # 解密
        start = time.time()
        dec = enc.decrypt()
        end = time.time()
        results[size]['decrypt_times'].append(end - start)
        
        # 检查解密结果是否正确
        assert np.allclose(random_vector[i:i+size], dec, atol=1e-2)
            
# 打印结果
print("CKKS encryption and decryption times")
for size, result in results.items():
    print(f"Context size: {size}")
    print(f"Total encryption time: {sum(result['encrypt_times'])}s")
    print(f"Total decryption time: {sum(result['decrypt_times'])}s")
    print(f"Average encryption time: {np.mean(result['encrypt_times'])}s")
    print(f"Average decryption time: {np.mean(result['decrypt_times'])}s")
    print()
    
BFV_contexts = {
    4096: ts.Context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193),
    8192: ts.Context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193),
    # 16384: ts.Context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=16384, plain_modulus=1032193),
    # 32768: ts.Context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=32768, plain_modulus=1032193)
}

for context in BFV_contexts.values():
    context.global_scale = 2**40
    
results = {size: {'encrypt_times': [], 'decrypt_times': []} for size in BFV_contexts.keys()}



random_vector = np.random.randint(0, 2, 4096*11415)
for size, context in BFV_contexts.items():
    for i in range(0, len(random_vector), size):
        start = time.time()
        enc = ts.bfv_vector(context, random_vector[i:i+size])
        end = time.time()
        results[size]['encrypt_times'].append(end - start)
        
        start = time.time()
        dec = enc.decrypt()
        end = time.time()
        results[size]['decrypt_times'].append(end - start)
        
        assert np.allclose(random_vector[i:i+size], dec, atol=1e-2)
            
print("BFV encryption and decryption times")
for size, result in results.items():
    print(f"Context size: {size}")
    print(f"Total encryption time: {sum(result['encrypt_times'])}s")
    print(f"Total decryption time: {sum(result['decrypt_times'])}s")
    print(f"Average encryption time: {np.mean(result['encrypt_times'])}s")
    print(f"Average decryption time: {np.mean(result['decrypt_times'])}s")
    print()
    
    
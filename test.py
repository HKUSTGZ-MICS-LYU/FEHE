import os
import sys
from sympy import nextprime
import tenseal as ts
import time
import numpy as np

# 创建TenSEAL上下文
context1 = ts.Context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=4096,
    plain_modulus=65537,  # Must be prime and congruent to 1 modulo 2n
    coeff_mod_bit_sizes=[32,32,32]
)
context1.generate_galois_keys()
context2 = ts.Context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=65537,  # Must be prime and congruent to 1 modulo 2n
    coeff_mod_bit_sizes=[32,32,32,32]
)
context2.generate_galois_keys()
context3 = ts.Context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=16384,
    plain_modulus=65537,  # Must be prime and congruent to 1 modulo 2n
    coeff_mod_bit_sizes=[32,32,32,32,32]
)
context3.generate_galois_keys()

# 测试次数
num_tests = 1
encrypt_times1 = []
decrypt_times1 = []
encrypt_times2 = []
decrypt_times2 = []
encrypt_times3 = []
decrypt_times3 = []

for i in range(num_tests):
    # 生成随机向量
    random_vector = np.random.randint(0, 30, 4096*4)
    
    # 测试加密时间
    start_time = time.time()
    encrypted_vector = ts.bfv_vector(context3, random_vector)
    encrypt_times3.append(time.time() - start_time)
    print(f"Size of Encryption Vector with slot 16394 : {sys.getsizeof(encrypted_vector)}")
    
    # 测试解密时间
    start_time = time.time()
    decrypted_vector = encrypted_vector.decrypt()
    decrypt_times3.append(time.time() - start_time)
    
    if random_vector.tolist() != decrypted_vector:
        print("Decryption failed!")
        print(random_vector)
        print(decrypted_vector)
        
    encry_list = []
    start_time = time.time()
    for i in range(2):
        encry_list.append(ts.bfv_vector(context2, random_vector[i*8192:(i+1)*8192]))
    encrypt_times2.append(time.time() - start_time)
    print("Size of Encryption Vector with slot 8192 : ", sys.getsizeof(encry_list[0]))
    
    decry_list = []
    start_time = time.time()
    for i in range(2):
        decry_list.append(encry_list[i].decrypt())
    decrypt_times2.append(time.time() - start_time)
    decry_list = decry_list[0] + decry_list[1]
    
    if random_vector.tolist() != decry_list:
        print("Decryption failed!")
        print(random_vector)
        print(decry_list)
    
    encry_list = []
    start_time = time.time()
    for i in range(4):
        encry_list.append(ts.bfv_vector(context1, random_vector[i*4096:(i+1)*4096]))
    encrypt_times1.append(time.time() - start_time)
    print("Size of Encryption Vector with slot 4096 : ", sys.getsizeof(encry_list[0]))
    
    decry_list = []
    start_time = time.time()
    for i in range(4):
        decry_list.append(encry_list[i].decrypt())
    decrypt_times1.append(time.time() - start_time)
    
    decry_list = decry_list[0] + decry_list[1] + decry_list[2] + decry_list[3]
    
    if random_vector.tolist() != decry_list:
        print("Decryption failed!")
        print(random_vector)
        print(decry_list)
        
print("Average decryption time with 4096 slots: ", sum(decrypt_times1)/num_tests)
print("Average encryption time with 8192 slots: ", sum(encrypt_times2)/num_tests)
print("Average encryption time with 16384 slots: ", sum(encrypt_times3)/num_tests)
        
        
    

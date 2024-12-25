import tenseal as ts
import numpy as np

# 创建 CKKS 上下文
poly_modulus_degree = 8192
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=poly_modulus_degree,
    coeff_mod_bit_sizes=[60, 40,  40, 60]  # 增加模数链
)
context.global_scale = 2**40  # 提高缩放因子
context.generate_galois_keys()

# 创建两个随机矩阵并归一化
matrix1 = np.random.rand(2, 3, 3)  # 更大的范围测试
matrix2 = np.random.rand(2, 3, 3) 


# 打印原始矩阵
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)

# 加密矩阵
ct_matrix1 = ts.ckks_tensor(context, matrix1)
ct_matrix2 = ts.ckks_tensor(context, matrix2)

# 同态逐元素乘法
ct_product = ct_matrix1 * ct_matrix2

# 明文逐元素乘法
product = matrix1 * matrix2

# 解密结果
decrypted_product = ct_product.decrypt()

# 打印结果和误差
print("Original Product:\n", product)
print("Decrypted Product:\n", decrypted_product.raw)

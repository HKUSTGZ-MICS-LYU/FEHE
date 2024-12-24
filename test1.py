import base64
import numpy as np
import tenseal as ts
import src.filedata as fd  # Replace with your own file read/write methods

#####################################
# 1) 创建上下文并保存（含私钥和公钥）
#####################################
print("\n1) Creating new TenSEAL CKKS context with private and public keys")

# 1.1 创建带私钥的上下文
# 这里使用了 poly_modulus_degree=4096 和 coeff_mod_bit_sizes=[40, 20, 40] 作为演示
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=4096,
    coeff_mod_bit_sizes=[40, 20, 40]
)
context.generate_galois_keys()       # 生成 Galois keys（旋转等操作需要）
context.global_scale = 2**40

# 1.2 保存完整上下文（含私钥）到 secret.txt
print("   Saving secret context -> 'secret.txt'")
secret_ctx_bytes = context.serialize(save_secret_key=True)  # 序列化包含私钥
fd.write_data("secret.txt", [secret_ctx_bytes])



# 1.3 保存仅包含公钥的上下文到 public.txt
print("   Saving public context -> 'public.txt'")
context.make_context_public()  # 使 context 不再包含私钥
public_ctx_bytes = context.serialize()
fd.write_data("public.txt", [public_ctx_bytes])


##############################################
# 2) 读取公钥上下文，加密随机向量，保存密文
##############################################
print("\n2) Reading public context, encrypting random vectors, saving ciphertext files")

# 2.1 读取 public.txt，得到公钥上下文
public_ctx_bytes = fd.read_data("public.txt")
public_ctx = ts.context_from(public_ctx_bytes[0])

# 2.2 生成随机向量，并用公钥上下文加密
pt0 = np.random.randint(0, 10000, 4096)  # 示例随机向量1
pt1 = np.random.randint(0, 10000, 4096)  # 示例随机向量2
print("   Plaintext vectors generated.")

# 使用公钥上下文加密
ct0 = ts.ckks_tensor(public_ctx, ts.plain_tensor(pt0))
ct1 = ts.ckks_tensor(public_ctx, ts.plain_tensor(pt1))
print("   Encrypted tensors created.")

ct_list = [ct0, ct1]  # 放到列表中，方便保存和后续处理

# 2.3 保存密文到 encrypted_tensor.txt（逐行 Base64）
print("\n3) Saving the encrypted tensor to a file -> 'encrypted_tensor.txt'")
for i in range(len(ct_list)):
    ct_list[i] = ct_list[i].serialize()
fd.write_data("encrypted_tensor.txt", ct_list)
print("   Encrypted tensors saved.")

# 删除内存中的密文，模拟真实场景（可选）
del ct0, ct1, ct_list

###########################################
# 3) 使用私钥上下文加载并解密
###########################################
print("\n4) Loading the encrypted tensor from a file, then decrypting")

# 3.1 读取 secret.txt 中包含私钥的上下文
print("   Reading secret context from 'secret.txt'")
secret_ctx_bytes = fd.read_data("secret.txt")
secret_ctx = ts.context_from(secret_ctx_bytes[0])  # 私钥上下文

# 3.2 逐行读取 encrypted_tensor.txt，Base64 解码后加载
print("   Reading encrypted tensor from 'encrypted_tensor.txt'")
ct_loaded = []
ct_list = fd.read_data("encrypted_tensor.txt")
for i in range(len(ct_list)):
    encrypted_tensor = ts.lazy_ckks_tensor_from(ct_list[i])
    encrypted_tensor.link_context(secret_ctx)  # 链接私钥上下文
    ct_loaded.append(encrypted_tensor)
    
# 3.3 解密并验证结果
print("\n5) Decrypting the tensors")
pt_dec = []
for item in ct_loaded:
    # .decrypt() 返回的是一个 CKKSTensor
    # .raw 可以拿到内部的 numpy 数组
    pt_dec.append(item.decrypt().raw)

# 验证解密后的前 10 个元素是否与原始向量一致
print("   First 10 elements of decrypted 1st tensor:", pt_dec[0][:10])
print("   First 10 elements of original 1st vector:", pt0[:10])

print("\nDone.")
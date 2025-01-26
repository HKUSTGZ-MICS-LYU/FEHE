import tenseal as ts
import random
import time
# Create TenSEAL context with BFV scheme
poly_modulus_degree = 1024
plain_modulus = 786433
coeff_mod_bit_sizes = [30, 30, 30]



context = ts.context(
    ts.SCHEME_TYPE.BFV, 
    poly_modulus_degree=poly_modulus_degree,
    plain_modulus=plain_modulus,
    coeff_mod_bit_sizes=coeff_mod_bit_sizes
)

context.global_scale = 2**20
context.generate_galois_keys()






# Create a random vector of length 4096 with values in [-plaintext_modulus/2, plaintext_modulus/2]
input_x = [random.randint(-(plain_modulus//2), plain_modulus//2) for _ in range(4096)]


# Encrypt the vector
encrypted_time = time.time()
# x = ts.plain_tensor(input_x)

x = ts.bfv_vector(context, input_x)
encrypted_time = time.time() - encrypted_time
print(f"Encryption time: {encrypted_time}")
# Decrypt the vector
x = x.decrypt()

# Print the decrypted vector
print(input_x == x)



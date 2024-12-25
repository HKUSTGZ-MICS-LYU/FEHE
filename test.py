import numpy as np
import tenseal as ts
import src.filedata as fd  # Replace with your own file read/write methods


degree = 4096
# ---------------------------------------------------------------
# 1) Create a CKKS context with a private key and save to files
# ---------------------------------------------------------------
def create_and_save_context():
    print("\n1) Creating new TenSEAL CKKS context with private and public keys")
    # Create full context (including private key)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=degree,
        coeff_mod_bit_sizes=[40, 20, 40]
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** 40

    # Save the full context (with private key) to "secret.txt"
    print("   Saving secret context -> 'secret.txt'")
    secret_ctx_bytes = context.serialize(save_secret_key=True)
    fd.write_data("secret.txt", [secret_ctx_bytes])

    # Convert existing context to "public only" by dropping private key
    print("   Saving public context -> 'public.txt'")
    context.make_context_public()
    public_ctx_bytes = context.serialize()
    fd.write_data("public.txt", [public_ctx_bytes])

    print("   Done creating/saving context.\n")


# ---------------------------------------------------------------
# 2) Read the public context, encrypt two random plaintexts
# ---------------------------------------------------------------
def encrypt_and_save_tensors():
    print("\n2) Reading public context, encrypting random vectors, saving ciphertext files")
    # Load the public-only context
    public_ctx_bytes = fd.read_data("public.txt")
    public_ctx = ts.context_from(public_ctx_bytes[0])

    # Create random plaintexts
    pt1 = np.random.randint(0, 10000, degree)
    pt2 = np.random.randint(0, 10000, degree)

  
    print("   Plaintext vectors generated.")

    # Convert to TenSEAL plain tensors
    plain1 = ts.plain_tensor(pt1)
    plain2 = ts.plain_tensor(pt2)

    # Encrypt using the public context
    enc1 = ts.ckks_tensor(public_ctx, plain1)
    enc2 = ts.ckks_tensor(public_ctx, plain2)

    # Save ciphertext to file
    fd.write_data("encrypted_tensor1.txt", [enc1.serialize()])
    fd.write_data("encrypted_tensor2.txt", [enc2.serialize()])

    print("   Wrote 'encrypted_tensor1.txt' and 'encrypted_tensor2.txt'")

    # For verification, return the random plaintext
    return pt1, pt2


# ---------------------------------------------------------------
# 3) Read the ciphertexts, link them to the public context, add
# ---------------------------------------------------------------
def add_ciphertexts_on_server():
    print("\n3) Reading ciphertext from files, linking to public context, summing")

    # Load the public context for homomorphic ops
    public_ctx_bytes = fd.read_data("public.txt")
    public_ctx = ts.context_from(public_ctx_bytes[0])

    # Read the ciphertext data from file
    ciphertext_bytes_1 = fd.read_data("encrypted_tensor1.txt")[0]
    ciphertext_bytes_2 = fd.read_data("encrypted_tensor2.txt")[0]

    # Build lazy tensors
    enc_tens_1 = ts.lazy_ckks_tensor_from(ciphertext_bytes_1)
    enc_tens_2 = ts.lazy_ckks_tensor_from(ciphertext_bytes_2)

    # Link them to the same public context
    enc_tens_1.link_context(public_ctx)
    enc_tens_2.link_context(public_ctx)

    # Add
    enc_sum = enc_tens_1 + enc_tens_2

    # Save the sum to file
    fd.write_data("encrypted_tensor_sum.txt", [enc_sum.serialize()])
    print("   Summed ciphertext saved -> 'encrypted_tensor_sum.txt'")

    # Return nothing, just demonstrates "server" combining
    pass


# ---------------------------------------------------------------
# 4) Read the sum, link to secret context, decrypt, verify
# ---------------------------------------------------------------
def decrypt_and_verify(pt1, pt2):
    print("\n4) Reading sum ciphertext, linking to secret context, verifying result")

    # Load the public context if you need it for further ops (only if actually needed)
    # public_key_context = ts.context_from(fd.read_data("public.txt"))

    # Load the secret context (with private key)
    secret_ctx_bytes = fd.read_data("secret.txt")[0]
    secret_ctx = ts.context_from(secret_ctx_bytes)

    # Read the sum ciphertext from file
    ciphertext_sum_bytes = fd.read_data("encrypted_tensor_sum.txt")[0]
    enc_sum_lazy = ts.lazy_ckks_tensor_from(ciphertext_sum_bytes)

    # Link the sum ciphertext to the *secret context*,
    # so TenSEAL knows how to decrypt
    enc_sum_lazy.link_context(secret_ctx)

    # Decrypt with no arguments
    decrypted_np = enc_sum_lazy.decrypt()
    decrypted_np = np.array(decrypted_np.raw)/2
    with open("decrypted.txt", "w") as f:
        for item in decrypted_np:
            f.write("%s\n" % item)
    f.close()
    print("   Decrypted result saved -> 'decrypted.txt'")
    
    
    expected = (pt1 + pt2 ) / 2

    print("   First 10 elements of decrypted:", decrypted_np[:10])
    print("   First 10 elements of expected:", expected[:10])

    assert np.allclose(decrypted_np, expected, atol=1e-3)


# ---------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Create context and save to files
    create_and_save_context()

    # Step 2: Load public context, encrypt random vectors, store ciphertext
    pt1, pt2 = encrypt_and_save_tensors()

    # Step 3: "Server" side add
    add_ciphertexts_on_server()

    # Step 4: "Client" side load sum, decrypt, verify
    decrypt_and_verify(pt1, pt2)
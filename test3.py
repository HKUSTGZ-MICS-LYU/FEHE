import src.filedata as fd
import numpy as np
import tenseal as ts

# Load public key to perform computations on encrypted data
public_key_context = ts.context_from(fd.read_data("src/encrypted/public_key.txt")[0])
secret_key_context = ts.context_from(fd.read_data("src/encrypted/secret_key.txt")[0])

aggregated_parameters = []
results_ex = None
num = 0

encrypted_weight_pth = 'src/encrypted/data_encrypted_0.txt'
encrypted_proto_list = fd.read_data(encrypted_weight_pth)


print(len(encrypted_proto_list))

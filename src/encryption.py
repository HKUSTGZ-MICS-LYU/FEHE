#Importing necessary libraries
import os
import tenseal as ts
import filedata as fd
import numpy as np
from enum import Enum
import sys
from Quantization import *


Quantization_Bits = 8

#This block decides if the Federated Learning setup undergoes encryption and decryption during communication
class Enc_needed(Enum):
    # If encryption_needed is 0, then FL without FHE
    # If encryption_needed is 1, then FL with FHE
    encryption_needed = 1
        
 
        
def create_context():                                       #Declaration of context to generate keys 
    global context
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=4096,
        plain_modulus=1032193        
    )
    
    #generating public key and private key pair
    context.generate_galois_keys()
        
    #generting secret key and saving it in a text file
    secret_key_context = context.serialize(save_secret_key = True)
    private_key_file = "encrypted/secret_key.txt"
    fd.write_data(private_key_file, [secret_key_context])
        
    #generating public key and saving it in a text file
    context.make_context_public()                           #drops the private key
    public_key_context = context.serialize()
    public_key_file = "encrypted/public_key.txt"
    fd.write_data(public_key_file, [public_key_context])

def param_encrypt(param_list, clientID: str, global_max, global_min):                
    
    # Checking if the public key and secret key files exist
    # If not, create the context and generate the keys
    if os.path.exists("encrypted/public_key.txt") and os.path.exists("encrypted/secret_key.txt"):
        pass
    else:        
        create_context()
    
    #Loading public key for encryption
    public_key_context = ts.context_from(fd.read_data("encrypted/public_key.txt")[0])    
    
    flattened_params = []
    for param_name, param_tensor in param_list.items():
        flat_tensor = param_tensor.flatten()
        for val in flat_tensor:
            flattened_params.append(val.item())

    # Write unencrypted params into a file
    with open(f"encrypted/unencrypt_params_{clientID}.txt", 'w') as f:
        for param in flattened_params:
            f.write(f"{param}\n")
    f.close()
    
    chunk_size = 4096
    quantizer = Quantizer()
    if global_max is None or global_min is None:
        global_max = max(flattened_params)
        global_min = min(flattened_params)
    qw, params = quantizer.quantize_weights_unified(
                        flattened_params, 
                        Quantization_Bits, 
                        'sigma', 
                        global_max = global_max, 
                        global_min = global_min, 
                        block_size = 32, 
                        sigma_bits = [8,8,8,8]
                        )
    
    # Write quantized weights to file
    with open(f"encrypted/quantized_params_{clientID}.txt", 'w') as f:
        for param in qw:
            f.write(f"{param}\n")
    f.close()
     
    num_chunks = (len(qw) + chunk_size - 1) // chunk_size  
    
    chunked_params = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(qw))
        chunked_params.append(qw[start_idx:end_idx])

 
    encrypted_params = []
    for chunk in chunked_params:
        ct = ts.bfv_vector(public_key_context, chunk)
        encrypted_params.append(ct.serialize())
        
    #Writing the encrypted data to a text file
    encrypted_params_pth = "encrypted/data_encrypted_" + str(clientID) + ".txt"

    #Writing the encrypted data list to a text file
    fd.write_data(encrypted_params_pth, encrypted_params)

    serialized_dataspace = sys.getsizeof(encrypted_params)/(1024*1024)
    
    return  None, serialized_dataspace, params

def param_decrypt(encrypted_weight_pth, params):                                        #Function to implement decryption
    
    #Loading secret key to decrypted the encrypted data
    secret_context = ts.context_from(fd.read_data('encrypted/secret_key.txt')[0])
    
    #Selecting the text file that stores aggregation results for decryption  
    encrypted_params = fd.read_data(encrypted_weight_pth)

    decrypted_params = []
    for ct in encrypted_params:
        ct = ts.bfv_vector_from(secret_context, ct)
        ct.link_context(secret_context)
        decrypted_chunk = ct.decrypt()
        decrypted_params.extend(decrypted_chunk)

    # Write decrypted parameters to a file
    with open("encrypted/decrypted.txt", "w") as f:
        for param in decrypted_params:
            f.write(f"{param}\n")
    f.close()
    
    decrypted_params = [decrypted_params[i] / 20 for i in range(len(decrypted_params))]    
  
    quantizer = Quantizer()
    decrypted_params = quantizer.dequantize_weights_unified(decrypted_params, params)
    
    # Write decrypted parameters to a file
    with open("encrypted/deq.txt", "w") as f:
        for param in decrypted_params:
            f.write(f"{param}\n")
    f.close()

    return decrypted_params
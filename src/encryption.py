#Importing necessary libraries
import os
import tenseal as ts
import filedata as fd
import numpy as np
from enum import Enum
import sys

#This block decides if the Federated Learning setup undergoes encryption and decryption during communication
class Enc_needed(Enum):
    encryption_needed = 1 # data encryption and decryption is not necessary and thus regular Federated Learning is carried out
    # encryption_needed = 1 # data encryption and decryption is necessary -> Full encryption
    

        
def create_context():                                       #Declaration of context to generate keys 
    global context
    context = ts.context(
    ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = 4096,
        coeff_mod_bit_sizes = [40, 20, 40]
    )
    
    #generating public key and private key pair
    context.generate_galois_keys()
    context.global_scale = 2**40
        
    #generting secret key and saving it in a text file
    secret_key_context = context.serialize(save_secret_key = True)
    private_key_file = "encrypted/secret_key.txt"
    fd.write_data(private_key_file, [secret_key_context])
        
    #generating public key and saving it in a text file
    context.make_context_public()                           #drops the private key
    public_key_context = context.serialize()
    public_key_file = "encrypted/public_key.txt"
    fd.write_data(public_key_file, [public_key_context])

def param_encrypt(param_list, clientID: str):              
    
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
    print("Length of flattened_params: ", len(flattened_params))

    # Splitting the data into slices of 4096 elements
    chunk_size = 4096
    num_chunks = (len(flattened_params) + chunk_size - 1) // chunk_size  # 向上取整
    chunked_params = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(flattened_params))
        chunked_params.append(flattened_params[start_idx:end_idx])

    # Encrypting the data
    encrypted_params = []
    for chunk in chunked_params:
        ct = ts.ckks_vector(public_key_context, chunk)
        encrypted_params.append(ct.serialize())
        
    #Writing the encrypted data to a text file
    encrypted_params_pth = "encrypted/data_encrypted_" + str(clientID) + ".txt"

    #Writing the encrypted data list to a text file
    fd.write_data(encrypted_params_pth, encrypted_params)

        
    serialized_dataspace = sys.getsizeof(encrypted_params)/(1024*1024)
    
    return  None, serialized_dataspace

def param_decrypt(encrypted_weight_pth):                                        #Function to implement decryption
    
    #Loading secret key to decrypted the encrypted data
    secret_context = ts.context_from(fd.read_data('encrypted/secret_key.txt')[0])
    
    #Selecting the text file that stores aggregation results for decryption  
    encrypted_params = fd.read_data(encrypted_weight_pth)

    decrypted_params = []
    for ct in encrypted_params:
        ct = ts.ckks_vector_from(secret_context, ct)
        ct.link_context(secret_context)
        decrypted_chunk = ct.decrypt()
        decrypted_params.extend(decrypted_chunk)
        
    return decrypted_params
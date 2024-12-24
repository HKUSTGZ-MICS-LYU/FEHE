#Importing necessary libraries
import os
import tenseal as ts
import filedata as fd
import numpy as np
from enum import Enum
import sys

#This block decides if the Federated Learning setup undergoes encryption and decryption during communication
class Enc_needed(Enum):
    # encryption_needed = 0 # data encryption and decryption is not necessary and thus regular Federated Learning is carried out
    encryption_needed = 1 # data encryption and decryption is necessary -> Full encryption
    

        
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
    

    concatenated_data = np.concatenate([np.array(arr).flatten() for arr in param_list])
    slice_size = 4096
    num_slices = max(1, len(concatenated_data) // slice_size)  # 确保至少有一个切片
    sliced_data = np.array_split(concatenated_data, num_slices)
    
    
    encrypted_data_list = [
        ts.ckks_tensor(public_key_context, ts.plain_tensor(slice_)).serialize()
        for slice_ in sliced_data
    ]


    #Creating a text file considering client ID and encryption depth selected
    if Enc_needed.encryption_needed.value == 0: #  No encryption
        encrypted_data_file_path = "encrypted/data_encrypted_" + str(clientID) + ".txt"
    elif Enc_needed.encryption_needed.value == 1: # Full encryption
        encrypted_data_file_path = "encrypted/data_encrypted_" + str(clientID) + ".txt"

    #Writing the encrypted data list to a text file
    fd.write_data(encrypted_data_file_path, encrypted_data_list)

        
    serialized_dataspace = sys.getsizeof(encrypted_data_list)/(1024*1024)
    
    return  None, serialized_dataspace

def param_decrypt(encrypted_weight_pth):                                        #Function to implement decryption
    
    #Loading secret key to decrypted the encrypted data
    secret_context = ts.context_from(fd.read_data('encrypted/secret_key.txt')[0])
    
    #Selecting the text file that stores aggregation results for decryption  
    new_results_proto = fd.read_data(encrypted_weight_pth)

    new_results = []
    for new_result_proto in new_results_proto:
        new_result = ts.lazy_ckks_tensor_from(new_result_proto)
        new_result.link_context(secret_context)
        new_result = np.array(new_result.decrypt())
        new_result = new_result.tolist()
        new_results.append(new_result.raw)
    # flatten the list
    new_results = [item for sublist in new_results for item in sublist]

    #Returning the decrypted data in the form of a list
    return new_results
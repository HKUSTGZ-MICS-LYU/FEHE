#Importing necessary libraries
import os
import tenseal as ts
import filedata as fd
import numpy as np
from enum import Enum
import sys
from quantization import *



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
        
    return context

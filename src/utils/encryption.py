#Importing necessary libraries
import os
import tenseal as ts
import utils.filedata as fd
import numpy as np
from enum import Enum
import sys



          
def create_context(poly_modulus_degree: int = 4096, plain_modulus: int = 1032193):                                       #Declaration of context to generate keys 
    global context
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_modulus_degree,
        plain_modulus=plain_modulus        
    )

    #generating public key and private key pair
    context.generate_galois_keys()
    return context

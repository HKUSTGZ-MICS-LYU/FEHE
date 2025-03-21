from math import log2
import math
import ast
import numpy as np
from sympy import primitive_root
from utils import *

def poly_mult(poly1, poly2, q):
    """
    Polynomial multiplication on the ring x^N + 1
    """
    n = len(poly1)
    temp = [0] * (2 * n)
    for i in range(n):
        for j in range(n):
            temp[i+j] += (poly1[i] * poly2[j]) % q
    res = [0] * n
    for i in range(n):
        res[i] = (temp[i] - temp[i+n]) % q
    return np.array(res)

np.set_printoptions(threshold=np.inf)

def ntt(a, p, w, PENum, n):
    """
    Performs the Number Theoretic Transform (NTT).
    a: Input data
    p: Prime number
    w: Twiddle factors
    PENum: Number of processing elements
    n: Polynomial degree
    check_stage: Stage to print debug information (optional)
    """
    
    RAMNum = int(math.sqrt(n))
    stage = int(log2(n))
     
    check_stage = None
    
    # Store the input data in the RAM
    a = np.array(a).reshape(RAMNum, RAMNum)
    a = roll(a, RAMNum)
    # print("Input Polynomials: ")
    # print(a)
    
    # Store the twidle factors in the RAM
    index = generate_twidle_indices(n)
    tf = permute_twidle_factors(w, index, PENum)
    # print(tf)
    
    for i in range(stage): # stage count

        for j in range(RAMNum): # address count
            temp = []
            for k in range(RAMNum): #column count
                if i < stage//2:
                    temp.append(a[(k-j)%(2**(stage//2-i))+(j>>(stage//2-i))*int(RAMNum//2**i)][k])
                else:
                    temp.append(a[j][k])
            
 
            # generate the input index
            input_index = generate_input_index(i, RAMNum, j)
            # generate the output index
            output_index = generate_output_index(i, RAMNum, j)
       
   
            port = []   
            for m in range(RAMNum):
                port.append(temp[input_index[m]])
            
        
            # calculate the butterfly
            result = []
            for m in range(PENum):
                
                if i < stage//2:    
                    input1 = port[int(RAMNum/PENum)*m]
                    input2 = port[int(RAMNum/PENum)*m+1]
                    inputtf = tf[2**i- 1 + (j>>(stage//2-i)) + (m>>(stage-i-1))][m]
                    print(2**i- 1 + (j>>(stage//2-i)) + (m>>(stage-i-1)))
                    result.append(Butterfly(input1, input2, inputtf, p)[0])
                    result.append(Butterfly(input1, input2, inputtf, p)[1])
                   
                else:
                    input1 = port[int(RAMNum/PENum)*m]
                    input2 = port[int(RAMNum/PENum)*m+1]
                    inputtf = tf[2**i - 1 + (j << (i - stage//2)) + (m>>(stage-i-1))][m]
                    print(2**i - 1 + (j << (i - stage//2)) + (m>>(stage-i-1)))
                    result.append(Butterfly(input1, input2, inputtf, p)[0])
                    result.append(Butterfly(input1, input2, inputtf, p)[1]) 
      

            for k in range(RAMNum):
                # print("Output index: ", output_index[k])
                # print("Write Address: ", (k-j)%(2**(stage//2-i))+(j>>(3-i))*int(RAMNum//2**i))
                if i < stage//2:
                    a[(k-j)%(2**(stage//2-i))+(j>>(stage//2-i))*int(RAMNum//2**i)][k] = result[output_index[k]]
                else:
                    a[j][k] = result[output_index[k]]

  
    return a

def intt(a, p, w, PENum, n):
    """
    Performs the Inverse Number Theoretic Transform (INTT).
    a: Input data
    p: Prime number
    w: Twiddle factors
    PENum: Number of processing elements
    n: Polynomial degree
    check_stage: Stage to print debug information (optional)
    """
    RAMNum = int(math.sqrt(n))
    stage = int(log2(n))
    check_stage = None
    # Store the input data in the RAM
    if type(a) == np.ndarray:
        pass
    else:
        a = np.array(a).reshape(RAMNum, RAMNum)
        a = roll(a, RAMNum)
    # print("Input Polynomials: ")
    # print(a)
    # Store the twidle factors in the RAM
    index = generate_twidle_indices(n)
    tf = permute_twidle_factors(w, index, PENum)

    for i in range(stage):
        for j in range(RAMNum):
            temp = []
            for k in range(RAMNum):
                if i < stage//2:
                    temp.append(a[j][k])
                else:
                    temp.append(a[(k-j)%(2**(stage//2-(stage-i-1)))+(j>>(stage//2-(stage-i-1)))*int(RAMNum//2**(stage-i-1))][k])
         
                
            # generate the input index
            input_index = generate_input_index((stage-i-1), RAMNum, j)
            # generate the output index
            output_index = generate_output_index((stage-i-1), RAMNum, j)
            
            port = []
            for m in range(RAMNum):
                port.append(temp[input_index[m]])
     
            # calculate the butterfly
            result = []
            for m in range(PENum):
                if i < stage//2:
                    input1 = port[int(RAMNum/PENum)*m]
                    input2 = port[int(RAMNum/PENum)*m+1]
                    inputtf = tf[2**(stage-1-i) -1 + (j * (2**((stage-i-1) - stage//2))) + (m>>(stage-(stage-1-i)-1))][m]
                    # print(2**(stage-1-i) -1 + (j * (2**((stage-i-1) - stage//2))) + (m>>(stage-(stage-1-i)-1)))
                    result.append(Butterfly(input1, input2, inputtf, p, inverse=True)[0])
                    result.append(Butterfly(input1, input2, inputtf, p, inverse=True)[1])
                   
                else:
                    input1 = port[int(RAMNum/PENum)*m]
                    input2 = port[int(RAMNum/PENum)*m+1]
                    inputtf = tf[2**(stage-i-1)+(j>>(stage//2-(stage-i-1)))-1][m]
                    result.append(Butterfly(input1, input2, inputtf, p, inverse=True)[0])
                    result.append(Butterfly(input1, input2, inputtf, p, inverse=True)[1])
                   
    
            for k in range(RAMNum):
                if i < stage//2:
                    a[j][k] = result[output_index[k]]
                else:
                    a[(k-j)%(2**(stage//2-(stage-i-1)))+(j>>(stage//2-(stage-i-1)))*int(RAMNum//2**(stage-i-1))][k] = result[output_index[k]]

    
    res = []        
    for i in range(RAMNum):
        a[i] = np.roll(a[i], -i)
        res += list(a[i])
    
    # for i in range(n):
    #     res[i] = res[i] * pow(n, p - 2, p) % p

    return res


if __name__ == "__main__":


    p = 4294967377
    RAMNum = 4
    RAMDepth = RAMNum
    PENum = int(RAMNum/2)
    n = RAMNum * RAMDepth

    # a = [-2147385344, -201585659, -1, -67894260]
    a = list(range(n))
    ntt_w = generate_twidle_factors(n, p)
    print(ntt_w)


    a = ntt(a, p, ntt_w, PENum, n)
    print(a)
    intt_w = generate_twidle_factors(n, p, inverse=True)
    print(intt_w)
    NTT_res = intt(a, p, intt_w, PENum, n)
    print(NTT_res)

    
    
    
    

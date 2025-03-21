import numpy as np
import random
from utils import *
from NTT import *
import sympy



def sample(num, q, n):
    coeffs = [0] * n
    for i in range(n):
        coeffs[i] = random.randint(0, num-1)
    return np.array(coeffs)

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

def poly_add(poly1, poly2, q):
    """
    Polynomial addition on the ring x^N + 1
    """
    n = len(poly1)
    res = [0] * n
    for i in range(n):
        res[i] = (poly1[i] + poly2[i]) % q
    return np.array(res)
print("================================")
print("Parameters")
print("================================")
N = 16
q = 65537
poly = list(range(N))
ntt_w = generate_twidle_factors(N, q)
intt_w = generate_twidle_factors(N, q, inverse=True)
poly_ntt = ntt(poly, q, ntt_w, 2, N)
poly_intt = intt(poly_ntt, q, intt_w, 2, N)
print("Original polynomial: ", poly)
# t = 37
# delta = q//t
# m = [random.randint(0, t-1) for _ in range(N)]
# ntt_w = generate_twidle_factors(N, q)
# intt_w = generate_twidle_factors(N, q, inverse=True)
# print("Original message: ", m)

# print("================================")
# print("Generate public and secret keys")
# print("================================")
# a = sample(q, q, N)
# s = sample(2, q, N)
# e = sample(2, q, N)

# pk = (-1 * (poly_add(poly_mult(a, s, q), e, q)), a)
# sk = s
# # print("pk: ", pk)
# # print("sk: ", sk)

# print("================================")
# print("Encrypt")
# print("================================")
# e1 = sample(2, q, N)
# e2 = sample(2, q, N)
# u = sample(2, q, N)

# m = np.array(m) * delta % q
# temp1 = poly_mult(pk[0], u, q)
# temp2 = poly_add(temp1, e1, q)
# c0 = poly_add(temp2, m, q)

# temp1 = poly_mult(pk[1], u, q)
# c1 = poly_add(temp1, e2, q)
# print("c0: ", c0)
# print("c1: ", c1)

# shape = ntt(m, q, ntt_w, 2, N).shape
# ntt_m = ntt(m, q, ntt_w, 2, N).flatten()
# ntt_e1 = ntt(e1, q, ntt_w, 1, N).flatten()
# ntt_pk0 = ntt(pk[0], q, ntt_w, 1, N).flatten()
# ntt_u = ntt(u, q, ntt_w, 1, N).flatten()



# temp1 = [0] * N
# for i in range(N):
#     temp1[i] = (ntt_pk0[i] * ntt_u[i]) % q
# for i in range(N):
#     temp1[i] = (temp1[i] + ntt_e1[i]) % q
# for i in range(N):
#     temp1[i] = (temp1[i] + ntt_m[i]) % q
# temp1 = np.array(temp1).reshape(shape)
# intt_c0 = intt(temp1, q, intt_w, 1, N)

# print("c0: ", intt_c0)
# ntt_e2 = ntt(e2, q, ntt_w, 1, N).flatten()
# ntt_pk1 = ntt(pk[1], q, ntt_w, 1, N).flatten()

# temp1 = [0] * N
# for i in range(N):
#     temp1[i] = (ntt_pk1[i] * ntt_u[i]) % q
# for i in range(N):
#     temp1[i] = (temp1[i] + ntt_e2[i]) % q
# temp1 = np.array(temp1).reshape(shape)
# intt_c1 = intt(temp1, q, intt_w, 1, N)
# print("c1: ", intt_c1)




# print("================================")
# print("Decrypt")
# print("================================")

# temp1 = poly_mult(c1, sk, q)
# temp2 = poly_add(c0, temp1, q)
# m = temp2 // delta
# print("Decrypted message: ", m)
# ntt_sk = ntt(sk, q, ntt_w, 1, N).flatten()
# ntt_c1 = ntt(c1, q, ntt_w, 1, N).flatten()
# ntt_c0 = ntt(c0, q, ntt_w, 1, N).flatten()

# temp1 = [0] * N
# for i in range(N):
#     temp1[i] = (ntt_c1[i] * ntt_sk[i]) % q
# for i in range(N):
#     temp1[i] = (ntt_c0[i] + temp1[i]) % q
# temp1 = np.array(temp1).reshape(shape)
# intt_m = intt(temp1, q, intt_w, 1, N)
# intt_m = [item // delta for item in intt_m]
# print("Decrypted message under NTT: ", intt_m)








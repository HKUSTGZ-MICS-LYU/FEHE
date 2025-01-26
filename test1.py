import sympy
import random
import time

prime = 998244353

def center(a, prime):
    if a > prime//2:
        return a - prime
    elif a < -prime//2:
        return a + prime
    else:
        return a
    

def poly_mul(a, b, prime):
    n = len(a)
    temp = [0] * (2 * n)
    for i in range(n):
        for j in range(n):
            temp[i + j] = center((temp[i + j] + a[i]*b[j]) , prime)
    c = [0] * n
    for i in range(n):
        c[i] = temp[i] + temp[i + n]
        c[i] = center(c[i], prime)
        
    return c

def ntt_poly_mul(a, b):
    ntt_a = sympy.ntt(a, prime)
    ntt_b = sympy.ntt(b, prime)
    ntt_c = [(ai * bi) % prime for ai, bi in zip(ntt_a, ntt_b)]
    c = sympy.intt(ntt_c, prime)
    return c

def poly_add(a, b):
    n = len(a)
    c = [0] * n
    for i in range(n):
        c[i] = (a[i] + b[i]) % prime
    return c

def ntt_poly_add(a, b):
    ntt_a = sympy.ntt(a, prime)
    ntt_b = sympy.ntt(b, prime)
    ntt_c = [(ai + bi) % prime for ai, bi in zip(ntt_a, ntt_b)]
    c = sympy.intt(ntt_c, prime)
    return c

# # 简化测试用例
# a = [random.randint(0, 8) for _ in range(8192)]
# b = [random.randint(0, 8) for _ in range(8192)]

# naive_start = time.time()
# c = poly_add(a, b)
# naive_end = time.time()
# print(f"Naive time: {naive_end - naive_start}")

# ntt_start = time.time()
# c1 = ntt_poly_add(a, b)
# ntt_end = time.time()
# print(f"NTT time: {ntt_end - ntt_start}")
# print(f"Time Ratio: {(ntt_end - ntt_start) / (naive_end - naive_start)}, {(naive_end - naive_start) / (ntt_end - ntt_start)}")

# print(f"Results are the same: {c == c1}")
# print(f"Results: {c[:10]}")
# print(f"Results: {c1[:10]}")
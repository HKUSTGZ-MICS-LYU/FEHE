import sympy
import random
import time
plaintext_modulus = 65537
prime = 1073750017

def center(a, prime):
    if a > prime//2:
        return a - prime
    elif a < -prime//2:
        return a + prime
    else:
        return a

def poly_mul(a, b):
    """
    Naive polynomial multiplication in (Z/primeZ)[x] / (x^n - 1),
    because we do c[i] = temp[i] + temp[i+n].
    Length of a and b must be the same.
    """
    n = len(a)
    temp = [0]*(2*n)
    for i in range(n):
        for j in range(n):
            temp[i + j] = (temp[i + j] + a[i]*b[j]) % prime
    c = [0]*n
    for i in range(n):
        # x^n = 1 => fold temp[i+n] back into temp[i]
        c[i] = (temp[i] + temp[i + n]) % prime
        c[i] = center(c[i], prime)
    return c

def ntt_poly_mul(a, b):
    """
    Sympy-based NTT polynomial multiplication in (Z/primeZ)[x] / (x^n - 1).
    """
    ntt_a = sympy.ntt(a, prime)
    ntt_b = sympy.ntt(b, prime)
    ntt_c = [(ai * bi) % prime for ai, bi in zip(ntt_a, ntt_b)]
    c = sympy.intt(ntt_c, prime)
    return c

def poly_add(a, b):
    """
    Element-wise addition of polynomials a and b mod prime.
    """
    return [ (ai + bi) % prime for (ai, bi) in zip(a, b) ]

def poly_sub(a, b):
    """
    Element-wise subtraction of polynomials a and b mod prime.
    """
    return [ (ai - bi) % prime for (ai, bi) in zip(a, b) ]

def ntt_poly_add(a, b):
    """
    Using NTT to do addition is overkill, but if you do:
      c = NTT^-1( NTT(a) + NTT(b) )
    it also yields the same result in (x^n - 1).
    """
    ntt_a = sympy.ntt(a, prime)
    ntt_b = sympy.ntt(b, prime)
    ntt_c = [(ai + bi) % prime for (ai, bi) in zip(ntt_a, ntt_b)]
    c = sympy.intt(ntt_c, prime)
    return c

# ---------------------------------------------------------
# Test / Demo
# ---------------------------------------------------------
delta = 17
n = 4096

# Random polynomials
m  = [random.randint(0, plaintext_modulus) for _ in range(n)]
sk = [random.randint(0, 2) for _ in range(n)]
u  = [random.randint(0, prime) for _ in range(n)]
e1 = [random.randint(0, 2) for _ in range(n)]

# ============= Naive approach =============
print("Naive approach to Encrypt and Decrypt")
naive_start = time.time()

# 1) res1 = m * delta, but do element-wise scalar multiply (NOT list repetition)
res1 = [ (mi * delta) % prime for mi in m ]

# 2) res2 = poly_mul(pk, u)  (Naive polynomial multiply mod x^n - 1)
res2 = poly_mul(sk, u)

# 3) res3 = res1 + res2 + e1 (element-wise polynomial addition)
res3 = [ (res1[i] + res2[i] + e1[i]) % prime for i in range(n) ]

# 4) res4 = res3 - u * sk
a_sk = poly_mul(u, sk)
res4 = poly_sub(res3, a_sk)

# 5) res5 = FLOOR(res4 / delta)
res5 = [ (res4[i] // delta)%plaintext_modulus for i in range(n) ]


naive_end = time.time()
naive_time = naive_end - naive_start
print(f"Naive time: {naive_time:.6f} s")




# ============= NTT approach =============
print("\nNTT approach to Encrypt and Decrypt")
ntt_start = time.time()

# 1) res1 = m * delta, still do element-wise scalar multiply
res1_ntt = [ (mi * delta) % prime for mi in m ]

# Convert all to NTT domain
ntt_res1 = sympy.ntt(res1_ntt, prime)
ntt_sk   = sympy.ntt(sk, prime)
ntt_u    = sympy.ntt(u, prime)
ntt_e1   = sympy.ntt(e1, prime)

# 2) res2 = pk * u in NTT domain (pointwise multiply)
ntt_res2 = [(ski * ui) % prime for ski, ui in zip(ntt_sk, ntt_u)]

# 3) res3 = res1 + res2 + e1 in NTT domain (pointwise add)
ntt_res3 = [ (res1i + res2i + e1i) % prime
             for (res1i, res2i, e1i) in zip(ntt_res1, ntt_res2, ntt_e1) ]

# 4) res4 = res3 - u * sk in NTT domain (pointwise subtract)
ntt_a_sk = [(ui * ski) % prime for ui, ski in zip(ntt_u, ntt_sk)]
ntt_res4 = [ (res3i - a_ski) % prime for (res3i, a_ski) in zip(ntt_res3, ntt_a_sk) ]

# 5) res5 = FLOOR(intt(res4) / delta)
intt_res4 = sympy.intt(ntt_res4, prime)
intt_res5 = [ (intti // delta)%plaintext_modulus for intti in intt_res4 ]

ntt_end = time.time()
ntt_time = ntt_end - ntt_start
print(f"NTT   time: {ntt_time:.6f} s")

if naive_time > 1e-9:
    print(f"SPEEDUP: {naive_time / ntt_time:.2f}x")

# Print partial results to confirm correctness
print("\nFirst 10 naive results: ", res5[:10])
print("\nFirst 10  NTT  results: ", intt_res5[:10])
print("\nOriginal Message: ", m[:10])



if res5 == intt_res5:
    print("Results match!")
else:
    print("Results differ!")


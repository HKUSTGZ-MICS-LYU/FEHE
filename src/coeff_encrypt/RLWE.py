import random
import numpy as np
from Rq import Rq
import math
from utils import crange




class RLWE:
    def __init__(self, n, p, t):
        assert np.log2(n) == int(np.log2(n))
        self.n = n
        self.p = p
        self.t = t
        self.delta = self.p // self.t

    def generate_keys(self):
        s = sample(2, self.p, self.n)
        e = sample(2, self.p, self.n)
        a = sample(self.p, self.p, self.n)
        return (s, (-1 * (a * s + e), a))  # (secret, public)

    def encrypt(self, m, a):
        '''
        # Args:
            m: plaintext (mod t)
            a: public key (a0, a1)
        '''
        e1 = sample(2, self.p, self.n)
        e2 = sample(2, self.p, self.n)
        u = sample(2, self.p, self.n)
        m = Rq(m.poly.coeffs, self.p) 
        ct = (self.delta * m + e1 + a[0] * u, a[1] * u + e2)

        return ct, e1, e2, u

    def decrypt(self, c, s):
        '''
        # Args:
            c: ciphertext (c0, c1, ..., ck)
            s: secret key
        '''
        temp = c[1] * s
        m = c[0] + temp
        for i in range(len(m.poly.coeffs)):
            m.poly.coeffs[i] = round(m.poly.coeffs[i] / self.delta)
        
        m = Rq(m.poly.coeffs, self.t)

        return m

    def add(self, c0, c1):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        c = ()

        k0 = len(c0)  # not necessary to compute (len - 1)
        k1 = len(c1)

        if k0 > k1:
            (c0, c1) = (c1, c0)  # c0 is always shorter

        for _ in range(abs(k0 - k1)):
            c0 += (Rq([0], self.p),)  # add 0 to shorter ciphertext

        for i in range(len(c0)):
            c += (c0[i] + c1[i],)

        return c

    def mul(self, c0, c1):
        '''
        # Args:
            c0: ciphertext (c0, c1, ..., ck)
            c1: ciphertext (c'0, c'1, ..., c'k')
        '''
        c = ()

        k0 = len(c0) - 1
        k1 = len(c1) - 1

        for _ in range(k1):
            c0 += (Rq([0], self.p),)

        for _ in range(k0):
            c1 += (Rq([0], self.p),)

        for i in range(k0 + k1 + 1):
            _c = Rq([0], self.p)
            for j in range(i+1):
                _c += c0[j] * c1[i-j]
            c += (_c,)

        return c

def sample(num, q, n):
    coeffs = [0] * n
    for i in range(n):
        coeffs[i] = random.randint(0, num-1)
    return Rq(coeffs, q)

def discrete_gaussian(n, q, mean=0., std=1.):
    coeffs = np.round(std * np.random.randn(n))
    return Rq(coeffs, q)


def discrete_uniform(n, q, min=0., max=None):
    if max is None:
        max = q
    coeffs = np.random.randint(min, max, size=n)
    return Rq(coeffs, q)
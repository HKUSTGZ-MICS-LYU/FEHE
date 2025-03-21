from math import ceil
import numpy as np
from Rq import Rq
from RLWE import RLWE
import sympy
from NTT import *
from utils import generate_twidle_factors, crange

def center_lift(x, q):
    return np.array([int(xi) if xi <= q//2 else int(xi - q) for xi in x])

if __name__ == '__main__':
    n = 16        # polynomial degree
    q = 67108289    # ciphertext modulus
    t = 37          # plaintext modulus


    rlwe = RLWE(n, q, t)
    (sec, pub) = rlwe.generate_keys()

    m0 = Rq(np.random.randint(t, size=n), t)  # plaintext
    c0, e1, e2, u = rlwe.encrypt(m0, pub)
    m_0 = rlwe.decrypt(c0, sec)
    print(f"{m0}")
    print(f"{m_0}")
    


    
  
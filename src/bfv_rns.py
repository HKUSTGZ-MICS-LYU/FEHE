from sympy import *
from decimal import *
import random
import sys
import math
import datetime
import pickle
from functools import reduce

getcontext().prec = 300

def barrett_reduction(a, MOD):
    r = a%MOD
    k_half = math.ceil(math.log(MOD, 2))
    m = math.floor(2**(k_half*2) / MOD)
   
    return r


def barrett_mul(a, b, MOD):
    temp = a * b
    return barrett_reduction(temp, MOD)

def chinese_remainder(n, a):
    sum = 0
    prod = reduce(lambda a, b: a * b, n)
    for n_i, a_i in zip(n, a):
        p = prod // n_i
        inv = mul_inv(p, n_i)
        inter_sum = a_i * inv * p
        sum += inter_sum
    return sum % prod

def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1

def center(a, b):
    if b == 0:
        return a
    else:
        return a % b
    if a > b // 2:
        a -= b
    elif a < -b // 2:
        a += b
    return a

def modInverse(a, m):
    m0 = m
    y = 0
    x = 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
    if x < 0:
        x = x + m0
    return center(x, m)

def FindPrime(q, n):
    while True:
        q = nextprime(q)
        if (q - 1) % (2 * n) == 0:
            break
    return q

def sample(num, q, n):
    ret = [0] * n
    for i in range(n):
        ret[i] = random.randint(0, num - 1)
    return polynomial(ret, q)

def bit_decomposition(a, q, T):
    l = math.floor(math.log(q, T))
    out = [[0 for x in range(a.n)] for y in range((l + 1))]
    for i in range(a.n):
        tmp = a.poly[i]
        for j in range(l + 1):
            num = tmp % T
            out[j][i] = num
            tmp -= num
            tmp = tmp // T
    ret = []
    for i in range(l + 1):
        ret.append(polynomial(out[i], q))
    return ret

def bit_combination(polys, q, T):
    l = math.floor(math.log(q, T))
    out = [0] * polys[0].n
    for i in range(polys[0].n):
        tmp = 0
        for j in range(l + 1):
            tmp = tmp + (polys[j].poly[i] * T ** j)
        out[i] = tmp
    return polynomial(ret, mod, False)

def ntt_inverse(poly, table):
    ret = [i for i in poly.poly]
    mod = poly.mod
    for i in range(int(math.log(poly.n, 2))):
        step = 2 ** (i)
        group = int(poly.n // len(table[int(math.log(poly.n, 2)) - 1 - i]))
        for k in range(len(table[int(math.log(poly.n, 2)) - 1 - i])):
            for j in range(step):
                ta = ret[k * group + j]
                tb = ret[k * group + j + step]
                ret[k * group + j] = center((ta + tb) % mod, mod)
                ret[k * group + j + step] = center(
                    barrett_mul((ta - tb), table[int(math.log(n, 2)) - 1 - i][k], mod), mod)
    for i in range(poly.n):
        ret[i] = center(ret[i] * modInverse(poly.n, mod) % mod, mod)
    return polynomial(ret, mod, False)

def ntt_forward(poly, table):
    ret = [i for i in poly.poly]
    mod = poly.mod
    for i in range(int(math.log(poly.n, 2))):
        step = 2 ** (int(math.log(poly.n, 2)) - 1 - i)
        group = int(poly.n // len(table[i]))
        for k in range(len(table[i])):
            for j in range(step):
                ta = ret[k * group + j]
                tb = ret[k * group + j + step]
                tf = table[i][k]
                ret[k * group + j] = center((ta + barrett_mul(tb, table[i][k], mod)) % mod, mod)
                ret[k * group + j + step] = center((ta - barrett_mul(tb, table[i][k], mod)) % mod, mod)
    return polynomial(ret, mod, True)

def ntt_inverse_bunch(polys, tables):
    ret = []
    for poly, table in zip(polys, tables):
        ret.append(ntt_inverse(poly, table))
    return ret

def ntt_forward_bunch(polys, tables):
    ret = []
    for poly, table in zip(polys, tables):
        ret.append(ntt_forward(poly, table))
    return ret

class polynomial(object):
    def __init__(self, input_list, mod, is_ntt=False):
        super(polynomial, self).__init__()
        if isinstance(input_list, list):
            self.poly = [center(i, mod) for i in input_list]
            self.n = len(self.poly)
            self.mod = mod
            self.ntt = is_ntt
        else:
            raise Exception("Polynomial input should be list.")

    def __str__(self):
        return str(self.poly)

    def __mul__(self, other):
        if isinstance(other, polynomial) and not self.ntt and not other.ntt:
            if self.mod != other.mod or self.n != other.n:
                raise Exception("mod or n not the same")
            temp = [0] * (2 * self.n)
            for i in range(self.n):
                for j in range(self.n):
                    temp[i + j] = (temp[i + j] + self.poly[i] * other.poly[j]) % self.mod
            ret = [0] * self.n
            for i in range(self.n):
                ret[i] = center((temp[i] - temp[i + self.n]) % self.mod, self.mod)
            return self.__class__(ret, self.mod, self.ntt)
        elif isinstance(other, polynomial) and self.ntt and other.ntt:
            if self.mod != other.mod or self.n != other.n:
                raise Exception("mod or n not the same")
            ret = [0] * self.n
            for i in range(self.n):
                ret[i] = center((self.poly[i] * other.poly[i]) % self.mod, self.mod)
            return self.__class__(ret, self.mod, self.ntt)
        elif (isinstance(other, int) or isinstance(other, float)) and not self.ntt:
            temp = [0] * self.n
            for i in range(self.n):
                temp[i] = center((self.poly[i] * other) % self.mod, self.mod)
            return self.__class__(temp, self.mod)
        else:
            raise Exception("Not supported")

    def __add__(self, other):
        if isinstance(other, polynomial):
            if self.mod != other.mod or self.n != other.n:
                raise Exception("mod or n not the same")
            temp = [0] * self.n
            for i in range(self.n):
                temp[i] = center((self.poly[i] + other.poly[i]) % self.mod, self.mod)
            return self.__class__(temp, self.mod, self.ntt)
        elif isinstance(other, int):
            temp = [0] * self.n
            for i in range(self.n):
                temp[i] = center((self.poly[i] + other) % self.mod, self.mod)
            return self.__class__(temp, self.mod, self.ntt)
        else:
            raise Exception("Not supported")

    def __neg__(self):
        temp = [0] * self.n
        for i in range(self.n):
            temp[i] = center((-self.poly[i]) % self.mod, self.mod)
        return self.__class__(temp, self.mod)

    def change_mod(self, new_mod):
        if self.mod > new_mod:
            return polynomial([center(self.poly[i] % new_mod, new_mod) for i in range(self.n)], new_mod)
        return polynomial(self.poly, new_mod)

    def center(self):
        for i in range(self.n):
            self.poly[i] = center(self.poly[i], self.mod)

class BFVEncryptor(object):
    def __init__(self, q, p, n, t, q_i, p_i, q_tilde, p_tilde, q_star, p_star, Q_tilde_q, Q_tilde_p,
                 ntt_forward_table, ntt_inverse_table):
        super(BFVEncryptor, self).__init__()

        self.q_i = q_i
        self.p_i = p_i
        self.q_tilde = q_tilde
        self.p_tilde = p_tilde
        self.q_star = q_star
        self.p_star = p_star
        self.Q_i = q_i + p_i
        self.Q_tilde_q = Q_tilde_q
        self.Q_tilde_p = Q_tilde_p
        self.ntt_forward_table = ntt_forward_table
        self.ntt_inverse_table = ntt_inverse_table
        self.RNS_NUM = len(self.q_i) + 1
        self.m = n * 2
        self.n = n
        self.bits = int(math.log(self.n, 2))
        self.q = q
        self.p = p
        self.t = t
        self.Q = self.p * self.q
        self.delta = self.q // self.t

        self.keygen()
        self.evaluation_keygen()
        self.galois_keygen()

    def keygen(self):
        a = sample(self.q, self.q, self.n)
        s = sample(2, self.q, self.n)
        e = sample(2, self.q, self.n)
        
        self.pk = (-(a * s + e), a)
        self.sk = s

    def evaluation_keygen(self):
        self.rlk = []
        s_square = self.sk * self.sk

        for i, q_i in enumerate(self.q_i):
            a = sample(self.q, self.q, self.n)
            e = sample(2, self.q, self.n)
            (alpha, beta) = (-(a * self.sk) + e + s_square * self.q_star[i] * self.q_tilde[i], a)
            rlk_0 = []
            rlk_1 = []
            for j, q_j in enumerate(self.q_i):
                rlk_0.append(ntt_forward(alpha.change_mod(q_j), self.ntt_forward_table[j]))
                rlk_1.append(ntt_forward(beta.change_mod(q_j), self.ntt_forward_table[j]))
            self.rlk.append((rlk_0, rlk_1))

    def galois_keygen(self):
        self.galois = []

        for sign in [1, -1]:
            for p in [1 << pp for pp in range(self.bits - 1)]:
                temp = []
                s_ = self.rotate(self.sk, (p * sign + (self.n // 2)) % (self.n // 2), 3)
                for i, q_i in enumerate(self.q_i):
                    a = sample(self.q, self.q, self.n)
                    e = sample(2, self.q, self.n)
                    (alpha, beta) = (-(a * self.sk) + e + s_ * self.q_star[i] * self.q_tilde[i], a)
                    gk_0 = []
                    gk_1 = []
                    for j, q_j in enumerate(self.q_i):
                        gk_0.append(ntt_forward(alpha.change_mod(q_j), self.ntt_forward_table[j]))
                        gk_1.append(ntt_forward(beta.change_mod(q_j), self.ntt_forward_table[j]))
                    temp.append((gk_0, gk_1))
                self.galois.append(temp)
        temp = []
        s_ = self.rotate(self.sk, 1, self.m - 1)
        for i, q_i in enumerate(self.q_i):
            a = sample(self.q, self.q, self.n)
            e = sample(2, self.q, self.n)
            (alpha, beta) = (-(a * self.sk) + e + s_ * self.q_star[i] * self.q_tilde[i], a)
            gk_0 = []
            gk_1 = []
            for j, q_j in enumerate(self.q_i):
                gk_0.append(ntt_forward(alpha.change_mod(q_j), self.ntt_forward_table[j]))
                gk_1.append(ntt_forward(beta.change_mod(q_j), self.ntt_forward_table[j]))
            temp.append((gk_0, gk_1))
        self.galois.append(temp)

    def encrypt(self, poly):
        e1 = sample(2, self.q, self.n)
        e2 = sample(2, self.q, self.n)
        u = sample(2, self.q, self.n)
        m = poly.change_mod(self.q)
        tmp_ct = (m * self.delta + self.pk[0] * u + e1, self.pk[1] * u + e2)

        ct0 = []
        ct1 = []
        for i, q_i in enumerate(self.q_i):
            ct0.append(ntt_forward(tmp_ct[0].change_mod(q_i), self.ntt_forward_table[i]))
            ct1.append(ntt_forward(tmp_ct[1].change_mod(q_i), self.ntt_forward_table[i]))

        return (ct0, ct1)

    def decrypt(self, ct):
        ct = (ntt_inverse_bunch(ct[0], self.ntt_inverse_table),
              ntt_inverse_bunch(ct[1], self.ntt_inverse_table))

        polys = []
        for i, mod in enumerate(self.q_i):
            x = (ct[0][i] + ct[1][i] * self.sk.change_mod(mod))
            polys.append(x.poly)
        ret = [0] * self.n
        for i in range(self.n):
            temp = 0
            for j, mod in enumerate(self.q_i):
                temp += polys[j][i] * self.q_tilde[j] * Decimal(self.t) / Decimal(mod)
            ret[i] = int(round(temp)) % self.t

        return polynomial(ret, self.t)

    def printBudget(self, ct):
        ct = (ntt_inverse_bunch(ct[0], self.ntt_inverse_table),
              ntt_inverse_bunch(ct[1], self.ntt_inverse_table))
        polys = []
        for i, mod in enumerate(self.q_i):
            x = (ct[0][i] + ct[1][i] * self.sk.change_mod(mod))
            polys.append(x.poly)
        first_element = 0
        for i, mod in enumerate(self.q_i):
            first_element += polys[i][0] * self.q_tilde[i] * Decimal(self.t) / Decimal(mod)
        budget = abs(math.log(abs((first_element - round(first_element))), 2))
        if abs((first_element - round(first_element))) > 0.5:
            print("Budget: ", 0)
        else:
            print("Budget: ", budget)

    def PlainAdd(self, rns_a, plain):
        ct0 = []
        for a, b in zip(rns_a[0], plain):
            ct0.append(a + b)

        return (ct0, rns_a[1])

    def PlainMul(self, rns_a, plain):
        ct0 = []
        ct1 = []
        for (a, b, c) in zip(rns_a[0], rns_a[1], plain):
            ct0.append(a * c)
            ct1.append(b * c)

        return (ct0, ct1)

    def HAdd(self, rns_a, rns_b):
        ct0 = []
        ct1 = []
        for i in range(len(self.q_i)):
            ct0.append(rns_a[0][i] + rns_b[0][i])
            ct1.append(rns_a[1][i] + rns_b[1][i])

        return (ct0, ct1)

    def HMul(self, rns_a, rns_b):

        inv_nttd_rns_a_0 = ntt_inverse_bunch(rns_a[0], self.ntt_inverse_table)
        inv_nttd_rns_a_1 = ntt_inverse_bunch(rns_a[1], self.ntt_inverse_table)
        inv_nttd_rns_b_0 = ntt_inverse_bunch(rns_b[0], self.ntt_inverse_table)
        inv_nttd_rns_b_1 = ntt_inverse_bunch(rns_b[1], self.ntt_inverse_table)
                
        rns_aa = (self.basis_extension_forward(inv_nttd_rns_a_0),
                  self.basis_extension_forward(inv_nttd_rns_a_1))
        
        rns_bb = (self.basis_extension_forward(inv_nttd_rns_b_0),
                  self.basis_extension_forward(inv_nttd_rns_b_1))
        
        
        nttd_rns_a_0 = ntt_forward_bunch(rns_aa[0], self.ntt_forward_table)
        nttd_rns_a_1 = ntt_forward_bunch(rns_aa[1], self.ntt_forward_table)
        nttd_rns_b_0 = ntt_forward_bunch(rns_bb[0], self.ntt_forward_table)
        nttd_rns_b_1 = ntt_forward_bunch(rns_bb[1], self.ntt_forward_table)
       
        c0 = []
        c1 = []
        c2 = []
        for i in range(len(self.q_i + self.p_i)):
            c0.append(nttd_rns_a_0[i] * nttd_rns_b_0[i])
            c1.append(nttd_rns_a_0[i] * nttd_rns_b_1[i] + nttd_rns_a_1[i] * nttd_rns_b_0[i])
            c2.append(nttd_rns_a_1[i] * nttd_rns_b_1[i])
        
        c0 = ntt_inverse_bunch(c0, self.ntt_inverse_table)
        c1 = ntt_inverse_bunch(c1, self.ntt_inverse_table)
        c2 = ntt_inverse_bunch(c2, self.ntt_inverse_table)

        c0 = self.scaling(c0)
        c1 = self.scaling(c1)
        c2 = self.scaling(c2)

        c0 = self.basis_extension_backward(c0)
        c1 = self.basis_extension_backward(c1)
        c2 = self.basis_extension_backward(c2)

        return self.relin(c0, c1, c2)

    def scaling(self, a):
        tmp = []
        for i, ii in enumerate(self.p_i):
            poly = []
            for j in range(self.n):
                temp = 0
                for idx, k in enumerate(self.q_i):
                    temp1 = a[idx].poly[j]
                    temp2 = int((self.t * self.Q_tilde_q[idx] * self.p / Decimal(k))) % ii
                    temp4 = (temp1 * temp2) % ii
                    temp3 = (self.t * self.Q_tilde_q[idx] * self.p / Decimal(k)) % ii - temp2
                    temp5 = (temp1 * temp3) % ii
                    temp += temp4 + temp5
                temp = int(round(temp))
                m = (a[len(self.q_i) + i].poly[j] * ((self.t * self.Q_tilde_p[i] * self.p_star[i])) % ii)
                temp = (temp + m) % ii
                poly.append(center(temp, ii))
            tmp.append(polynomial(poly, ii))
        return tmp

    def basis_extension_forward(self, a):
        tmp = []
        for i in self.p_i:
            poly1 = []
            for j in range(self.n):
                temp1 = 0
                v1 = 0
                pre1 = []
                for idx, k in enumerate(self.q_i):
                    pre1.append(center(a[idx].poly[j] * self.q_tilde[idx], k))
                for idx, k in enumerate(self.q_i):
                    temp1 = (temp1 + (pre1[idx] * (self.q_star[idx] % i)) % i) % i
                    v1 += pre1[idx] * (1 / k)
                v1 = round(v1)
                v1 = (v1 * (self.q % i)) % i
                temp1 = center(temp1 - v1, i)
                poly1.append(temp1)
            tmp.append(polynomial(poly1, i))
        return a + tmp

    def basis_extension_backward(self, a):
        tmp = []
        for i in self.q_i:
            poly1 = []
            for j in range(self.n):
                temp1 = 0
                v1 = 0
                pre1 = []
                for idx, k in enumerate(self.p_i):
                    pre1.append(center(a[idx].poly[j] * self.p_tilde[idx], k))
                for idx, k in enumerate(self.p_i):
                    temp1 = (temp1 + (pre1[idx] * (self.p_star[idx] % i)) % i) % i
                    v1 += (pre1[idx]) / Decimal(k)
                v1 = int(round(v1))
                v1 = v1 * (self.p % i) % i
                temp1 = center(temp1 - v1, i)
                poly1.append(temp1)
            tmp.append(polynomial(poly1, i))
        return tmp

    def relin(self, c0, c1, c2):
        
        nttd_c0 = []
        nttd_c1 = []
        nttd_c2 = []
        

        for i in range(len(self.q_i)):
            for j in range(len(self.q_i)):
                c2[i] = c2[i].change_mod(self.q_i[j])
                nttd_c2.append(ntt_forward(c2[i], self.ntt_forward_table[j]))
        
        
        for i in range(len(self.q_i)):
            nttd_c0.append(ntt_forward(c0[i], self.ntt_forward_table[i]))
            nttd_c1.append(ntt_forward(c1[i], self.ntt_forward_table[i]))
           
        decomposition = self.keyswitching(c2, self.rlk)
    
        return self.HAdd((nttd_c0, nttd_c1), decomposition)

    def rotate(self, poly, r, basis):
        rotate_in_Z_m = basis ** r
        after0 = [0] * self.n
        for i in range(self.n):
            new_value = poly.poly[i]
            if ((i * rotate_in_Z_m) >> self.bits) % 2 == 1:
                new_value *= -1
            new_idx = (i * rotate_in_Z_m) % self.n
            after0[new_idx] = new_value
        return polynomial(after0, poly.mod)

    def rotate_column(self, poly, r, idx):
        r = (r + (self.n // 2)) % (self.n // 2)
        ct0 = []
        ct1 = []
        for i in range(len(self.q_i)):
            a = self.rotate(ntt_inverse(poly[0][i], self.ntt_inverse_table[i]), r, 3)
            b = self.rotate(ntt_inverse(poly[1][i], self.ntt_inverse_table[i]), r, 3)
            ct0.append(ntt_forward(a, self.ntt_forward_table[i]))
            ct1.append(b)

        decomposition = self.keyswitching(ct1, self.galois[idx])

        ret = []
        for i in range(len(self.q_i)):
            ret.append(ct0[i] + decomposition[0][i])

        return (ret, decomposition[1])

    def rotate_row(self, poly, r, idx):
        r = (r + 2) % 2
        ct0 = []
        ct1 = []
        for i in range(len(self.q_i)):
            a0 = ntt_inverse(poly[0][i], self.ntt_inverse_table[i])
            a = self.rotate(a0, r, self.m - 1)
            b = self.rotate(ntt_inverse(poly[1][i], self.ntt_inverse_table[i]), r, self.m - 1)
            ct0.append(ntt_forward(a, self.ntt_forward_table[i]))
            ct1.append(b)

        decomposition = self.keyswitching(ct1, self.galois[idx])

        ret = []
        for i in range(len(self.q_i)):
            ret.append(ct0[i] + decomposition[0][i])

        return (ret, decomposition[1])

    def keyswitching(self, rns, keys):
        ct0 = []
        ct1 = []
        for i, q_i in enumerate(self.q_i):
            c0_ = 0
            c1_ = 0
            for j in range(len(self.q_i)):
                decomposed_c1 = rns[j].change_mod(q_i)
                decomposed_c1 = ntt_forward(decomposed_c1, self.ntt_forward_table[i])

           
                c0_ = (decomposed_c1 * keys[j][0][i]) + c0_
                c1_ = (decomposed_c1 * keys[j][1][i]) + c1_
            ct0.append(c0_)
            ct1.append(c1_)
        return (ct0, ct1)

class BFVEncoder(object):
    def __init__(self, t, n):
        super(BFVEncoder, self).__init__()

        self.n = n
        self.m = self.n * 2
        self.t = t
        G = 2
        for i in range(2, self.t):
            if is_primitive_root(i, self.t):
                G = i
                break
        self.G = G

        self.root = pow(self.G, ((self.t - 1) // self.m), self.t)
        generators = [3, self.m - 1]
        orders = [n_order(i, self.m) for i in generators]

        basis = [0] * self.n
        for i in range(self.n):
            basis[i] = generators[0] ** (i % orders[0]) * generators[1] ** (i // orders[0]) % self.m

        self.basis = basis

    def encode(self, poly):
        ret = [0] * self.n
        for i in range(self.n):
            s = 0
            for j in range(self.n):
                s = (s + poly[j] * pow(self.root, (-i * self.basis[j]), self.t)) % self.t
            ret[i] = s
        for i in range(self.n):
            ret[i] = ret[i] * modInverse(self.n, self.t) % self.t
        return polynomial(ret, self.t)

    def decode(self, poly):
        ret = [0] * self.n
        for i in range(self.n):
            s = 0
            for j in range(self.n):
                s = (s + poly.poly[j] * pow(self.root, (j * self.basis[i]), self.t)) % self.t
            ret[i] = center(s, self.t)
        return ret

    def rns_ntt_pt_addition(self, poly, encryptor):
        pt = []
        for q_i in encryptor.q_i:
            temp = encryptor.delta % q_i
            pt.append(poly.change_mod(q_i) * temp)
        return ntt_forward_bunch(pt, encryptor.ntt_forward_table)

    def rns_ntt_pt_multiplication(self, poly, encryptor):
        pt = []
        for q_i in encryptor.q_i:
            pt.append(poly.change_mod(q_i))
        return ntt_forward_bunch(pt, encryptor.ntt_forward_table)

class BFVContext(object):
    def __init__(self, bit_list, n, t):
        super(BFVContext, self).__init__()

        security = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
        total_bits = 0
        for i in bit_list:
            total_bits += i

        self.t = t
        self.n = n
        self.t_all = 1

        self.q_i = []
        q = 1
        for i in bit_list:
            prime = FindPrime(2 ** i, n)
            while prime in self.q_i:
                prime = FindPrime(prime, n)
            self.q_i.append(prime)
            q *= prime
        self.p_i = []
        p = 1
        for i in bit_list + [bit_list[-1]]:
            prime = FindPrime(2 ** i, n)
            while prime in self.q_i or prime in self.p_i:
                prime = FindPrime(prime, n)
            self.p_i.append(prime)
            p *= prime

        self.RNS_NUM = len(self.q_i) + 1
        self.m = n * 2
        self.n = n
        self.bits = int(math.log(self.n, 2))
        self.q = q
        self.p = p
        self.Q = self.p * self.q

        self.set_constants()

        self.encryptor = []
        self.encoder = []

        for i in self.t:
            self.encoder.append(BFVEncoder(i, self.n))
            self.encryptor.append(BFVEncryptor(self.q, self.p, self.n, i, self.q_i, self.p_i, self.q_tilde, self.p_tilde,
                                               self.q_star, self.p_star, self.Q_tilde_q, self.Q_tilde_p,
                                               self.ntt_forward_table, self.ntt_inverse_table))
            self.t_all *= i

        self.t_star = []
        self.t_tilde = []
        for i in self.t:
            temp = int(self.t_all // i)
            self.t_star.append(temp)
            self.t_tilde.append(center(modInverse(temp, i), i))

    def rns_ntt_pt_addition(self, pt):
        ret = []
        for p, encoder, encryptor in zip(pt, self.encoder, self.encryptor):
            ret.append(encoder.rns_ntt_pt_addition(p, encryptor))
        return ret

    def rns_ntt_pt_multiplication(self, pt):
        ret = []
        for p, encoder, encryptor in zip(pt, self.encoder, self.encryptor):
            ret.append(encoder.rns_ntt_pt_multiplication(p, encryptor))
        return ret

    def encrypt(self, pt):
        ret = []
        for p, encryptor in zip(pt, self.encryptor):
            ret.append(encryptor.encrypt(p))
        return ret

    def decrypt(self, ct):
        ret = []
        for c, encryptor in zip(ct, self.encryptor):
            ret.append(encryptor.decrypt(c))
        return ret

    def PlainAdd(self, ct, pt):
        ret = []
        for c, p, encryptor in zip(ct, pt, self.encryptor):
            ret.append(encryptor.PlainAdd(c, p))
        return ret

    def PlainMul(self, ct, pt):
        ret = []
        for c, p, encryptor in zip(ct, pt, self.encryptor):
            ret.append(encryptor.PlainMul(c, p))
        return ret

    def HAdd(self, ct0, ct1):
        ret = []
        for c0, c1, encryptor in zip(ct0, ct1, self.encryptor):
            ret.append(encryptor.HAdd(c0, c1))
        return ret

    def HMul(self, ct0, ct1):
        ret = []
        for c0, c1, encryptor in zip(ct0, ct1, self.encryptor):
            ret.append(encryptor.HMul(c0, c1))
        return ret

    def decompose_rotate(self, a):
        l = []
        while a != 0:
            t = (round(math.log(abs(a), 2)))
            num = 1 << t
            if a < 0:
                num = -num
                t = t + (int(math.log(self.n, 2)) - 1)
            a -= num
            if num % (self.n // 2) == 0:
                continue
            l.append((num, t))
        return l

    def rotate_column(self, ct, r):
        r = -r
        if r >= self.n // 2 and r <= -self.n // 2:
            raise Exception("Not supported")

        l = self.decompose_rotate(r)
        ret = []
        for c, encryptor in zip(ct, self.encryptor):
            cc = c
            for i in l:
                cc = encryptor.rotate_column(cc, i[0], i[1])
            ret.append(cc)
        return ret

    def rotate_row(self, ct, r):
        if r != 1:
            raise Exception("Not supported")
        ret = []
        for c, encryptor in zip(ct, self.encryptor):
            ret.append(encryptor.rotate_row(c, r, 2 * (self.bits - 1)))
        ret.append(ret)
        return ret

    def decode_and_reconstruct(self, polys):
        decoded_poly = []
        for i, p in enumerate(polys):
            decoded_poly.append(self.encoder[i].decode(p))

        ret = []
        for i in range(self.n):
            temp = 0
            for j, poly in enumerate(decoded_poly):
                temp += poly[i] * self.t_star[j] * self.t_tilde[j]
            ret.append(center(temp % self.t_all, self.t_all))
        return ret

    def crt_and_encode(self, poly):
        ret = []
        for i, t in enumerate(self.t):
            temp = [0] * self.n
            for j in range(self.n):
                temp[j] = center(poly[j] % t, t)
            ret.append(self.encoder[i].encode(temp))
        return ret

    def printBudget(self, ct):
        self.encryptor[0].printBudget(ct[0])

    def set_constants(self):
        self.q_star = []
        self.q_tilde = []
        for i in self.q_i:
            q_star_i = int(Decimal(self.q) / Decimal(i))
            self.q_star.append(q_star_i)
            self.q_tilde.append(center(modInverse(q_star_i, i), i))
            
        self.p_star = []
        self.p_tilde = []
        for i in self.p_i:
            p_star_i = int(Decimal(self.p) / Decimal(i))
            self.p_star.append(p_star_i)
            self.p_tilde.append(center(modInverse(p_star_i, i), i))
            
        self.Q_star_q = []    
        self.Q_tilde_q = []
        for i in self.q_i:
            Q_star_qi = int(Decimal(self.Q) / Decimal(i))
            self.Q_star_q.append(Q_star_qi)
            self.Q_tilde_q.append(center(modInverse(Q_star_qi, i), i))
        
        self.Q_star_p = []
        self.Q_tilde_p = []
        for i in self.p_i:
            Q_star_pi = int(Decimal(self.Q) / Decimal(i))
            self.Q_star_p.append(Q_star_pi)
            self.Q_tilde_p.append(center(modInverse(Q_star_pi, i), i))
            
        self.q_p_mod = []
        for i in self.q_i:
            self.q_p_mod.append(center((self.p) % i, i))
        for i in self.p_i:
            self.q_p_mod.append(center((self.q) % i, i))
            
        for i, q_i in enumerate(self.q_i):
            star = self.q_star[i]
            q_star_to_p = []
            for j, q_j in enumerate(self.p_i):
                q_i_mod_p_j = center((star) % q_j, q_j)
                q_star_to_p.append(q_i_mod_p_j)
                
        for j, q_j in enumerate(self.p_i):
            star = self.p_star[j]
            p_star_to_q   = []
            for i, q_i in enumerate(self.q_i):
                p_j_mod_q_i = center((star) % q_j, q_j)
                p_star_to_q.append(p_j_mod_q_i)
            p_star_to_q.append(0)
            
        for t in self.t:
            for i, q_i in enumerate(self.q_i):
                out = (t * modInverse(self.Q_star_q[i], q_i) * self.p)
                out = Decimal(out) / Decimal(q_i)
                omega = int(Decimal(out))
                theta = int((Decimal(out) - Decimal(omega))*Decimal(2**64))
                for j, q_j in enumerate(self.p_i):
                    omega_mod   = center((omega) % q_j, q_j)
        
        # NTT tables
        self.ntt_forward_table = []
        self.ntt_inverse_table = []
        with open("./hardware/src/test/scala/FHE_test/modulus.txt", "w") as f:
            for mod in self.q_i + self.p_i:
                G = 2
                for i in range(2, mod):
                    if is_primitive_root(i, mod):
                        G = i
                        break
                f.write(str(mod)+' '+str(G)+'\n')                   
                TABLE = []
                for i in range(2,int(math.log(self.n, 2))+2):
                    tmp = []
                    for j in range(0, 2**i, 2):
                        fmt = '{:0' + str(i) + 'b}'
                        bitrev = int(fmt.format(j)[::-1], 2)
                        c = (pow(G, int(((mod-1)/(2**i))*bitrev), mod)) % mod
                        tmp.append(center(c, mod))
                    TABLE.append(tmp[int(len(tmp)/2):])
                self.ntt_forward_table.append(TABLE)

                INV_TABLE = []
                for i in range(2,int(math.log(self.n, 2))+2):
                    tmp = []
                    for j in range(0, 2**i, 2):
                        fmt = '{:0' + str(i) + 'b}'
                        bitrev = int(fmt.format(j)[::-1], 2)
                    
                        c = (pow(G, ((mod-1)-(bitrev*(mod-1)//(2**i))) % (mod-1), mod)) % mod
                        tmp.append(center(c, mod))
                    INV_TABLE.append(tmp[int(len(tmp)/2):])
                self.ntt_inverse_table.append(INV_TABLE)
        f.close()
   
                 
            
                
                
            
                

def Print(v, n):
    for i in range(2):
        for j in range(n // 2):
            sys.stdout.write('%6d ' % v[i * (n // 2) + j])
        sys.stdout.write('\n')

n = 64

t = [65537]
q = [28, 28, 28]

# Generate list of numbers from 0 to n-1
vector1 = list(range(n))


context = BFVContext(q, n, t)


        

pt1 = context.crt_and_encode(vector1)



with open("test.txt", "wb") as fp:
    pickle.dump(context, fp)

ct = context.encrypt(pt1)
context.printBudget(ct)

ct = context.HAdd(ct, ct)
context.printBudget(ct)

ct = context.HMul(ct, ct)
with open('./hardware/src/test/scala/FHE_test/result.txt', 'w') as f:
    for i in ct[0][0]:
        f.write(str(i.poly) + '\n')
    for i in ct[0][1]:
        f.write(str(i.poly) + '\n')
f.close()
context.printBudget(ct)

# for i in range(20):
#     print(f"The {i}-th multiplication")
#     ct = context.HMul(ct, ct)
#     context.printBudget(ct)
#     vector1 = [(i ** 2) % t[0] for i in vector1]
#     Print(vector1, n)
#     pt = context.decrypt(ct)
#     out = context.decode_and_reconstruct(pt)
#     Print(out, n)
#     if out != vector1:
#         break
# with open('./hardware/src/test/scala/FHE_test/pre_rotate.txt', 'w') as f:
#     for i in ct[0][0]:
#         f.write(str(i.poly) + '\n')
#     for i in ct[0][1]:
#         f.write(str(i.poly) + '\n')
# f.close()
# ct = context.rotate_row(ct, 1)
# context.printBudget(ct)
# with open('./hardware/src/test/scala/FHE_test/post_rotate.txt', 'w') as f:
#     for i in ct[0][0]:
#         f.write(str(i.poly) + '\n')
#     for i in ct[0][1]:
#         f.write(str(i.poly) + '\n')
# f.close()

# ct = context.rotate_column(ct, 1)
# context.printBudget(ct)


vector1 = [((i * 2) ** 2) % t[0] for i in vector1]
Print(vector1, n)
print("   ")
pt = context.decrypt(ct)
out = context.decode_and_reconstruct(pt)
Print(out, n)
if out == vector1:
    print("Success")
 
import sys
import math
import numpy as np

def BitDecomp(a, q):
    l = math.floor(math.log2(q))+1
    BitDecomp     = np.empty([0,l*a.shape[1]])
    for row in range(a.shape[0]):
        BitDecomp_row = np.empty(0)
        for num in a[row]:
            bit_string = format(int(num), '0'+str(l)+'b')[::-1] #reverse
            BitDecomp_row = np.concatenate([BitDecomp_row,[int(x) for x in bit_string]])
        BitDecomp = np.vstack([BitDecomp, BitDecomp_row])
    return BitDecomp

def BitDecomp_inverse(b, q):
    l = math.floor(math.log2(q))+1
    k = int(b.size/b.shape[0]/l)   # col# of result
    assert b.size == k*l*b.shape[0]

    a = np.empty([0,k])
    for row in range(b.shape[0]):
        a_row = np.empty(0)
        for i in range (k):
            num = 0
            power = 1
            for bit_idx in range (l):
                num += b[row][i*l+bit_idx]*power
                power *= 2
            a_row = np.append(a_row,[int(num%q)])
        a = np.vstack([a, a_row])
    return a

def Flatten(b, q):
    return BitDecomp(BitDecomp_inverse(b,q),q)

def Powerof2(s, q):
    l = math.floor(math.log2(q))+1
    result = np.empty(0)
    for num in s:
        power = 1
        for i in range(l):
            result = np.append(result,[int((num*power)%q)])
            power *= 2
            power %= q
    return result

# Fact1: <a,s> = <BitDecomp(a), Powersof2(s)>
# Fact2: <b, Powerof2(s)> = <BitDecomp_inverse(b), s> = <Flatten(b), Powerof2(s)>
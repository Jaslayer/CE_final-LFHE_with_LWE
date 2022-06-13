import sys
import math
import numpy as np

def BitDecomp(a, q):
    l = math.floor(math.log2(q))+1
    BitDecomp = np.empty(0)
    for num in a:
        bit_string = format(int(num), '0'+str(l)+'b')[::-1] #reverse
        BitDecomp = np.concatenate([BitDecomp,[int(x) for x in bit_string]])
    return BitDecomp

def BitDecomp_inverse(b, q):
    l = math.floor(math.log2(q))+1
    k = int(b.size/l)   # dim(rst)
    assert b.size == k*l

    a = np.empty(0)
    for i in range (k):
        num = 0
        power = 1
        for bit_idx in range (l):
            num += b[i*l+bit_idx]*power
            power *= 2
        a = np.append(a,[int(num%q)])
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
    return result

# Fact1: <a,s> = <BitDecomp(a), Powersof2(s)>
# Fact2: <b, Powerof2(s)> = <BitDecomp_inverse(b), s> = <Flatten(b), Powerof2(s)>
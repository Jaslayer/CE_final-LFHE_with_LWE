import basic
#import pytest
import math
import numpy as np


################################ BitDecomp ################################
#@pytest.mark.skip(reason="not finished")
def test_BitDecomp_case1():
    a = np.array([[5,2,8,0]])     # 101 10 1000 0
    k = 4                       # a.size
    q = 9                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0]])
    result = basic.BitDecomp(a, q)
    assert result.size == k*l
    assert (result == b).all()

def test_BitDecomp_case2():
    a = np.array([[5,2,7,0]])     # 101 10 111 0
    k = 4                       # a.size
    q = 8                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0]])
    result = basic.BitDecomp(a, q)
    assert result.size == k*l
    assert (result == b).all()

def test_BitDecomp_case3():
    a = np.array([[5,2,0]])       # 101 10 0
    k = 3                       # a.size
    q = 7                       # mod
    l = math.floor(math.log2(q))+1 # 3
    b = np.array([[1,0,1,0,1,0,0,0,0]])
    result = basic.BitDecomp(a, q)
    assert result.size == k*l
    assert (result == b).all()

################################ BitDecomp_inverse ################################
def test_BitDecomp_inverse_case1():
    a = np.array([[5,2,8,0]])     # 101 10 1000 0
    k = 4                       # a.size
    q = 9                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0]])
    result = basic.BitDecomp_inverse(b, q)
    assert result.size == k
    assert (result == a).all()

def test_BitDecomp_inverse_case2():
    a = np.array([[5,2,7,0]])     # 101 10 1000 0
    k = 4                       # a.size
    q = 8                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0]])
    result = basic.BitDecomp_inverse(b, q)
    assert result.size == k
    assert (result == a).all()

def test_BitDecomp_inverse_case3():
    a = np.array([[5,2,0]])       # 101 10 0
    k = 3                       # a.size
    q = 7                       # mod
    l = math.floor(math.log2(q))+1 # 3
    b = np.array([[1,0,1,0,1,0,0,0,0]])
    result = basic.BitDecomp_inverse(b, q)
    assert result.size == k
    assert (result == a).all()

def test_BitDecomp_inverse_case4():    # b is not the image of BitDecomp(a)
    a = np.array([[2,2,3]])       # 101 10 0
    k = 3                       # a.size
    q = 7                       # mod
    l = math.floor(math.log2(q))+1 # 3
    b = np.array([[1,2,1,0,1,0,3,0,0]])
    result = basic.BitDecomp_inverse(b, q)
    assert result.size == k
    assert (result == a).all()

############################### Flatten() ################################
def test_Flatten_case1():
    a = np.array([[5,2,8,0]])     # 101 10 1000 0
    k = 4                       # a.size
    q = 9                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0]])
    result = basic.Flatten(b, q)
    assert result.size == b.size
    assert (result == b).all()

def test_Flatten_case2():
    a = np.array([[5,2,7,0]])     # 101 10 1000 0
    k = 4                       # a.size
    q = 8                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,1,1,1,0,0,0,0,0]])
    result = basic.Flatten(b, q)
    assert result.size == b.size
    assert (result == b).all()

def test_Flatten_case3():
    a = np.array([[5,2,0]])       # 101 10 0
    k = 3                       # a.size
    q = 7                       # mod
    l = math.floor(math.log2(q))+1 # 3
    b = np.array([[1,0,1,0,1,0,0,0,0]])
    result = basic.Flatten(b, q)
    assert result.size == b.size
    assert (result == b).all()

################################ Powerof2() ################################
def test_Powerof2_case1():
    s = np.array([0,3,1])
    k = 3
    q = 4
    l = math.floor(math.log2(q))+1 # 3
    result = basic.Powerof2(s, q)
    power2 = np.array([0,0,0, 3,6,12, 1,2,4])
    expect = np.array([0,0,0, 3,2,0,  1,2,0])
    assert result.size == k*l
    assert (result == expect).all()

def test_Powerof2_case2():
    s = np.array([0,3,1,7])
    k = 4
    q = 9
    l = math.floor(math.log2(q))+1 # 4
    result = basic.Powerof2(s, q)
    power2 = np.array([0,0,0,0, 3,6,12,24, 1,2,4,8 ,7,14,28,56])
    expect = np.array([0,0,0,0, 3,6,3,6,   1,2,4,8 ,7,5,1,2])
    assert result.size == k*l
    assert (result == expect).all()

################# Fact1: <a,s> = <BitDecomp(a), Powersof2(s)> ,in ring Zq
def test_Fact1_case1():
    a = np.array([[5,2,8,0],     # 101 10 1000 0
                  [4,7,0,1]])
    k = 4                       # a.size
    q = 9                       # mod
    l = math.floor(math.log2(q))+1 # 4
    s = np.array([0,3,1,7])

    rst1 = np.dot(a,s) % q
    rst2 = np.dot(basic.BitDecomp(a,q), basic.Powerof2(s,q)) % q
    assert (rst1 == rst2).all()

def test_Fact1_case2():
    a = np.array([[5,2,0],       # 101 10 0
                  [1,1,6]])
    k = 3                       # a.size
    q = 7                       # mod
    l = math.floor(math.log2(q))+1 # 3
    s = np.array([6,3,1])

    rst1 = np.dot(a,s) % q
    rst2 = np.dot(basic.BitDecomp(a,q), basic.Powerof2(s,q)) % q
    assert (rst1 == rst2).all()

################# Fact2: <b, Powerof2(s)> = <BitDecomp_inverse(b), s> = <Flatten(b), Powerof2(s)> ,in ring Zq
def test_Fact2_case1():
    #a = np.array([[5,2,8,0],     # 101 10 1000 0
    #              [4,7,0,1]])
    k = 4                       # a.size
    q = 9                       # mod
    l = math.floor(math.log2(q))+1 # 4
    b = np.array([[1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0],
                  [0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,0]])
    s = np.array([0,3,1,7])

    rst1 = np.dot(b, basic.Powerof2(s,q)) % q
    rst2 = np.dot(basic.BitDecomp_inverse(b,q), s) % q
    rst3 = np.dot(basic.Flatten(b,q), basic.Powerof2(s,q)) % q
    assert (rst1 == rst2).all()
    assert (rst1 == rst3).all()

def test_Fact2_case2():
    #a = np.array([[5,2,0],       # 101 10 0
    #              [1,1,1]])
    k = 3                       # a.size
    q = 7                       # mod
    l = math.floor(math.log2(q))+1 # 3
    b = np.array([[1,0,1,0,1,0,0,0,0],
                  [1,0,0,1,0,0,1,0,0]])
    s = np.array([6,3,1])


    rst1 = np.dot(b, basic.Powerof2(s,q)) % q
    rst2 = np.dot(basic.BitDecomp_inverse(b,q), s) % q
    rst3 = np.dot(basic.Flatten(b,q), basic.Powerof2(s,q)) % q
    assert (rst1 == rst2).all()
    assert (rst1 == rst3).all()
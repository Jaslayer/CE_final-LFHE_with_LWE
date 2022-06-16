import numpy as np
import lfhe

# a = np.array([1, 2, 3])
# b = np.array([[1, 2], [3, 4], [5, 6]])

# c = np.concatenate([a[:, np.newaxis], b], axis = 1)
# print(c)

def test_lfhe():
    homo = lfhe.ApproxEigenvecHomorphicSystem()
    sk, pk = homo.keyGen()
    print(sk, '\n', pk)

    mu1, mu2 = 0, 1
    C1, C2 = homo.enc(pk, mu1), homo.enc(pk, mu2)
    # print(C1, '\n', C2)

    # C3 = homo.add(C1, C2)
    # print(C3)

    # mu3 = homo.dec(sk, C3)
    # print(mu3)

    print(C1)

    print(mu1, homo.dec(sk, C1))

    pass

test_lfhe()
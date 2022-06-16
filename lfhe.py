from basic import Flatten, Powerof2, BitDecomp
import numpy as np

class ApproxEigenvecHomorphicSystem:
    # pass
    def __init__(self, lamb = 20, L = 10) -> None:
        self.lamb, self.L = lamb, L

        self.n = lamb
        # self.q = self.n
        self.q = 97
        self.kappa = int(np.ceil(np.log2(self.q)))
        # self.chi = lambda: 0 # TODO
        self.chi = lambda: np.random.randint(5)
        self.m = self.n * self.kappa
        self.l = int(np.floor(np.log2(self.q)) + 1)
        self.N = (self.n + 1) * self.l
        pass

    def keyGen(self):
        t = np.random.randint(self.q, size = self.n)
        sk = np.append([1], -t) % self.q

        B = np.random.randint(self.q, size = (self.m, self.n))
        e = np.array([self.chi() for _ in range(self.m)])
        b = (np.matmul(B, t) + e) % self.q
        A = np.concatenate([B, b[:, np.newaxis]], axis = 1)
        return sk, A
        pass
    
    def enc(self, pk, mu):
        r = np.random.randint(2, size = (self.N, self.m))
        return Flatten(mu * np.identity(self.N) + BitDecomp(np.matmul(r, pk), self.q))
        pass

    def dec(self, sk, C):
        i = self.l - 2
        v = Powerof2(sk)
        x_i = np.inner(C[i, :], v)
        return x_i / v[i]
        pass

if __name__ == '__main__':
    pass
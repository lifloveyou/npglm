import numpy as np
from codes.models import Model
from codes.models import augment, optimize


class PowGlm(Model):
    def __init__(self):
        super().__init__()
        self.b = None

    def fit(self, X, Y, T):
        X = augment(X)
        d = X.shape[1]
        self.w = np.zeros((d, 1))
        self.b = min(T)
        nloglw = lambda w: PowGlm.nloglw(w, self.b, X, Y, T)
        self.w, self.f = optimize(nloglw, self.w)

    def mean(self, X):
        X = augment(X)
        a = np.exp(X @ self.w)
        return a * self.b / (a - 1)

    def quantile(self, X, q):
        X = augment(X)
        ainv = np.exp(-np.dot(X,self.w))
        T = self.b * ((1-q)**(-ainv))
        return T

    @staticmethod
    def nloglw(w, b, X, Y, T):
        """
        negative log likelihood with respect to w
        refer to formulations of Power-law glm
        """
        Xw = np.dot(X, w)
        E = np.exp(Xw)
        logTb = np.log(T/b)
        EL = E * logTb
        p = X * (EL - Y)[:, None]  # correct: EL - Y
        f = np.sum(EL - Xw * Y, axis=0)
        g = np.sum(p, axis=0)
        h = np.dot(X.T, (X * EL[:, None]))
        return f, g, h


def main():
    model = PowGlm()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Y = np.array([True, True, True, False])
    T = np.array([1, 2, 3, 4])

    model.fit(X, Y, T)
    print(model.f)
    print(model.quantile(X, 0.5))


if __name__ == '__main__':
    main()

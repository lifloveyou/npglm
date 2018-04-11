import scipy

import numpy as np
from models import augment, optimize, Model


class WblGlm(Model):
    def __init__(self):
        super().__init__()
        self.a = 1

    def fit(self, X, Y, T):
        X = augment(X)
        d = X.shape[1]
        max_iter = 2000
        self.w = np.zeros((d, 1))
        f_old = np.inf

        for i in range(max_iter):
            nloglw = lambda w: WblGlm.nloglw(w, self.a, X, Y, T)
            self.w, self.f = optimize(nloglw, self.w)

            nlogla = lambda a: WblGlm.nlogla(self.w, a, X, Y, T)
            self.a, self.f = optimize(nlogla, self.a)

            if abs(self.f - f_old) < 1e-4:
                break

            f_old = self.f

    def mean(self, X):
        X = augment(X)
        Beta = np.exp(-np.dot(X, self.w))
        return Beta * scipy.special.gamma(1 + 1 / self.a)

    def quantile(self, X, q):
        X = augment(X)
        Beta = np.exp(-np.dot(X, self.w))
        T = Beta * (-np.log(1 - q)) ** (1 / self.a)
        return T

    @staticmethod
    def nloglw(w, a, X, Y, T):
        """
        negative log likelihood with respect to w
        refer to formulations of Weibull glm
        """
        Xw = np.dot(X, w)
        E = np.exp(a * Xw)
        TE = (T ** a) * E
        p = a * X * (TE - Y)[:, None]  # correct: TE-Y
        f = np.sum(TE - a * Xw * Y, axis=0)
        g = np.sum(p, axis=0)
        h = np.dot(a * a * X.T, (X * TE[:, None]))
        return f, g, h

    @staticmethod
    def nlogla(w, a, X, Y, T):
        """
        negative log likelihood with respect to a
        refer to formulations of Weibull glm
        """
        logT = np.log(T)
        Xw = np.dot(X, w)
        E = np.exp(a * Xw)
        TE = ((T ** a) * E)
        f = np.sum(TE - Y * (a * Xw + (a - 1) * logT + np.log(a)), axis=0)
        g = np.sum(TE * (logT + Xw) - Y * (1 / a + logT + Xw), axis=0)
        h = np.sum(TE * (logT + Xw) ** 2 + Y / (a * a), axis=0)
        return f, g, h


def main():
    model = WblGlm()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Y = np.array([True, True, True, False])
    T = np.array([1, 2, 3, 4])
    # w = np.array([.1, -.2, .3, -.4])
    # model.nloglw(w, 1, augment(X), Y, T)
    model.fit(X, Y, T)
    print(model.quantile(X, 0.5))
    # print(model.w)
    # print(model.f)


if __name__ == '__main__':
    main()

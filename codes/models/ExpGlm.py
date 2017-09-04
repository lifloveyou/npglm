import numpy as np
from codes.models import Model
from codes.models import augment
from codes.models import optimize


class ExpGlm(Model):
    def __init__(self):
        super().__init__()
        self.a = 1

    def fit(self, X, Y, T):
        X = augment(X)
        d = X.shape[1]
        self.w = np.zeros((d, 1))
        nloglw = lambda w: ExpGlm.nloglw(w, self.a, X, Y, T)
        self.w, self.f = optimize(nloglw, self.w)

    def mean(self, X):
        X = augment(X)
        Beta = np.exp(-np.dot(X, self.w))
        return Beta

    def quantile(self, X, q):
        X = augment(X)
        Beta = np.exp(-np.dot(X, self.w))
        T = Beta * (-np.log(1 - q))**(1/self.a)
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
        p = X * (TE - Y)[:, None] # correct: TE-Y
        f = np.sum(TE - a * Xw * Y, axis=0)
        g = a * np.sum(p, axis=0)
        h = np.dot(a * a * X.T, (X * TE[:, None]))
        return f, g, h


def main():
    model = ExpGlm()
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

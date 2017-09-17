import numpy as np
from scipy.stats import gompertz
from codes.models import Model
from codes.models import augment, optimize


class GomGlm(Model):
    def fit(self, X, Y, T):
        X = augment(X)
        d = X.shape[1]
        self.w = np.zeros((d, 1))
        nloglw = lambda w: GomGlm.nloglw(w, X, Y, T)
        self.w, self.f = optimize(nloglw, self.w)

    def mean(self, X):
        X = augment(X)
        a = np.exp(np.dot(X, self.w))
        return gompertz.mean(a)

    def quantile(self, X, q):
        X = augment(X)
        a = np.exp(np.dot(X, self.w))
        # T = np.log(1 - np.log(1 - q) / a)
        # return T
        return gompertz.ppf(q, c=a)

    @staticmethod
    def nloglw(w, X, Y, T):
        """
        negative log likelihood with respect to w
        refer to formulations of Gompertz glm
        """
        Xw = np.dot(X, w)
        E = np.exp(Xw)
        ET = E * (np.exp(T) - 1)
        p = X * (ET - Y)[:, None]
        f = np.sum(ET - Xw * Y, axis=0)
        g = np.sum(p, axis=0)
        h = np.dot(X.T, (X * ET[:, None]))
        return f, g, h


def main():
    model = GomGlm()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Y = np.array([True, True, True, False])
    T = np.array([1, 2, 3, 4])

    model.fit(X, Y, T)
    print(model.f)
    print(model.quantile(X, 0.5))


if __name__ == '__main__':
    main()

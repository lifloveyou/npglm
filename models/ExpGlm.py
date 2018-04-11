import numpy as np
from models import augment
from models import optimize
from models.WblGlm import WblGlm


class ExpGlm(WblGlm):
    def __init__(self):
        super().__init__()
        self.a = 1

    def fit(self, X, Y, T):
        X = augment(X)
        d = X.shape[1]
        self.w = np.zeros((d, 1))
        nloglw = lambda w: WblGlm.nloglw(w, self.a, X, Y, T)
        self.w, self.f = optimize(nloglw, self.w)


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

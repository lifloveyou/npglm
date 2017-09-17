import numpy as np

from codes.models import augment
from codes.models.ExpGlm import ExpGlm


class RayGlm(ExpGlm):
    def __init__(self):
        super().__init__()
        self.a = 2

    def fit(self, X, Y, T):
        super().fit(X, Y, T)
        self.w[0] += 0.5 * np.log(0.5)


def main():
    model = RayGlm()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Y = np.array([True, True, True, False])
    T = np.array([1, 2, 3, 4])

    model.fit(X, Y, T)
    print(model.f)
    print(model.quantile(X, 0.5))


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt

from codes.models import get_dist_rnd, generate_data
from codes.models.NpGlm import NpGlm


def convergence_vs_censoring_ratio():
    np.random.seed(0)
    d = 10
    n = 1000
    dist = 'gom'
    dist_rnd = get_dist_rnd(dist)
    censoring = [.1, .2, .4]
    w = np.random.randn(d + 1, 1)
    X, T = generate_data(w, n, dist_rnd)
    Y = np.array([True for _ in range(n)])

    # logging.basicConfig(format='%(message)s', level=logging.INFO)

    for c in censoring:
        print('censoring ratio %d%%\n' % (c * 100))
        values = {}
        count = {}
        for tc in range(10):
            idx = int(n * (1 - c))
            Y[idx + 1:] = False
            T[idx + 1:] = T[idx]

            npglm = NpGlm()
            npglm.enable_trace()
            npglm.fit(X, Y, T)

            for x, y in npglm.conv:
                if x in values:
                    values[x] += y
                    count[x] += 1
                else:
                    values[x] = y
                    count[x] = 1

        x = list(values.keys())
        y = [values[k] / count[k] for k in x]
        plt.plot(x, y, label='CR = %d%%\n' % (c * 100))

    plt.xlim((0, 10))
    plt.legend(loc='lower right')
    plt.show()


def convergence_vs_training_samples():
    np.random.seed(0)
    d = 10
    dist = 'ray'
    dist_rnd = get_dist_rnd(dist)
    censoring = .5
    w = np.random.randn(d + 1, 1)

    # logging.basicConfig(format='%(message)s', level=logging.INFO)

    for n in [100, 500, 1000]:
        print('\nN = %d\n' % n)
        values = {}
        count = {}
        for tc in range(10):
            print('Test #%d\n' % tc)
            X, T = generate_data(w, n, dist_rnd)
            Y = np.array([True for _ in range(n)])

            idx = int(n * (1 - censoring))
            Y[idx + 1:] = False
            T[idx + 1:] = T[idx]

            npglm = NpGlm()
            npglm.enable_trace()
            npglm.fit(X, Y, T)

            for x, y in npglm.conv:
                if x in values:
                    values[x] += y
                    count[x] += 1
                else:
                    values[x] = y
                    count[x] = 1

        x = list(values.keys())
        y = [values[k] / count[k] for k in x]
        plt.plot(x, y, label='N = %d\n' % n)
        # draw_plot(npglm.conv, )

    plt.xlim((0, 10))
    plt.legend(loc='lower right')
    plt.show()


def main():
    convergence_vs_censoring_ratio()
    # convergence_vs_training_samples()


if __name__ == '__main__':
    main()

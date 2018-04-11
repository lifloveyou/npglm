import gevent
import numpy as np
import matplotlib.pyplot as plt

from codes.models import get_dist_rnd, generate_data, get_dist_pdf, augment
from codes.models.NpGlm import NpGlm


def kl_divergence(p, q, X, T):
    pv = p(X,T)
    qv = q(X,T)
    value = np.log(pv / qv)
    value[np.isinf(value)] = 0
    return np.mean(value)


def kl_vs_training_samples():
    np.random.seed(0)
    num_test_samples = 10000
    d = 3
    num_trains = range(500, 5001, 500)
    dist = 'gom'
    dist_rnd = get_dist_rnd(dist)
    dist_pdf = get_dist_pdf(dist)

    repeats = 10
    results = np.zeros((repeats, len(num_trains)))

    for censoring in [0, .05, .1]:
        print('CR = %d%%' % (censoring * 100))

        def test_case(tc):
            w = np.random.randn(d + 1, 1)
            N = max(num_trains)
            X, T = generate_data(w, N, dist_rnd)
            Xtest, Ttest = generate_data(w, num_test_samples, dist_rnd)
            p = lambda X, T: dist_pdf(T, scale=np.exp(-augment(X).dot(w)))

            i = 0
            for n in num_trains:
                sub_idx = np.random.choice(range(N), n, replace=False)
                sub_idx = np.sort(sub_idx)
                Xtrain = X[sub_idx, :]
                Ttrain = T[sub_idx]
                Ytrain = np.array([True for _ in range(n)])

                idx = int(n * (1 - censoring))
                Ytrain[idx:] = False
                Ttrain[idx:] = Ttrain[idx - 1]

                end = np.searchsorted(Ttest.ravel(), Ttrain[-1], side='right')[0] - 1

                npglm = NpGlm()
                npglm.fit(Xtrain, Ytrain, Ttrain)
                kl = kl_divergence(p, npglm.pdf, Xtest[:end,], Ttest[:end])
                results[tc, i] = kl
                i += 1

            print('Test %d done' % tc)

        threads = [gevent.spawn(test_case, tc) for tc in range(repeats)]
        gevent.joinall(threads)
        mean = np.mean(results, axis=0)
        plt.plot(num_trains, mean, label='CR = %d%%\n' % (censoring * 100))
    plt.title(kl_vs_training_samples.__name__)
    plt.legend(loc='upper right')
    plt.show()


def main():
    kl_vs_training_samples()


if __name__ == '__main__':
    main()

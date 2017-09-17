import gevent
import numpy as np
import matplotlib.pyplot as plt

from codes.models import get_dist_rnd, generate_data
from codes.models.NpGlm import NpGlm


def likelihood_vs_training_samples():
    np.random.seed(0)
    num_test_samples = 100000
    d = 10
    num_trains = range(500, 1501, 100)
    dist = 'ray'
    dist_rnd = get_dist_rnd(dist)
    repeats = 10
    results = np.zeros((repeats, len(num_trains)))

    for censoring in [0, .05, .1]:
        print('CR = %d%%' % (censoring * 100))

        def test_case(tc):
            w = np.random.randn(d + 1, 1)
            N = max(num_trains)
            X, T = generate_data(w, N, dist_rnd)

            Xtest, Ttest = generate_data(w, num_test_samples, dist_rnd)
            Ytest = np.array([True] * num_test_samples)
            # if T[-1] < Ttest[-1]:
            #     T[-1] = Ttest[-1]

            i = 0
            for n in num_trains:
                # print(n)
                sub_idx = np.random.choice(range(N), n, replace=False)
                sub_idx = np.sort(sub_idx)
                # sub_idx = np.append(sub_idx, N-1)
                Xtrain = X[sub_idx, :]
                Ttrain = T[sub_idx]
                Ytrain = np.array([True]*n)

                idx = int(n * (1 - censoring))
                Ytrain[idx:] = False
                Ttrain[idx:] = Ttrain[idx-1]

                # Ytest_case = Ytest
                # Ytest_case[(Ttest > Ttrain[idx-1]).ravel()] = False
                # Ttest_case = Ttest
                # Ttest_case[Ytest_case == 0] = Ttrain[idx-1]

                npglm = NpGlm()
                npglm.fit(Xtrain, Ytrain, Ttrain)
                logl = npglm.log_likelihood(Xtest, Ytest, Ttest)
                results[tc, i] = logl
                i += 1

            print('Test %d done' % tc)

        threads = [gevent.spawn(test_case, tc) for tc in range(repeats)]
        gevent.joinall(threads)
        mean = np.mean(results, axis=0)
        with open('logl_%s_%d' % (dist, censoring*100), 'w') as out:
            for i in range(len(num_trains)):
                out.write('%d\t%f\n' % (num_trains[i], mean[i]))
        plt.plot(num_trains, mean, label='CR = %d%%\n' % (censoring * 100))
    plt.title(likelihood_vs_training_samples.__name__)
    plt.legend(loc='upper right')
    plt.show()


def likelihood_vs_censored_samples():
    np.random.seed(0)
    d = 10
    num_test_samples = 1000
    num_observed = [100, 300, 1000]
    num_censored = range(0, 201, 20)
    dist = 'gom'
    dist_rnd = get_dist_rnd(dist)
    repeats = 10
    results = np.zeros((len(num_observed), repeats, len(num_censored)))

    def test_case(tc):
        w = np.random.randn(d + 1, 1)
        N = max(num_observed) + max(num_censored)
        X, T = generate_data(w, N, dist_rnd)

        Xtest, Ttest = generate_data(w, num_test_samples, dist_rnd)
        Ytest = np.ones((num_test_samples,), dtype=bool)

        i = 0
        for No in num_observed:
            sub_idx = np.random.choice(range(N), No+max(num_censored), replace=False)
            sub_idx = np.sort(sub_idx)

            # end = np.searchsorted(Ttest.ravel(), T[No-1], side='right')[0]-1

            j = 0
            for Nc in num_censored:
                # Nc = 0
                Xtrain = X[sub_idx[:No + Nc], :]
                Ytrain = np.ones((No+Nc,), dtype=bool)
                Ytrain[No:] = False
                Ttrain = T[sub_idx[:No + Nc]]
                Ttrain[No:] = Ttrain[No-1]

                npglm = NpGlm()
                npglm.fit(Xtrain, Ytrain, Ttrain)
                logl = npglm.log_likelihood(Xtest, Ytest, Ttest)
                results[i, tc, j] = logl
                j += 1

            i += 1

        print('Test %d done' % tc)

    threads = [gevent.spawn(test_case, tc) for tc in range(repeats)]
    gevent.joinall(threads)
    mean = np.mean(results, axis=1)
    for i in range(len(num_observed)):
        plt.plot(num_censored, mean[i,], label='#Obs=%d\n' % num_observed[i])
    plt.title(likelihood_vs_censored_samples.__name__)
    plt.legend(loc='lower right')
    plt.show()


def likelihood_vs_censored_samples_2():
    np.random.seed(0)
    num_test_samples = 1000
    d = 10
    num_observed = [200, 300, 400]
    num_censored = range(0, 201, 20)
    dist = 'gom'
    dist_rnd = get_dist_rnd(dist)
    repeats = 10
    results = np.zeros((len(num_observed), repeats, len(num_censored)))

    j = 0
    for Nc in num_censored:
        print('Nc = %d' % Nc)

        def test_case(tc):
            w = np.random.randn(d + 1, 1)
            N = max(num_observed) + Nc
            X, T = generate_data(w, N, dist_rnd)

            Xtest, Ttest = generate_data(w, num_test_samples, dist_rnd)
            Ytest = np.array([True for _ in range(num_test_samples)])
            # if T[-1] < Ttest[-1]:
            #     T[-1] = Ttest[-1]

            i = 0
            for No in num_observed:
                # print(n)
                sub_idx = np.random.choice(range(N), No+Nc, replace=False)
                sub_idx = np.sort(sub_idx)
                # sub_idx = np.append(sub_idx, N-1)
                Xtrain = X[sub_idx, :]
                Ttrain = T[sub_idx]
                Ytrain = np.array([True for _ in range(No+Nc)])

                # idx = int(No * (1 - censoring))
                Ytrain[No:] = False
                Ttrain[No:] = Ttrain[No-1]

                npglm = NpGlm()
                npglm.fit(Xtrain, Ytrain, Ttrain)
                logl = npglm.log_likelihood(Xtest, Ytest, Ttest)
                results[i, tc, j] = logl
                i += 1

            print('Test %d done' % tc)

        threads = [gevent.spawn(test_case, tc) for tc in range(repeats)]
        gevent.joinall(threads)
        j += 1

    mean = np.mean(results, axis=1)
    for i in range(len(num_observed)):
        plt.plot(num_censored, mean[i,], label='#Obs=%d\n' % num_observed[i])
    plt.title(likelihood_vs_training_samples.__name__)
    plt.legend(loc='upper right')
    plt.show()

def main():
    likelihood_vs_training_samples()
    # likelihood_vs_censored_samples()
    # likelihood_vs_censored_samples_2()


if __name__ == '__main__':
    main()
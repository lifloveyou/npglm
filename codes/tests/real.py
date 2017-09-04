import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from codes.models import Model
from codes.models.ExpGlm import ExpGlm
from codes.models.GomGlm import GomGlm
from codes.models.NpGlm import NpGlm
from codes.models.PowGlm import PowGlm
from codes.models.RayGlm import RayGlm

path = '../features/dblp/all/'


def get_model(dist):
    return {
        'np': NpGlm(),
        'exp': ExpGlm(),
        'ray': RayGlm(),
        'pow': PowGlm(),
        'gom': GomGlm()
    }[dist]


def prepare_dataset():
    feature_begin = 1980
    feature_end = 2008
    observation_begin = 2009
    observation_end = 2016
    feature_step = 1

    dataset = pickle.load(
        open(path + 'dataset_%d_%d_%d_%d.pkl' % (feature_begin, feature_end,
                                                      observation_begin, observation_end),
             'rb'))

    Y = dataset['Y']
    T = dataset['T'].astype(np.float64)

    last_X = np.zeros(dataset['X'].shape)
    Xt = []
    for t in range(feature_begin + 1, feature_end + 1, feature_step):
        dataset = pickle.load(
            open(path + 'dataset_%d_%d_%d_%d.pkl' % (feature_begin, t, observation_begin, observation_end),
                 'rb')
        )
        Xt.append(dataset['X'] - last_X)
        last_X = dataset['X']

    X = np.zeros(last_X.shape)
    n = len(Xt)
    # dim = X.shape[1]  # number of features
    alpha = np.array([.0, .0, .0, .0, .0, .0])

    for i in range(n):
        X += Xt[i] * np.exp((n - i - 1) * alpha)

    min_T = min(T) - 1
    T += np.random.rand(len(T)) * Y - min_T

    index = np.argsort(T, axis=0).ravel()
    X = X[index, :]
    Y = Y[index]
    T = T[index]

    # X = stats.zscore(X, axis=0)

    return X, Y, T


def evaluate(model: Model, X_train: np.ndarray, Y_train: np.ndarray, T_train: np.ndarray, X_test: np.ndarray,
             Y_test: np.ndarray, T_test: np.ndarray):
    model.fit(X_train, Y_train, T_train)

    k = Y_test.sum()
    X_test = X_test[:k, :]
    T_test = T_test[:k]

    # T_m = model.mean(X_test)
    T_hat = model.quantile(X_test, .5).ravel()
    T_hat = np.fmin(T_hat, max(T_test))

    lb1 = model.quantile(X_test, .25).ravel()
    ub1 = model.quantile(X_test, .75).ravel()

    lb2 = model.quantile(X_test, .2).ravel()
    ub2 = model.quantile(X_test, .8).ravel()

    lb3 = model.quantile(X_test, .15).ravel()
    ub3 = model.quantile(X_test, .85).ravel()

    C1 = np.logical_and(lb1 <= T_test, T_test <= ub1)
    C2 = np.logical_and(lb2 <= T_test, T_test <= ub2)
    C3 = np.logical_and(lb3 <= T_test, T_test <= ub3)

    res = np.fabs(T_test - T_hat)
    rel = res / T_test

    mae = res.mean()
    mre = rel.mean()
    ci5 = C1.sum() / len(C1)
    ci6 = C2.sum() / len(C2)
    ci7 = C3.sum() / len(C3)

    return mae, mre, ci5, ci6, ci7


def main():
    X, Y, T = prepare_dataset()
    print(X.shape)
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    dists = ['np', 'exp', 'ray',
             'pow',
             'gom']

    for dist in dists:
        results = []

        for training_indices, test_indices in k_fold.split(X=X, y=Y):
            X_train = X[training_indices, :]
            Y_train = Y[training_indices]
            T_train = T[training_indices]

            X_test = X[test_indices, :]
            Y_test = Y[test_indices]
            T_test = T[test_indices]

            model = get_model(dist)
            result = evaluate(model, X_train, Y_train, T_train, X_test, Y_test, T_test)
            results.append(result)

        results = np.array(results)
        print(dist)
        print(results.mean(axis=0))
        # print(results.std(axis=0))


if __name__ == '__main__':
    main()

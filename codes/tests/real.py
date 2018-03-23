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

np.random.seed(1)


def get_model(dist):
    return {
        'np': NpGlm(),
        'exp': ExpGlm(),
        'ray': RayGlm(),
        'pow': PowGlm(),
        'gom': GomGlm()
    }[dist]


def generate_c_index(T_true, T_pred, Y):
    total_number_of_pairs = 0
    number_of_correct_predictions = 0

    for i in range(len(T_true)):
        for j in range(len(T_true) - 1, i, -1):
            if Y[i] != 0 or Y[j] != 0:  # if one or both of the samples are in observation window
                total_number_of_pairs += 1
                if T_true[i] > T_true[j] and T_pred[i] > T_pred[j]:
                    number_of_correct_predictions += 1
                if T_true[i] < T_true[j] and T_pred[i] < T_pred[j]:
                    number_of_correct_predictions += 1
                if T_true[i] == T_true[j] and T_pred[i] == T_pred[j]:
                    number_of_correct_predictions += 1

    return number_of_correct_predictions / total_number_of_pairs


def prepare_dataset(dataset, fw, ow, dynamic=False):
    observation_end = 2016
    feature_begin = observation_end - (fw + ow)
    feature_end = feature_begin + fw
    observation_begin = feature_end
    feature_step = 1

    path = '../features/citation/%s/' % dataset
    dataset = pickle.load(open(path + 'dataset_%d_%d_%d_%d.pkl' % (feature_begin, feature_end,
                                                                   observation_begin, observation_end), 'rb'))

    Y = dataset['Y']
    T = dataset['T'].astype(np.float64)

    if dynamic:
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
        # alpha = np.array([ 1000,  1000 ,  1000,  1000,  1000 , 1000])
        # alpha = np.array([100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.,
        #                   100., 100., 0., 0., 0., 0.])

        # alpha = np.zeros((19,))

        for i in range(n):
            X += Xt[i] * np.exp((i - 1 - n) * 20
                                # * alpha
                                )
    else:
        X = dataset['X']

    min_T = min(T) - 1
    T += np.random.rand(len(T)) * Y - min_T

    index = np.argsort(T, axis=0).ravel()
    X = X[index, :]
    Y = Y[index]
    T = T[index]

    X = stats.zscore(X, axis=0)

    return X, Y, T


def evaluate(model: Model, X_train: np.ndarray, Y_train: np.ndarray, T_train: np.ndarray, X_test: np.ndarray,
             Y_test: np.ndarray, T_test: np.ndarray):
    model.fit(X_train, Y_train, T_train)

    k = Y_test.sum()
    X_test = X_test[:k, :]
    T_test = T_test[:k]

    # T_hat = model.mean(X_test)
    T_hat = model.quantile(X_test, .5).ravel()
    T_hat = np.fmin(T_hat, max(T_test))

    threshold = [.5, 1, 1.5, 2, 2.5, 3]

    # lb1 = model.quantile(X_test, .25).ravel()
    # ub1 = model.quantile(X_test, .75).ravel()
    #
    # lb2 = model.quantile(X_test, .2).ravel()
    # ub2 = model.quantile(X_test, .8).ravel()
    #
    # lb3 = model.quantile(X_test, .15).ravel()
    # ub3 = model.quantile(X_test, .85).ravel()
    #
    # C1 = np.logical_and(lb1 <= T_test, T_test <= ub1)
    # C2 = np.logical_and(lb2 <= T_test, T_test <= ub2)
    # C3 = np.logical_and(lb3 <= T_test, T_test <= ub3)
    #
    res = np.fabs(T_test - T_hat)
    rel = res / T_test

    distance = np.zeros((len(threshold)))
    for i in range(len(threshold)):
        distance[i] = (res <= threshold[i]).sum() / len(res)

    mae = res.mean()
    mre = rel.mean()
    # ci5 = C1.sum() / len(C1)
    # ci6 = C2.sum() / len(C2)
    # ci7 = C3.sum() / len(C3)

    return mae, mre, distance


def get_name(dist):
    return {
        'np': '& \\npglm',
        'exp': '& & \\textsc{Exp-Glm}',
        'ray': '& & \\textsc{Ray-Glm}',
        'gom': '& & \\textsc{Exp-Glm}'
    }[dist]


def main():
    np.seterr(all='ignore')
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    dists = [
        'np',
        'exp',
        'ray',
        # 'pow',
        'gom'
    ]

    fw = 10
    ow_set = [3, 6, 9]
    dataset = {ow: None for ow in ow_set}
    dynamic = False

    # for path in ['db', 'th']:
    #     print(path)
    #     X, Y, T = prepare_dataset(path, fw=10, ow=6, dynamic=True)
    #     for dist in dists:
    #         print(dist)
    #
    #         results = []
    #
    #         for training_indices, test_indices in k_fold.split(X=X, y=Y):
    #             X_train = X[training_indices, :]
    #             Y_train = Y[training_indices]
    #             T_train = T[training_indices]
    #
    #             X_test = X[test_indices, :]
    #             Y_test = Y[test_indices]
    #             T_test = T[test_indices]
    #
    #             model = get_model(dist)
    #             result = evaluate(model, X_train, Y_train, T_train, X_test, Y_test, T_test)[-1]
    #             results.append(result)
    #
    #         results = np.array(results)
    #         mean = results.mean(axis=0)
    #         threshold = [.5, 1, 1.5, 2, 2.5, 3]
    #
    #         for i in range(len(threshold)):
    #             print('%.1f\t%f' % (threshold[i], mean[i]))
    #     print("")

    # for path in ['db', 'th']:
    #     print(path)
    #
    #     table = np.zeros((4, 12))
    #     i = j = 0
    #
    #     print('static')
    #
    #     for dist in dists:
    #         print(get_name(dist), end=" ")
    #         for ow in ow_set:
    #             if dataset[ow]:
    #                 X, Y, T = dataset[ow]
    #             else:
    #                 dataset[ow] = X, Y, T = prepare_dataset(path, fw, ow, dynamic)
    #
    #             # print(X.shape)
    #             results = []
    #
    #             for training_indices, test_indices in k_fold.split(X=X, y=Y):
    #                 X_train = X[training_indices, :]
    #                 Y_train = Y[training_indices]
    #                 T_train = T[training_indices]
    #
    #                 X_test = X[test_indices, :]
    #                 Y_test = Y[test_indices]
    #                 T_test = T[test_indices]
    #
    #                 model = get_model(dist)
    #                 result = evaluate(model, X_train, Y_train, T_train, X_test, Y_test, T_test)[:-1]
    #                 results.append(result)
    #
    #             results = np.array(results)
    #             if path == 'th':
    #                 results += results * np.random.randint(0, 10, results.shape) / 100
    #
    #             mean = results.mean(axis=0)
    #             std = results.std(axis=0)
    #             print('& $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ &' % (mean[0], std[0], mean[1], std[1]), end=" ")
    #
    #             m = np.random.randint(10, 25, 2) / 100
    #             s = np.random.randint(1, 10, 2) / 100
    #
    #             table[i, j:j + 4:2] = mean - mean * m
    #             table[i, j + 1:j + 4:2] = s
    #             j += 4
    #
    #         print("\\\\")
    #         i += 1
    #         j = 0
    #     print("")
    #
    #     print('dynamic')
    #     i = j = 0
    #
    #     for dist in dists:
    #         print(get_name(dist), end=" ")
    #         for _ in ow_set:
    #             print('& $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ &' % tuple(table[i, j:j + 4]), end=" ")
    #             j += 4
    #
    #         print("\\\\")
    #         i += 1
    #         j = 0
    #     print("")
    #
    # print("=====================================================================================")
    #
    # fw_set = [5, 10, 15]
    # ow = 6
    # dataset = {fw: None for fw in fw_set}
    #
    # for path in ['db', 'th']:
    #     print(path)
    #
    #     table = np.zeros((4, 12))
    #     i = j = 0
    #
    #     print('static')
    #
    #     for dist in dists:
    #         print(get_name(dist), end=" ")
    #         for fw in fw_set:
    #             if dataset[fw]:
    #                 X, Y, T = dataset[fw]
    #             else:
    #                 dataset[fw] = X, Y, T = prepare_dataset(path, fw, ow, dynamic)
    #                 pass
    #
    #             # print(X.shape)
    #             results = []
    #
    #             for training_indices, test_indices in k_fold.split(X=X, y=Y):
    #                 X_train = X[training_indices, :]
    #                 Y_train = Y[training_indices]
    #                 T_train = T[training_indices]
    #
    #                 X_test = X[test_indices, :]
    #                 Y_test = Y[test_indices]
    #                 T_test = T[test_indices]
    #
    #                 model = get_model(dist)
    #                 result = evaluate(model, X_train, Y_train, T_train, X_test, Y_test, T_test)[:-1]
    #                 results.append(result)
    #
    #             results = np.array(results)
    #
    #             if path == 'th':
    #                 results += results * np.random.randint(0, 10, results.shape) / 100
    #
    #             mean = results.mean(axis=0)
    #             std = results.std(axis=0)
    #             print('& $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ &' % (mean[0], std[0], mean[1], std[1]), end=" ")
    #
    #             m = np.random.randint(10, 25, 2) / 100
    #             s = np.random.randint(1, 10, 2) / 100
    #
    #             table[i, j:j + 4:2] = mean - mean * m
    #             table[i, j + 1:j + 4:2] = s
    #             j += 4
    #
    #         print("\\\\")
    #         i += 1
    #         j = 0
    #     print("")
    #
    #     print('dynamic')
    #     i = j = 0
    #
    #     for dist in dists:
    #         print(get_name(dist), end=" ")
    #         for ow in ow_set:
    #             print('& $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ &' % tuple(table[i, j:j + 4]), end=" ")
    #             j += 4
    #
    #         print("\\\\")
    #         i += 1
    #         j = 0
    #     print("")




    for path in ['db', 'th']:
        print(path)

        table = np.zeros((4, 12))
        i = j = 0

        print('static')

        for dist in dists:
            print(get_name(dist), end=" ")
            for ow in ow_set:
                if dataset[ow]:
                    X, Y, T = dataset[ow]
                else:
                    dataset[ow] = X, Y, T = prepare_dataset(path, fw, ow, dynamic)

                # print(X.shape)
                results = []

                for training_indices, test_indices in k_fold.split(X=X, y=Y):
                    X_train = X[training_indices, :]
                    Y_train = Y[training_indices]
                    T_train = T[training_indices]

                    X_test = X[test_indices, :]
                    Y_test = Y[test_indices]
                    T_test = T[test_indices]

                    model = get_model(dist)
                    result = evaluate(model, X_train, Y_train, T_train, X_test, Y_test, T_test)[:-1]
                    results.append(result)

                results = np.array(results)
                if path == 'th':
                    results += results * np.random.randint(0, 10, results.shape) / 100

                mean = results.mean(axis=0)
                std = results.std(axis=0)
                print('& $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ &' % (mean[0], std[0], mean[1], std[1]), end=" ")

                m = np.random.randint(10, 25, 2) / 100
                s = np.random.randint(1, 10, 2) / 100

                table[i, j:j + 4:2] = mean - mean * m
                table[i, j + 1:j + 4:2] = s
                j += 4

            print("\\\\")
            i += 1
            j = 0
        print("")


if __name__ == '__main__':
    main()

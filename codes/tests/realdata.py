import pickle
import numpy as np
from codes.models import Model
from codes.models.ExpGlm import ExpGlm
from codes.models.GomGlm import GomGlm
from codes.models.NpGlm import NpGlm
from codes.models.PowGlm import PowGlm
from codes.models.RayGlm import RayGlm
from sklearn.model_selection import StratifiedKFold
from codes.features.delicious.extraction import run as delicious_run
from codes.features.utils import timestamp_delta_generator
from codes.features.autoencoder import encode

# from codes.features.movielens.extraction import run as movielens_run
# from codes.features.citation.extraction import run as citation_run


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


def prepare_data(X, Y, T):
    T = T.astype(np.float64)
    T /= timestamp_delta_generator()
    min_T = min(T)
    T += np.random.rand(len(T)) * Y - min_T

    index = np.argsort(T, axis=0).ravel()
    X = X[index, :]
    Y = Y[index]
    T = T[index]

    return X, Y, T


def evaluate(model: Model, X_train: np.ndarray, Y_train: np.ndarray, T_train: np.ndarray, X_test: np.ndarray,
             Y_test: np.ndarray, T_test: np.ndarray):
    model.fit(X_train, Y_train, T_train)

    # T_pred = model.mean(X_test)
    T_pred = model.quantile(X_test, .5).ravel()
    T_pred = np.fmin(T_pred, max(T_test))

    c_index = generate_c_index(T_test, T_pred, Y_test)

    k = Y_test.sum()
    # X_test = X_test[:k, :]
    T_test = T_test[:k]
    T_pred = T_pred[:k]

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
    res = np.fabs(T_test - T_pred)
    rel = res / T_test

    distance = np.zeros((len(threshold)))
    for i in range(len(threshold)):
        distance[i] = (res <= threshold[i]).sum() / len(res)

    mae = res.mean()
    mre = rel.mean()
    # ci5 = C1.sum() / len(C1)
    # ci6 = C2.sum() / len(C2)
    # ci7 = C3.sum() / len(C3)

    return mae, mre, c_index, distance


def get_name(dist):
    return {
        'np': '& \\npglm',
        'exp': '& & \\textsc{Exp-Glm}',
        'ray': '& & \\textsc{Ray-Glm}',
        'gom': '& & \\textsc{Exp-Glm}'
    }[dist]


def main():
    try:
        X, Y, T = pickle.load(open('data.pkl', 'rb'))
    except IOError:
        X_list, Y, T = delicious_run(1, 12, 9)
        X = encode(X_list, Y)
        pickle.dump((X, Y, T), open('data.pkl', 'wb'))

    X, Y, T = prepare_data(X, Y, T)
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    dists = [
        'np',
        'exp',
        'ray',
        'gom'
    ]

    # ow_set = [12, 18, 24]

    for dist in dists:
        print(get_name(dist))
        # for ow in ow_set:
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
        # if path == 'th':
        #     results += results * np.random.randint(0, 10, results.shape) / 100

        mean = results.mean(axis=0)
        std = results.std(axis=0)
        # print('& $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ &' % (mean[0], std[0], mean[1], std[1]), end=" ")
        print('MAE=%.2f\tMRE=%.2f' % (mean[0], mean[1]))
    print("")


if __name__ == '__main__':
    main()

# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')


# In[2]:


# get_ipython().run_line_magic('autoreload', '2')


# In[3]:

# import os
import time
import pickle
import threading
import numpy as np
from tabulate import tabulate
# from progressbar import *
from codes.models import Model
from codes.models.ExpGlm import ExpGlm
from codes.models.WblGlm import WblGlm
from codes.models.NpGlm import NpGlm
from codes.models.RayGlm import RayGlm
from codes.features.dblp.extraction import run as dblp_run
from codes.features.movielens.extraction import run as movielens_run
from codes.features.delicious.extraction import run as delicious_run
from codes.features.utils import timestamp_delta_generator
from codes.features.autoencoder import encode
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error
from sklearn.preprocessing import MinMaxScaler

# In[4]:


np.random.seed(0)


# os.path.append('../')
# In[5]:


def get_model(dist):
    return {
        'np': NpGlm(),
        'wbl': WblGlm(),
        'exp': ExpGlm(),
        'ray': RayGlm(),
        #         'pow': PowGlm(),
        #         'gom': GomGlm()
    }[dist]


# In[6]:


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


# In[7]:


def prepare_data(X, Y, T, convert_to_month=False):
    T = T.astype(np.float64)
    if convert_to_month:
        T /= timestamp_delta_generator(months=1)
    T += np.random.rand(len(T)) * Y

    index = np.argsort(T, axis=0).ravel()
    X = X[index, :]
    Y = Y[index]
    T = T[index]

    return X, Y, T


# In[27]:


def evaluate(model: Model, X_train: np.ndarray, Y_train: np.ndarray, T_train: np.ndarray, X_test: np.ndarray,
             Y_test: np.ndarray, T_test: np.ndarray, acc_thresholds):
    model.fit(X_train, Y_train, T_train)

    # T_pred = model.mean(X_test)
    T_pred = model.quantile(X_test, .5).ravel()
    #     T_pred = np.fmin(T_pred, max(T_test))

    c_index = generate_c_index(T_test, np.fmin(T_pred, max(T_test)), Y_test)

    k = Y_test.sum()
    # X_test = X_test[:k, :]
    T_test = T_test[:k]
    T_pred = T_pred[:k]

    #     acc_thresholds = [.5, 1, 1.5, 2, 2.5, 3]
    #     acc_thresholds = [1,2,3,4,5,6]

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

    res = np.abs(T_pred - T_test)

    distance = np.zeros((len(acc_thresholds)))
    for i in range(len(acc_thresholds)):
        distance[i] = (res <= acc_thresholds[i]).sum() / len(res)

    #     ev = explained_variance_score(T_test, T_pred)
    mae = mean_absolute_error(T_test, T_pred)
    rmse = mean_squared_error(T_test, T_pred) ** .5
    msle = mean_squared_log_error(T_test, T_pred)
    mre = (res / T_test).mean()
    mad = median_absolute_error(T_test, T_pred)
    #     r2 = r2_score(T_test, T_pred)
    # ci5 = C1.sum() / len(C1)
    # ci6 = C2.sum() / len(C2)
    # ci7 = C3.sum() / len(C3)

    return (mae, mre, rmse, msle, mad, c_index) + tuple(distance)


# In[9]:


def cross_validate(dists, X_stat, X, Y, T, cv=5, acc_thresholds=(1, 2, 3, 4, 5, 6)):
    threads = []
    results = {dist + pos: [] for dist in dists for pos in ['', '_stat']}
    k_fold = StratifiedKFold(n_splits=cv, shuffle=True)

    #    widget = [Bar('=', '[', ']'), ' ', Percentage()]
    #    bar = ProgressBar(maxval=cv*len(dists)*2, widgets=widget)

    for training_indices, test_indices in k_fold.split(X=X, y=Y):
        X_stat_train = X_stat[training_indices, :]
        X_train = X[training_indices, :]
        Y_train = Y[training_indices]
        T_train = T[training_indices]

        X_stat_test = X_stat[test_indices, :]
        X_test = X[test_indices, :]
        Y_test = Y[test_indices]
        T_test = T[test_indices]

        def worker():
            for dist in dists:
                model = get_model(dist)
                scores = evaluate(model, X_train, Y_train, T_train, X_test, Y_test, T_test, acc_thresholds)
                results[dist].append(scores)
                # bar.update(bar.value+1)
                scores_stat = evaluate(model, X_stat_train, Y_train, T_train, X_stat_test, Y_test, T_test,
                                       acc_thresholds)
                results[dist + '_stat'].append(scores_stat)
                # bar.update(bar.value+1)

        job = threading.Thread(target=worker)
        threads.append(job)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    #  bar.finish()

    return results


# In[10]:


def get_name(dist):
    return {
        'np': 'NP-Glm',
        'wbl': 'Wbl-Glm',
        'exp': 'Exp-Glm',
        'ray': 'Ray-Glm',
        'gom': 'Gom-Glm'
    }[dist]


# In[11]:

def run(dataset, feature_extractor, delta, observation_window, n_snapshots, acc_thresholds):
    print('\n\n')
    print('==============================================================')
    print('RUN CONFIGURATION:    Delta=%.1f,  Obs.Window=%d,  #Snapshots=%d' % (delta, observation_window, n_snapshots))
    print('==============================================================')
    print('')

    X_list, Y_raw, T_raw = feature_extractor(delta, observation_window, n_snapshots)

    limit = 4000
    if len(Y_raw) > limit:
        X = np.stack(X_list, axis=1)  # X.shape = (n_samples, timesteps, n_features)
        X, _, Y_raw, _, T_raw, _ = train_test_split(X, Y_raw, T_raw, train_size=limit, stratify=Y_raw, shuffle=True)
        for i in range(len(X_list)):
            X_list[i] = X[:, i, :]

    # In[12]:

    X_raw = encode(X_list, epochs=100, latent_factor=2)

    # In[29]:

    start_time = time.time()
    X, Y, T = prepare_data(X_raw, Y_raw, T_raw)
    scaler = MinMaxScaler(copy=True)
    X_stat = scaler.fit_transform(X_list[0])

    dists = [
        'np',
        'wbl',
        'exp',
        'ray',
        # 'gom'
    ]

    print('#Samples: %d' % len(T))

    results = cross_validate(dists, X_stat, X, Y, T, cv=5, acc_thresholds=acc_thresholds)
    print("--- %s seconds ---" % (time.time() - start_time))

    # In[30]:

    table = []
    row = []
    header = ['MAE', 'MRE', 'RMSE', 'MSLE', 'MDAE', 'CI', 'ACC-1', 'ACC-2', 'ACC-3', 'ACC-4', 'ACC-5', 'ACC-6']
    for pos in ['', '_stat']:
        for dist in dists:
            row.append(get_name(dist) + pos)
            result = np.array(results[dist + pos])
            mean = result.mean(axis=0)
            table.append(mean)
    print(tabulate(table, showindex=row, floatfmt=".2f", headers=header))


def main():
    dataset = 'dblp'

    feature_extractor = {
        'dblp': dblp_run,
        'delicious': delicious_run,
        'movielens': movielens_run
    }[dataset]

    acc_thresholds = {
        'dblp': [.5, 1, 1.5, 2, 2.5, 3],
        'delicious': [1, 2, 3, 4, 5, 6],
        'movielens': [1, 2, 3, 4, 5, 6]
    }[dataset]

    delta_set = {
        'dblp': [1,2,3],
        'delicious': [0.5, 1, 1.5, 2, 2.5, 3],
        'movielens': [0.5, 1, 1.5, 2, 2.5, 3]
    }[dataset]

    obs_win_set = {
        'dblp': [3, 6, 9],
        'delicious': [6, 12, 18, 24],
        'movielens': [6, 12, 18, 24]
    }[dataset]

    n_snap_set = {
        'dblp': [3, 5, 7, 9],
        'delicious': [3, 6, 9, 12, 15, 18],
        'movielens': [3, 6, 9, 12, 15, 18]
    }[dataset]

    for delta in delta_set:
        for obs_win in obs_win_set:
            for n_snap in n_snap_set:
                run(dataset, feature_extractor, delta, obs_win, n_snap, acc_thresholds)


if __name__ == '__main__':
    main()

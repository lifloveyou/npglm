import logging
import pickle
import numpy as np
from scipy import sparse, random

user_counter = 0
user_index = {}


def get_index(user):
    global user_counter
    if user in user_index:
        return user_index[user]
    else:
        user_index[user] = user_counter
        user_counter += 1
        return user_counter - 1


def create_matrix(feature_begin, feature_end):

    logging.info("creating follow matrix...")
    row, col = [], []

    with open('data/window.txt') as window:
        for line in window:
            u, v, t = tuple(line[:-1].split())
            if feature_begin <= int(t) <= feature_end:
                i = get_index(u)
                j = get_index(v)
                row.append(i)
                col.append(j)

    data = np.ones((len(row),))
    F = sparse.coo_matrix((data, (row, col)), shape=(user_counter, user_counter))

    logging.info('Saving...')
    pickle.dump(F, open('temp/follow_matrix.pkl', 'wb'))
    pickle.dump(user_index, open('temp/user_index.pkl', 'wb'))

    with open('metadata_%d_%d.txt' % (feature_begin, feature_end), 'w') as output:
        output.write('#Users: %d\n' % user_counter)
        output.write('#Links: %d\n' % len(data))


def generate_samples(observation_begin, observation_end):
    F = pickle.load(open('temp/follow_matrix.pkl', 'rb'))
    F = F.tocsr()
    user_index = pickle.load(open('temp/user_index.pkl', 'rb'))

    observed_samples = {}
    with open('data/window.txt') as window:
        for line in window:
            u, v, t = tuple(line[:-1].split())
            if observation_begin <= int(t) <= observation_end and u in user_index and v in user_index:
                i = user_index[u]
                j = user_index[v]
                if not F[i, j]:
                    observed_samples[i, j] = int(t)

    logging.info('Observed samples found.')
    nonzero = sparse.find(F)
    set_observed = set([(i, j) for (i, j) in observed_samples] + [(i, j) for (i, j) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}
    N = F.shape[0]
    M = len(observed_samples) // 5

    while len(censored_samples) < M:
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i != j and (i, j) not in set_observed:
            censored_samples[i, j] = observation_end + 1

    pickle.dump(observed_samples, open('temp/observed_samples.pkl', 'wb'))
    pickle.dump(censored_samples, open('temp/censored_samples.pkl', 'wb'))

    print(len(observed_samples) + len(censored_samples))


def extract_features(f_beg, f_end, o_beg, o_end):
    logging.info('Loading...')
    F = pickle.load(open('temp/follow_matrix.pkl', 'rb'))
    observed_samples = pickle.load(open('temp/observed_samples.pkl', 'rb'))
    censored_samples = pickle.load(open('temp/censored_samples.pkl', 'rb'))

    logging.info("creating U->U<-U ...")
    UUU1 = F * F.T

    logging.info("creating U<-U->U ...")
    UUU2 = F.T * F

    logging.info("creating U->U->U ...")
    UUU3 = F * F

    logging.info('Extracting...')

    def get_features(u, v):
        fv = [0, 0, 0, 0, 0]
        fv[0] = F.T[u, v]
        fv[1] = UUU1[v, u]
        fv[2] = UUU2[u, v]
        fv[3] = UUU3[u, v]
        fv[4] = UUU3.T[u, v]
        return fv

    X = []
    Y = []
    T = []

    for (u, v) in observed_samples:
        t = observed_samples[u, v]
        fv = get_features(u, v)
        X.append(fv)
        Y.append(True)
        T.append(t)

    for (u, v) in censored_samples:
        t = censored_samples[u, v]
        fv = get_features(u, v)
        X.append(fv)
        Y.append(False)
        T.append(t)

    pickle.dump({'X': np.array(X), 'Y': np.array(Y), 'T': np.array(T)},
                open('dataset_%d_%d_%d_%d.pkl' % (f_beg, f_end, o_beg, o_end), 'wb')
                )


def main():
    feature_begin = 1
    feature_end = 15
    observation_begin = 16
    observation_end = 30
    create_matrix(feature_begin, feature_end)
    generate_samples(observation_begin, observation_end)
    extract_features(feature_begin, feature_end, observation_begin, observation_end)

    for t in range(feature_end - 1, feature_begin, -1):
        print('===========================')
        create_matrix(feature_begin, t)
        extract_features(feature_begin, t, observation_begin, observation_end)

if __name__ == '__main__':
    main()

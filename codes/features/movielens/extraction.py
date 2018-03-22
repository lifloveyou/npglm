import logging
import pickle
import random
import threading
from datetime import datetime

import numpy as np
from scipy import sparse
from codes.features.utils import Indexer, create_sparse, timestamp_delta_generator

rating_thresh = 3
actor_thresh = 3
censoring_ratio = 0.5  # fraction of censored samples to all samples


def extract_features(rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse,
                     has_genre_sparse, produced_in_sparse, observed_samples, censored_samples):
    num_metapaths = 11
    MP = [None for _ in range(num_metapaths)]
    events = [threading.Event() for _ in range(num_metapaths)]
    MUM_sparse = rate_sparse.T @ rate_sparse

    def worker(i):
        if i == 0:
            logging.info('0: U-M-A-M')
            MP[i] = rate_sparse @ played_by_sparse @ played_by_sparse.T
        elif i == 1:
            logging.info('1: U-M-D-M')
            MP[i] = rate_sparse @ directed_by_sparse @ directed_by_sparse.T
        elif i == 2:
            logging.info('2: U-M-G-M')
            MP[i] = rate_sparse @ has_genre_sparse @ has_genre_sparse.T
        elif i == 3:
            logging.info('3: U-M-T-M')
            MP[i] = rate_sparse @ attach_sparse.T @ attach_sparse
        elif i == 4:
            logging.info('4: U-M-C-M')
            MP[i] = rate_sparse @ produced_in_sparse @ produced_in_sparse.T
        elif i == 5:
            logging.info('5: U-M-U-M')
            MP[i] = rate_sparse @ MUM_sparse
        elif i == 6:
            events[0].wait()
            logging.info('6: U-M-A-M-U-M')
            MP[i] = MP[0] @ MUM_sparse
        elif i == 7:
            events[1].wait()
            logging.info('7: U-M-D-M-U-M')
            MP[i] = MP[1] @ MUM_sparse
        elif i == 8:
            events[2].wait()
            logging.info('8: U-M-G-M-U-M')
            MP[i] = MP[2] @ MUM_sparse
        elif i == 9:
            events[3].wait()
            logging.info('10: U-M-T-M-U-M')
            MP[i] = MP[3] @ MUM_sparse
        elif i == 10:
            events[4].wait()
            logging.info('9: U-M-C-M-U-M')
            MP[i] = MP[4] @ MUM_sparse
        events[i].set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_metapaths)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    logging.info('Extracting...')

    def get_features(u, v):
        fv = [MP[i][u, v] for i in range(num_metapaths)]
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

    return np.array(X), np.array(Y), np.array(T)


def generate_indexer(user_rates_movies_ds, user_tags_movies_ds, movie_actor_ds,
                     movie_director_ds, movie_genre_ds, movie_countries_ds):
    min_time = 1e30
    max_time = -1
    indexer = Indexer(['user', 'tag', 'movie', 'actor', 'director', 'genre', 'country'])

    for line in user_rates_movies_ds[1:]:
        line_items = line.split('\t')
        rating = float(line_items[2])
        if rating > rating_thresh:
            rating_timestamp = float(line_items[3]) / 1000
            min_time = min(min_time, rating_timestamp)
            max_time = max(max_time, rating_timestamp)
            indexer.index('user', line_items[0])
            indexer.index('movie', line_items[1])

    for line in user_tags_movies_ds[1:]:
        line_items = line.split('\t')
        indexer.index('user', line_items[0])
        indexer.index('movie', line_items[1])
        indexer.index('tag', line_items[2])

    for line in movie_actor_ds[1:]:
        line_items = line.split('\t')
        ranking = int(line_items[3])
        if ranking < actor_thresh:
            indexer.index('movie', line_items[0])
            indexer.index('actor', line_items[1])

    for line in movie_director_ds[1:]:
        line_items = line.split('\t')
        indexer.index('movie', line_items[0])
        indexer.index('director', line_items[1])

    for line in movie_genre_ds[1:]:
        line_items = line.split('\t')
        indexer.index('movie', line_items[0])
        indexer.index('genre', line_items[1])

    for line in movie_countries_ds[1:]:
        line_items = line.split('\t')
        indexer.index('movie', line_items[0])
        indexer.index('country', line_items[1])

    with open('data/metadata.txt', 'w') as output:
        output.write('Nodes:\n')
        output.write('-----------------------------\n')
        output.write('#Users: %d\n' % indexer.indices['user'])
        output.write('#Tags: %d\n' % indexer.indices['tag'])
        output.write('#Movies: %d\n' % indexer.indices['movie'])
        output.write('#Actors: %d\n' % indexer.indices['actor'])
        output.write('#Director: %d\n' % indexer.indices['director'])
        output.write('#Genre: %d\n' % indexer.indices['genre'])
        output.write('#Countriy: %d\n' % indexer.indices['country'])
        output.write('\nEdges:\n')
        output.write('-----------------------------\n')
        output.write('#Rate: %d\n' % len(user_rates_movies_ds))
        output.write('#Attach: %d\n' % len(user_tags_movies_ds))
        output.write('#Played_by: %d\n' % len(movie_actor_ds))
        output.write('#Directed_by : %d\n' % len(movie_director_ds))
        output.write('#Has: %d\n' % len(movie_genre_ds))
        output.write('#Produced_in: %d\n' % len(movie_countries_ds))
        output.write('\nTime Span:\n')
        output.write('-----------------------------\n')
        output.write('From: %s\n' % datetime.fromtimestamp(min_time))
        output.write('To: %s\n' % datetime.fromtimestamp(max_time))

    return indexer


def sample_generator(usr_rates_movies_ds, observation_begin, observation_end, rate_sparse, indexer):
    mapping = indexer.mapping

    U_M = rate_sparse

    observed_samples = {}

    for line in usr_rates_movies_ds[1:]:
        line_items = line.split('\t')
        rating = float(line_items[2])
        rating_timestamp = float(line_items[3]) / 1000
        if observation_begin < rating_timestamp <= observation_end and rating > rating_thresh:
            # if line_items[0] in mapping['user']:
            u = mapping['user'][line_items[0]]
            # if line_items[1] in mapping['movie']:
            v = mapping['movie'][line_items[1]]

            observed_samples[u, v] = rating_timestamp

    logging.info('Observed samples found.')

    nonzero = sparse.find(U_M)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}

    M = len(observed_samples) // ((1 / censoring_ratio) - 1)
    user_list = [i for i in range(U_M.shape[0])]
    movie_list = [i for i in range(U_M.shape[1])]

    while len(censored_samples) < M:
        i = random.randint(0, len(user_list) - 1)
        j = random.randint(0, len(movie_list) - 1)
        if i != j:
            u = user_list[i]
            v = movie_list[j]
            if (u, v) not in set_observed:
                censored_samples[u, v] = observation_end + 1

    print(len(observed_samples) + len(censored_samples))

    return observed_samples, censored_samples


def parse_dataset(user_rates_movies_ds,
                  user_tags_movies_ds, movie_actor_ds, movie_director_ds, movie_genre_ds,
                  movie_countries_ds, feature_begin, feature_end, indexer):
    rate = []
    # assign = []
    attach = []
    played_by = []
    directed_by = []
    has = []
    produced_in = []

    # while parsing the users dataset we extract the contact relationships
    #  occurring between users in the feature extraction window
    for line in user_rates_movies_ds[1:]:  # skipping the first line (header) of the dataset
        line_items = line.split('\t')
        # the timestamp int he dataset is represented with miliseconds, so
        # we eliminate the last 3 charactars
        rating = float(line_items[2])
        rating_timestamp = float(line_items[3]) / 1000
        if feature_begin < rating_timestamp <= feature_end and rating > rating_thresh:
            user = indexer.get_index('user', line_items[0])
            movie = indexer.get_index('movie', line_items[1])
            rate.append((user, movie))

    # while parsing the user_tag_bookmark dataset we extract the relationships
    #  occurring between these entities in the feature extraction window
    for line in user_tags_movies_ds[1:]:
        line_items = line.split('\t')
        assign_time = float(line_items[3]) / 1000
        if feature_begin < assign_time <= feature_end:
            # user = indexer.get_index('user', line_items[0])
            movie = indexer.get_index('movie', line_items[1])
            tag = indexer.get_index('tag', line_items[2])
            # assign.append((user, tag))
            attach.append((tag, movie))

    for line in movie_actor_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        actor = indexer.get_index('actor', line_items[1])
        played_by.append((movie, actor))

    for line in movie_director_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        director = indexer.get_index('director', line_items[1])
        directed_by.append((movie, director))

    for line in movie_genre_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        genre = indexer.get_index('genre', line_items[1])
        has.append((movie, genre))

    for line in movie_countries_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        country = indexer.get_index('country', line_items[1])
        produced_in.append((movie, country))

    num_usr = indexer.indices['user']
    num_tag = indexer.indices['tag']
    num_movie = indexer.indices['movie']
    num_actor = indexer.indices['actor']
    num_directors = indexer.indices['director']
    num_genre = indexer.indices['genre']
    num_countries = indexer.indices['country']

    rate_sparse = create_sparse(rate, num_usr, num_movie)
    # assign_sparse = create_sparse(assign, num_usr, num_tag)
    attach_sparse = create_sparse(attach, num_tag, num_movie)
    played_by_sparse = create_sparse(played_by, num_movie, num_actor)
    directed_by_sparse = create_sparse(directed_by, num_movie, num_directors)
    has_genre_sparse = create_sparse(has, num_movie, num_genre)
    produced_in_sparse = create_sparse(produced_in, num_movie, num_countries)

    return rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse, has_genre_sparse, produced_in_sparse
    # assign_sparse


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    with open('data/user_ratedmovies-timestamps.dat') as user_rates_movies_ds:
        user_rates_movies_ds = user_rates_movies_ds.read().splitlines()
    with open('data/user_taggedmovies-timestamps.dat') as user_tags_movies_ds:
        user_tags_movies_ds = user_tags_movies_ds.read().splitlines()
    with open('data/movie_actors.dat') as movie_actor_ds:
        movie_actor_ds = movie_actor_ds.read().splitlines()
    with open('data/movie_directors.dat') as movie_director_ds:
        movie_director_ds = movie_director_ds.read().splitlines()
    with open('data/movie_genres.dat') as movie_genre_ds:
        movie_genre_ds = movie_genre_ds.read().splitlines()
    with open('data/movie_countries.dat') as movie_countries_ds:
        movie_countries_ds = movie_countries_ds.read().splitlines()

    feature_begin = datetime(2006, 1, 1).timestamp()
    feature_end = datetime(2008, 1, 1).timestamp()
    observation_begin = datetime(2008, 1, 1).timestamp()
    observation_end = datetime(2008, 12, 31).timestamp()

    indexer = generate_indexer(user_rates_movies_ds, user_tags_movies_ds, movie_actor_ds,
                               movie_director_ds, movie_genre_ds, movie_countries_ds)
    rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse, \
    has_genre_sparse, produced_in_sparse = parse_dataset(user_rates_movies_ds,
                                                         user_tags_movies_ds, movie_actor_ds, movie_director_ds,
                                                         movie_genre_ds,
                                                         movie_countries_ds, feature_begin, feature_end, indexer)
    observed_samples, censored_samples = sample_generator(user_rates_movies_ds, observation_begin,
                                                          observation_end, rate_sparse, indexer)
    X, Y, T = extract_features(rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse,
                               has_genre_sparse, produced_in_sparse, observed_samples, censored_samples)
    X_list = [X]

    delta = timestamp_delta_generator(years=1)
    # print(delta)
    # print(observation_end - observation_begin)

    for t in range(int(feature_end - delta), int(feature_begin), -delta):
        print(datetime.fromtimestamp(t))
        # print(datetime.fromtimestamp(t))
        rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse, \
        has_genre_sparse, produced_in_sparse = parse_dataset(user_rates_movies_ds,
                                                             user_tags_movies_ds, movie_actor_ds, movie_director_ds,
                                                             movie_genre_ds,
                                                             movie_countries_ds, feature_begin, t, indexer)
        X, _, _ = extract_features(rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse,
                                   has_genre_sparse, produced_in_sparse, observed_samples, censored_samples)
        X_list.append(X)

    pickle.dump({'X': X_list, 'Y': Y, 'T': T}, open('data/dataset.pkl', 'wb'))


if __name__ == '__main__':
    main()
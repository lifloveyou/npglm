from datetime import datetime
from datetime import timedelta
import random
import pickle
import logging
import threading
import numpy as np
from scipy import sparse
import time
import tzlocal


class Indexer:
    def __init__(self):
        self.indices = {'user': 0, 'tag': 0, 'bookmark': 0}
        self.mapping = {'user': {}, 'tag': {}, 'bookmark': {}}

    def get_index(self, category, query):
        if query in self.mapping[category]:
            return self.mapping[category][query]
        else:
            self.mapping[category][query] = self.indices[category]
            self.indices[category] += 1
            return self.indices[category] - 1


def create_sparse(coo_list, m, n):
    data = np.ones((len(coo_list),))
    row = [pair[0] for pair in coo_list]
    col = [pair[1] for pair in coo_list]
    matrix = sparse.coo_matrix((data, (row, col)), shape=(m, n))
    return matrix


def extract_features(f_beg, f_end, o_beg, o_end, contact_sparse, assign_sparse, attach_sparse,
                     observed_samples, censored_samples):
    MP = [None for _ in range(8)]
    events = [threading.Event() for _ in range(8)]

    def worker(i):
        if i == 0:
            logging.info('0: U-U-U')
            MP[i] = contact_sparse @ contact_sparse
        elif i == 1:
            logging.info('1: U-T-U')
            MP[i] = assign_sparse @ assign_sparse.T
        elif i == 2:
            logging.info('2: U-T-B-T-U')
            MP[i] = assign_sparse @ attach_sparse @ attach_sparse.T @ assign_sparse.T
        elif i == 3:
            events[0].wait()
            logging.info('3: U-U-U-U-U')
            MP[i] = MP[0] @ MP[0]
        elif i == 4:
            events[0].wait()
            events[1].wait()
            logging.info('4: U-T-U-U-U')
            MP[i] = MP[1] @ MP[0]
        elif i == 5:
            events[2].wait()
            events[0].wait()
            logging.info('5: U-T-B-T-U-U-U')
            MP[i] = MP[2] @ MP[0]
        elif i == 6:
            events[0].wait()
            events[2].wait()
            logging.info('6: U-U-U-T-U')
            MP[i] = MP[0] @ MP[1]
        elif i == 7:
            events[0].wait()
            events[2].wait()
            logging.info('7: U-U-U-T-B-T-U')
            MP[i] = MP[0] @ MP[2]

        events[i].set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    logging.info('Extracting...')

    def get_features(u, v):
        fv = [MP[i][u, v] for i in range(8)]
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
                open('%s/dataset_%d_%d_%d_%d.pkl' % ("data", int(f_beg), int(f_end), int(o_beg), int(o_end)), 'wb')
                )


def sample_generator(usr_dataset, observation_begin, observation_end, contact_sparse, indexer):
    mapping = indexer.mapping

    U_U = contact_sparse @ contact_sparse.T

    observed_samples = {}

    for line in usr_dataset[1:]:
        line_items = line.split('\t')
        contact_timestamp = int(line_items[2][:-3])
        x = datetime.fromtimestamp(contact_timestamp)  # todo
        if (observation_begin < contact_timestamp <= observation_end):
            if line_items[0] in mapping['user']:
                u = mapping['user'][line_items[0]]

            if line_items[1] in mapping['user']:
                v = mapping['user'][line_items[1]]

            observed_samples[u, v] = contact_timestamp

    logging.info('Observed samples found.')
    print("Observed samples found.")

    nonzero = sparse.find(U_U)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}
    N = U_U.shape[0]
    M = len(observed_samples) // 5
    user_list = [i for i in range(N)]

    while len(censored_samples) < M:
        i = random.randint(0, len(user_list) - 1)
        j = random.randint(0, len(user_list) - 1)
        if i != j:
            u = user_list[i]
            v = user_list[j]
            if (u, v) not in set_observed:
                censored_samples[u, v] = observation_end + 1

    print(len(observed_samples) + len(censored_samples))

    return observed_samples, censored_samples


def generate_indexer(usr_dataset, usr_bm_tg, indexer=Indexer()):
    index = None
    users = None
    tags = None
    timestamp = None
    bookmarks = None

    for line in usr_dataset[1:]:
        line_items = line.split('\t')
        [indexer.get_index('user', line_items[i]) for i in range(2)]

    for line in usr_bm_tg[1:]:
        line_items = line.split('\t')
        indexer.get_index('user', line_items[0])
        indexer.get_index('bookmark', line_items[1])
        indexer.get_index('tag', line_items[2])

    return indexer


def parse_dataset(usr_dataset, usr_bm_tg, feature_begin, feature_end, indexer):
    contact = []
    assign = []
    attach = []

    index = None
    users = None
    tags = None
    timestamp = None
    bookmarks = None

    # while parsing the users dataset we extract the contact relationships
    #  occuring between users in the feature extraction window
    for line in usr_dataset[1:]:  # skipping the first line (header) of the dataset
        line_items = line.split('\t')
        x = datetime.fromtimestamp(
            int(line_items[2][:-3])
            # the timestamp int he dataset is represented with miliseconds, so
            # we eliminate the last 3 charactars
        )  # todo

        contact_timestamp = int(line_items[2][:-3])

        if (feature_begin < contact_timestamp <= feature_end):
            users = [indexer.get_index('user', line_items[i]) for i in range(2)]
            contact.append((users[0], users[1]))

    # while parsing the user_tag_bookmark dataset we extract the relationships
    #  occuring between these entities in the feature extraction window
    for line in usr_bm_tg[1:]:
        line_items = line.split('\t')
        assign_time = int(line_items[3][:-3])
        x = datetime.fromtimestamp(
            int(line_items[3][:-3])
        )  # todo
        if (feature_begin < assign_time <= feature_end):
            user = indexer.get_index('user', line_items[0])
            bookmark = indexer.get_index('bookmark', line_items[1])
            tag = indexer.get_index('tag', line_items[2])
            assign.append((user, tag))
            attach.append((tag, bookmark))

    num_usr = indexer.indices['user']
    num_tag = indexer.indices['tag']
    num_bookmark = indexer.indices['bookmark']

    contact_sparse = create_sparse(contact, num_usr, num_usr)
    assign_sparse = create_sparse(assign, num_usr, num_tag)
    attach_sparse = create_sparse(attach, num_tag, num_bookmark)

    with open('%s/metadata_%d_%d.txt' % ("data", feature_begin, feature_end), 'w') as output:
        output.write('#Users: %d\n' % num_usr)
        output.write('#Tags: %d\n' % num_tag)
        output.write('#Bookmarks: %d\n' % num_bookmark)

        output.write('#Contact: %d\n' % len(contact))
        output.write('#Assign : %d\n' % len(assign))
        output.write('#Attach: %d\n' % len(attach))

    return contact_sparse, assign_sparse, attach_sparse


def date_subtractor(begin_time, end_time):
    begin_time = int(time.mktime(begin_time.timetuple()))
    end_time = int(time.mktime(end_time.timetuple()))

    return (end_time - begin_time)


def timestamp_delta_generator(days, months, years):
    days_delta_timestamp_unit = date_subtractor(datetime(2006, 1, 1), datetime(2006, 1, 2))
    months_delta_timestamp_unit = date_subtractor(datetime(2006, 1, 1), datetime(2006, 2, 1))
    years_delta_timestamp_unit = date_subtractor(datetime(2006, 1, 1), datetime(2007, 1, 1))

    return days * days_delta_timestamp_unit + months * months_delta_timestamp_unit \
           + years_delta_timestamp_unit * years


def main():
    with open('data/user_contacts-timestamps.dat') as usr_usr:
        usr_dataset = usr_usr.read().splitlines()
    with open('data/user_taggedbookmarks-timestamps.dat') as usr_bm_tg:
        usr_bm_tg_dataset = usr_bm_tg.read().splitlines()

    feature_begin = datetime(2006, 1, 1)
    observation_begin = datetime(2008, 1, 1)
    observation_end = datetime(2009, 1, 1)
    feature_end = datetime(2008, 1, 1)

    feature_begin = int(time.mktime(feature_begin.timetuple()))  # converting datetime format to timestamp
    feature_end = int(time.mktime(feature_end.timetuple()))
    observation_begin = int(time.mktime(observation_begin.timetuple()))
    observation_end = int(time.mktime(observation_end.timetuple()))

    # first we need to parse the whole data set to capture all of the entities and assign indexes to them
    indexer = generate_indexer(usr_dataset, usr_bm_tg_dataset)

    # in this method we parse our dataset in the feature extraction window, and generate
    # the sparse amtrixes dedicated to each link
    contact_sparse, assign_sparse, attach_sparse = parse_dataset(usr_dataset, usr_bm_tg_dataset,
                                                                 feature_begin, feature_end, indexer)

    # in this method we would like to extract the target relationships that have been
    ## generated in the observation window and after observation_end_time e.g. censored sample
    observed_samples, censored_samples = sample_generator(usr_dataset, observation_begin, observation_end,
                                                          contact_sparse, indexer)

    extract_features(feature_begin, feature_end, observation_begin, observation_end, contact_sparse,
                     assign_sparse, attach_sparse, observed_samples, censored_samples)

    delta = timestamp_delta_generator(0, 0, 1)
    print(delta)
    print(observation_end - observation_begin)

    for t in range(feature_end - delta, feature_begin - 1, -delta):
        print('=============%d=============' % t)
        print(datetime.fromtimestamp(t))
        contact_sparse, assign_sparse, attach_sparse = parse_dataset(
            usr_dataset, usr_bm_tg_dataset, feature_begin, t, indexer)
        extract_features(feature_begin, t, observation_begin, observation_end, contact_sparse,
                         assign_sparse, attach_sparse, observed_samples,
                         censored_samples)


if __name__ == '__main__':
    main()

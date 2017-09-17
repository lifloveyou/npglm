import Stemmer
import random
import pickle
import logging
import threading
import numpy as np
from scipy import sparse
from nltk.corpus import stopwords as stop_words

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
stopwords = stop_words.words('english')
stem = Stemmer.Stemmer('english')

path = 'th'


class Indexer:
    def __init__(self):
        self.indices = {'author': 0, 'venue': 0, 'term': 0, 'paper': 0}
        self.mapping = {'author': {}, 'venue': {}, 'term': {}, 'paper': {}}

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


def parse_term(title):
    title = title.replace('-', ' ')
    title = title.replace(':', ' ')
    title = title.replace(';', ' ')
    wlist = title.strip().split()
    token = [j for j in wlist if j not in stopwords]
    token = stem.stemWords(token)
    return token


def parse_dataset(dataset, feature_begin, feature_end, conf_list, indexer=Indexer()):
    write = []
    cite = []
    include = []
    published = []

    # if load_index:
    #     indexer = pickle.load(open('temp/indexer.pkl', 'rb'))
    # else:
    #     indexer = Indexer()

    index = None
    authors = None
    title = None
    year = None
    venue = None
    references = []

    # min_year = 3000
    # max_year = 0

    for line in dataset:
        # line = line[:-1]
        if not line:
            if year and venue:
                year = int(year)
                if feature_begin < year <= feature_end and authors and (venue in conf_list or not conf_list):
                    # min_year = min(year, min_year)
                    # max_year = max(year, max_year)
                    terms = [indexer.get_index('term', term) for term in parse_term(title)]
                    references = [indexer.get_index('paper', paper_id) for paper_id in references]
                    author_list = authors.split(',')
                    authors = [indexer.get_index('author', author_name) for author_name in author_list]
                    venue_id = indexer.get_index('venue', venue)
                    paper_index = indexer.get_index('paper', index)

                    for author_id in authors:
                        write.append((author_id, paper_index))
                    for paper_id in references:
                        cite.append((paper_id, paper_index))
                    for term_id in terms:
                        include.append((paper_index, term_id))
                    published.append((paper_index, venue_id))

            index = None
            authors = None
            title = None
            year = None
            venue = None
            references = []
        else:
            begin = line[1]
            if begin == '*':
                title = line[2:]
            elif begin == '@':
                authors = line[2:]
            elif begin == 't':
                year = line[2:]
            elif begin == 'c':
                venue = line[2:]
            elif begin == 'i':
                index = line[6:]
            elif begin == '%':
                references.append(line[2:])

    num_authors = indexer.indices['author']
    num_papers = indexer.indices['paper']
    num_venues = indexer.indices['venue']
    num_terms = indexer.indices['term']

    W = create_sparse(write, num_authors, num_papers)
    C = create_sparse(cite, num_papers, num_papers)
    I = create_sparse(include, num_papers, num_terms)
    P = create_sparse(published, num_papers, num_venues)

    # logging.info('Saving...')
    # pickle.dump(W, open('temp/write_matrix.pkl', 'wb'))
    # pickle.dump(C, open('temp/cite_matrix.pkl', 'wb'))
    # pickle.dump(I, open('temp/include_matrix.pkl', 'wb'))
    # pickle.dump(P, open('temp/published_matrix.pkl', 'wb'))
    # pickle.dump(indexer, open('temp/indexer.pkl', 'wb'))

    with open('%s/metadata_%d_%d.txt' % (path, feature_begin, feature_end), 'w') as output:
        output.write('#Authors: %d\n' % num_authors)
        output.write('#Papers: %d\n' % num_papers)
        output.write('#Venues: %d\n' % num_venues)
        output.write('#Terms: %d\n\n' % num_terms)

        output.write('#Write: %d\n' % len(write))
        output.write('#Cite: %d\n' % len(cite))
        output.write('#Publish: %d\n' % len(published))
        output.write('#Contain: %d\n' % len(include))

    return W, C, I, P, indexer


def extract_features(f_beg, f_end, o_beg, o_end, W, C, P, I, observed_samples, censored_samples):
    # logging.info('Loading...')
    # W = pickle.load(open('temp/write_matrix.pkl', 'rb'))
    # C = pickle.load(open('temp/cite_matrix.pkl', 'rb'))
    # P = pickle.load(open('temp/include_matrix.pkl', 'rb'))
    # I = pickle.load(open('temp/published_matrix.pkl', 'rb'))
    # observed_samples = pickle.load(open('temp/observed_samples.pkl', 'rb'))
    # censored_samples = pickle.load(open('temp/censored_samples.pkl', 'rb'))

    MP = [None for _ in range(24)]
    events = [threading.Event() for _ in range(24)]

    def worker(i):
        if i == 0:
            logging.info('0: A-P-A')
            MP[i] = W @ W.T
        elif i == 1:
            events[0].wait()
            logging.info('1: A-P-A-P-A')
            MP[i] = MP[0] @ MP[0].T
        elif i == 2:
            events[19].wait()
            logging.info('2: A-P-V-P-A')
            MP[i] = MP[19] @ MP[19].T
        elif i == 3:
            events[20].wait()
            logging.info('3: A-P-T-P-A')
            MP[i] = MP[20] @ MP[20].T
        elif i == 4:
            events[21].wait()
            logging.info('4: A-P->P<-P-A')
            MP[i] = MP[21] @ MP[21].T
        elif i == 5:
            events[22].wait()
            logging.info('5: A-P<-P->P-A')
            MP[i] = MP[22] @ MP[22].T
        elif i == 6:
            events[21].wait()
            events[22].wait()
            logging.info('6: A-P->P->P-A')
            MP[i] = MP[21] @ MP[22].T
        elif i == 7:
            events[0].wait()
            events[23].wait()
            logging.info('7: A-P-P-A-P-A')
            MP[i] = MP[23] @ MP[0]
        elif i == 8:
            events[1].wait()
            events[23].wait()
            logging.info('8: A-P-P-A-P-A-P-A')
            MP[i] = MP[23] @ MP[1]
        elif i == 9:
            events[2].wait()
            events[23].wait()
            logging.info('9: A-P-P-A-P-V-P-A')
            MP[i] = MP[23] @ MP[2]
        elif i == 10:
            events[3].wait()
            events[23].wait()
            logging.info('10: A-P-P-A-P-T-P-A')
            MP[i] = MP[23] @ MP[3]
        elif i == 11:
            events[4].wait()
            events[23].wait()
            logging.info('11: A-P-P-A-P->P<-P-A')
            MP[i] = MP[23] @ MP[4]
        elif i == 12:
            events[5].wait()
            events[23].wait()
            logging.info('12: A-P-P-A-P<-P->P-A')
            MP[i] = MP[23] @ MP[5]
        elif i == 13:
            events[0].wait()
            events[23].wait()
            logging.info('13: A-P-A-P-P-A')
            MP[i] = MP[0] @ MP[23]
        elif i == 14:
            events[1].wait()
            events[23].wait()
            logging.info('14: A-P-A-P-A-P-P-A')
            MP[i] = MP[1] @ MP[23]
        elif i == 15:
            events[2].wait()
            events[23].wait()
            logging.info('15: A-P-V-P-A-P-P-A')
            MP[i] = MP[2] @ MP[23]
        elif i == 16:
            events[3].wait()
            events[23].wait()
            logging.info('16: A-P-T-P-A-P-P-A')
            MP[i] = MP[3] @ MP[23]
        elif i == 17:
            events[4].wait()
            events[23].wait()
            logging.info('17: A-P->P<-P-A-P-P-A')
            MP[i] = MP[4] @ MP[23]
        elif i == 18:
            events[5].wait()
            events[23].wait()
            logging.info('18: A-P<-P->P-A-P-P-A')
            MP[i] = MP[5] @ MP[23]
        elif i == 19:
            logging.info('A-P-V')
            MP[i] = W @ P
        elif i == 20:
            logging.info('A-P-T')
            MP[i] = W @ I
        elif i == 21:
            logging.info('A-P->P')
            MP[i] = W @ C
        elif i == 22:
            logging.info('A-P<-P')
            MP[i] = W @ C.T
        elif i == 23:
            events[21].wait()
            logging.info('A-P-P-A')
            MP[23] = MP[21] @ W.T

        events[i].set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    logging.info('Extracting...')

    def get_features(u, v):
        fv = [MP[i][u, v] for i in range(19)]
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
                open('%s/dataset_%d_%d_%d_%d.pkl' % (path, f_beg, f_end, o_beg, o_end), 'wb')
                )


def generate_samples(dataset, observation_begin, observation_end, conf_list, W, C, indexer):
    mapping = indexer.mapping
    written_by = {}
    elements = sparse.find(W)
    for i in range(len(elements[0])):
        u = elements[0][i]
        p = elements[1][i]
        if p in written_by:
            written_by[p].append(u)
        else:
            written_by[p] = [u]

    APPA = W @ C @ W.T
    num_papers = (W @ W.T).diagonal()
    observed_samples = {}

    authors = None
    year = None
    venue = None
    references = []

    paper_threshold = 5

    for line in dataset:
        if not line:
            if year and venue:
                year = int(year)
                if observation_begin < year <= observation_end and authors and references and (venue in conf_list):
                    author_list = authors.split(',')
                    for author in author_list:
                        if author in mapping['author']:
                            u = mapping['author'][author]
                            if num_papers[u] >= paper_threshold:
                                # paper_index = new_indexer.get_index('paper', index)
                                # write_old_new.append((author_id, paper_index))
                                for ref in references:
                                    if ref in mapping['paper']:
                                        paper_id = mapping['paper'][ref]
                                        # paper_index = new_indexer.get_index('paper', index)
                                        # cite_new_old.append((paper_index, paper_id))
                                        if paper_id in written_by:
                                            for v in written_by[paper_id]:
                                                if num_papers[u] >= paper_threshold and APPA[u, v]:
                                                    if (u, v) in observed_samples:
                                                        observed_samples[u, v] = min(year, observed_samples[u, v])
                                                    else:
                                                        observed_samples[u, v] = year

            # index = None
            authors = None
            year = None
            venue = None
            references = []
        else:
            begin = line[1]
            if begin == '@':
                authors = line[2:]
            elif begin == 't':
                year = line[2:]
            elif begin == 'c':
                venue = line[2:]
            elif begin == '%':
                references.append(line[2:])
                # elif begin == 'i':
                #     index = line[6:]

    logging.info('Observed samples found.')
    nonzero = sparse.find(APPA)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}
    N = APPA.shape[0]
    M = len(observed_samples) // 5
    author_list = [i for i in range(N) if num_papers[i] >= paper_threshold]

    while len(censored_samples) < M:
        i = random.randint(0, len(author_list) - 1)
        j = random.randint(0, len(author_list) - 1)
        if i != j:
            u = author_list[i]
            v = author_list[j]
            if (u, v) not in set_observed:
                censored_samples[u, v] = observation_end + 1

    print(len(observed_samples) + len(censored_samples))

    return observed_samples, censored_samples


def main():
    # conf_list_db = [
    #     'KDD', 'PKDD', 'ICDM', 'SDM', 'PAKDD', 'SIGMOD', 'VLDB', 'ICDE', 'PODS', 'EDBT', 'SIGIR', 'ECIR',
    #     'ACL', 'WWW', 'CIKM', 'NIPS', 'ICML', 'ECML', 'AAAI', 'IJCAI',
    # ]

    conf_list_th = [
        'STOC', 'FOCS', 'COLT', 'LICS', 'SCG', 'SODA', 'SPAA', 'PODC', 'ISSAC', 'CRYPTO', 'EUROCRYPT', 'CONCUR',
        'ICALP',
        'STACS', 'COCO', 'WADS', 'MFCS', 'SWAT', 'ESA', 'IPCO', 'LFCS', 'ALT', 'EUROCOLT', 'WDAG', 'ISTCS', 'ISAAC',
        'FSTTCS', 'LATIN', 'RECOMB', 'CADE', 'ISIT', 'MEGA', 'ASIAN', 'CCCG', 'FCT', 'WG', 'CIAC', 'ICCI', 'CATS',
        'COCOON', 'GD',
        'SIROCCO', 'WEA', 'ALENEX', 'FTP', 'CSL', 'DMTCS'
    ]

    with open('data/dblp.txt') as file:
        dataset = file.read().splitlines()

    # ow_set = [
    #     # 3,
    #     6,
    #     # 9
    # ]
    #
    # fw_set = [
    #     5,
    #     10,
    #     # 15
    # ]
    #
    # observation_end = 2016
    # for ow_len in ow_set:
    #     if ow_len == 6:
    #         rel_fw_set = fw_set
    #     else:
    #         rel_fw_set = [10]
    #     for fw_len in rel_fw_set:
    #         feature_begin = observation_end - (fw_len + ow_len)
    #         feature_end = feature_begin + fw_len
    #         observation_begin = feature_end
    #
    #         W, C, I, P, indexer = parse_dataset(dataset, feature_begin, feature_end, conf_list_db)
    #         observed_samples, censored_samples = generate_samples(dataset, observation_begin, observation_end,
    #                                                               conf_list_db, W, C, indexer)
    #         extract_features(feature_begin, feature_end, observation_begin, observation_end, W, C, P, I,
    #                          observed_samples,
    #                          censored_samples)
    #
    #         for t in range(feature_end - 1, feature_begin, -1):
    #             print('=============%d=============' % t)
    #             W, C, I, P, _ = parse_dataset(dataset, feature_begin, t, conf_list_db, indexer)
    #             extract_features(feature_begin, t, observation_begin, observation_end, W, C, P, I, observed_samples,
    #                              censored_samples)

    parse_dataset(dataset, 1995, 2016, conf_list_th)


if __name__ == '__main__':
    main()

import Stemmer
import random
import pickle
import logging
import numpy as np
from scipy import sparse
from nltk.corpus import stopwords as stop_words

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

indices = {'author': 0, 'venue': 0, 'term': 0, 'paper': 0}
mapping = {'author': {}, 'venue': {}, 'term': {}, 'paper': {}}
stopwords = stop_words.words('english')
stem = Stemmer.Stemmer('english')

path = 'all'


def get_index(category, query):
    global indices, mapping
    if query in mapping[category]:
        return mapping[category][query]
    else:
        mapping[category][query] = indices[category]
        indices[category] += 1
        return indices[category] - 1


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


def parse_dataset(filename, feature_begin, feature_end, conf_list):
    write = []
    cite = []
    include = []
    published = []

    with open(filename) as file:
        dataset = file.read().splitlines()

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
                if feature_begin <= year <= feature_end and authors and (venue in conf_list or not conf_list):
                    # min_year = min(year, min_year)
                    # max_year = max(year, max_year)
                    terms = [get_index('term', term) for term in parse_term(title)]
                    references = [get_index('paper', paper_id) for paper_id in references]
                    author_list = authors.split(',')
                    authors = [get_index('author', author_name) for author_name in author_list]
                    venue_id = get_index('venue', venue)
                    paper_index = get_index('paper', index)

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

    num_authors = indices['author']
    num_papers = indices['paper']
    num_venues = indices['venue']
    num_terms = indices['term']

    W = create_sparse(write, num_authors, num_papers)
    C = create_sparse(cite, num_papers, num_papers)
    I = create_sparse(include, num_papers, num_terms)
    P = create_sparse(published, num_papers, num_venues)

    logging.info('Saving...')
    pickle.dump(W, open('temp/write_matrix.pkl', 'wb'))
    pickle.dump(C, open('temp/cite_matrix.pkl', 'wb'))
    pickle.dump(I, open('temp/include_matrix.pkl', 'wb'))
    pickle.dump(P, open('temp/published_matrix.pkl', 'wb'))
    # pickle.dump(indices, open('temp/indices.pkl', 'wb'))
    pickle.dump(mapping, open('temp/mapping.pkl', 'wb'))
    # pickle.dump(APPA, open('APPA.pkl', 'wb'))
    # pickle.dump(APPPA_in, open('APPPA_in.pkl', 'wb'))
    # pickle.dump(APPPA_out, open('APPPA_out.pkl', 'wb'))
    # pickle.dump(APVPA, open('APVPA.pkl', 'wb'))
    # pickle.dump(APTPA, open('APTPA.pkl', 'wb'))

    with open('%s/metadata_%d_%d.txt' % (path, feature_begin, feature_end), 'w') as output:
        output.write('#Authors: %d\n' % num_authors)
        output.write('#Papers: %d\n' % num_papers)
        output.write('#Venues: %d\n' % num_venues)
        output.write('#Terms: %d\n\n' % num_terms)

        output.write('#Write: %d\n' % len(write))
        output.write('#Cite: %d\n' % len(cite))
        output.write('#Publish: %d\n' % len(published))
        output.write('#Contain: %d\n' % len(include))

        # output.write('First Year: %d\n' % min_year)
        # output.write('Last Year: %d\n' % max_year)


def extract_features(f_beg, f_end, o_beg, o_end):
    logging.info('Loading...')
    W = pickle.load(open('temp/write_matrix.pkl', 'rb'))
    C = pickle.load(open('temp/cite_matrix.pkl', 'rb'))
    P = pickle.load(open('temp/include_matrix.pkl', 'rb'))
    I = pickle.load(open('temp/published_matrix.pkl', 'rb'))
    observed_samples = pickle.load(open('temp/observed_samples.pkl', 'rb'))
    censored_samples = pickle.load(open('temp/censored_samples.pkl', 'rb'))

    logging.info('A-P-P-A')
    WC = W @ C
    APPA = WC @ W.T

    logging.info('A-P->P<-P-A')
    APPPA_in = WC @ WC.T

    logging.info('A-P<-P->P-A')
    WCT = W @ C.T
    APPPA_out = WCT @ WCT.T

    logging.info('A-P->V<-P-A')
    WP = W @ P
    APVPA = WP @ WP.T

    logging.info('A-P->T<-P-A')
    WI = W @ I
    APTPA = WI @ WI.T

    logging.info('Extracting...')

    def get_features(u, v):
        fv = [0, 0, 0, 0, 0, 0]
        fv[0] = APPA[u, v]
        fv[1] = APPA[v, u]
        fv[2] = APPPA_in[u, v]
        fv[3] = APPPA_out[u, v]
        fv[4] = APVPA[u, v]
        fv[5] = APTPA[u, v]
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


def generate_samples(filename, observation_begin, observation_end, conf_list):
    """
    :param filename: dataset file name
    :param observation_begin: beginning of the observation window
    :param observation_end: ending of the observation window
    :param conf_list: list of venue names to include
    """

    W = pickle.load(open('temp/write_matrix.pkl', 'rb'))
    # indices = pickle.load(open('indices.pkl', 'rb'))
    mapping = pickle.load(open('temp/mapping.pkl', 'rb'))

    coauthor = W @ W.T
    observed_samples = {}

    with open(filename) as file:
        dataset = file.read().splitlines()

    authors = None
    year = None
    venue = None

    paper_threshold = 5

    for line in dataset:
        # line = line[:-1]
        if not line:
            if year and venue:
                year = int(year)
                if observation_begin <= year <= observation_end and authors and (venue in conf_list or not conf_list):
                    author_list = authors.split(',')
                    # authors = [get_index('author', author_name) for author_name in author_list]
                    for i in range(len(author_list)):
                        if author_list[i] in mapping['author']:
                            u = mapping['author'][author_list[i]]
                            n_u = coauthor[u, u]
                            if n_u >= paper_threshold:
                                for j in range(i + 1, len(author_list)):
                                    if author_list[j] in mapping['author']:
                                        v = mapping['author'][author_list[j]]
                                        n_v = coauthor[v, v]
                                        if n_v >= paper_threshold and not coauthor[u, v]:
                                            if (u, v) in observed_samples:
                                                observed_samples[u, v] = min(year, observed_samples[u, v])
                                            else:
                                                observed_samples[u, v] = year

            authors = None
            year = None
            venue = None
        else:
            begin = line[1]
            if begin == '@':
                authors = line[2:]
            elif begin == 't':
                year = line[2:]
            elif begin == 'c':
                venue = line[2:]

    logging.info('Observed samples found.')
    nonzero = sparse.find(coauthor)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}
    N = coauthor.shape[0]
    M = len(observed_samples) // 5
    author_list = [i for i in range(N) if coauthor[i, i] >= paper_threshold]

    while len(censored_samples) < M:
        i = random.randint(0, len(author_list) - 1)
        u = author_list[i]
        # n_u = coauthor[u,u]
        # if n_u >= paper_threshold:
        try:
            j = random.randint(i + 1, len(author_list) - 1)
            v = author_list[j]
            # n_v = coauthor[v,v]
            if (u, v) not in set_observed:
                censored_samples[u, v] = observation_end + 1
        except ValueError:
            pass

    pickle.dump(observed_samples, open('temp/observed_samples.pkl', 'wb'))
    pickle.dump(censored_samples, open('temp/censored_samples.pkl', 'wb'))

    print(len(observed_samples) + len(censored_samples))


def main():
    conf_list = [
        'KDD', 'PKDD', 'ICDM', 'SDM', 'PAKDD', 'SIGMOD', 'VLDB', 'ICDE', 'PODS', 'EDBT', 'SIGIR', 'ECIR',
                 'ACL', 'WWW', 'CIKM', 'NIPS', 'ICML', 'ECML', 'AAAI', 'IJCAI',

        'STOC', 'FOCS', 'COLT', 'LICS', 'SCG', 'SODA', 'SPAA', 'PODC', 'ISSAC', 'CRYPTO', 'EUROCRYPT', 'CONCUR',
        'ICALP',
        'STACS', 'COCO', 'WADS', 'MFCS', 'SWAT', 'ESA', 'IPCO', 'LFCS', 'ALT', 'EUROCOLT', 'WDAG', 'ISTCS', 'ISAAC',
        'FSTTCS', 'LATIN', 'RECOMB', 'CADE', 'ISIT', 'MEGA', 'ASIAN', 'CCCG', 'FCT', 'WG', 'CIAC', 'ICCI', 'CATS',
        'COCOON', 'GD',
        'SIROCCO', 'WEA', 'ALENEX', 'FTP', 'CSL', 'DMTCS'
    ]

    feature_begin = 1980
    feature_end = 2000
    observation_begin = 2001
    observation_end = 2016
    parse_dataset('data/dblp.txt', feature_begin, feature_end, conf_list)
    generate_samples('data/dblp.txt', observation_begin, observation_end, conf_list)
    extract_features(feature_begin, feature_end, observation_begin, observation_end)

    for t in range(feature_end - 1, feature_begin, -1):
        print('===========================')
        parse_dataset('data/dblp.txt', feature_begin, t, conf_list)
        extract_features(feature_begin, t, observation_begin, observation_end)


if __name__ == '__main__':
    main()

import numpy as np
from datetime import datetime
from scipy import sparse


class Indexer:
    def __init__(self, nodes):
        self.indices = {node: 0 for node in nodes}
        self.mapping = {node: {} for node in nodes}

    def index(self, category, query):
        self.mapping[category][query] = self.indices[category]
        self.indices[category] += 1
        return self.indices[category] - 1

    def get_index(self, category, query):
        if query in self.mapping[category]:
            return self.mapping[category][query]
        else:
            return self.index(category, query)


def create_sparse(coo_list, m, n):
    data = np.ones((len(coo_list),))
    row = [pair[0] for pair in coo_list]
    col = [pair[1] for pair in coo_list]
    matrix = sparse.coo_matrix((data, (row, col)), shape=(m, n))
    return matrix


def date_subtractor(begin_time, end_time):
    return end_time.timestamp() - begin_time.timestamp()


def timestamp_delta_generator(days=0, months=0, years=0):
    days_delta_timestamp_unit = date_subtractor(datetime(2006, 1, 1), datetime(2006, 1, 2))
    months_delta_timestamp_unit = date_subtractor(datetime(2006, 1, 1), datetime(2006, 2, 1))
    years_delta_timestamp_unit = date_subtractor(datetime(2006, 1, 1), datetime(2007, 1, 1))

    return days * days_delta_timestamp_unit + months * months_delta_timestamp_unit + years_delta_timestamp_unit * years

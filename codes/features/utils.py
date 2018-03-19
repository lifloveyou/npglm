import numpy as np
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

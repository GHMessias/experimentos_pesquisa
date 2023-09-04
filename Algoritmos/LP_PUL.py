from Auxiliares.requirements import *


def mst_graph(X):
    """Returns Minimum Spanning Tree (MST) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed mst graph.
    """
    D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return csr_matrix(adj)

class LP_PUL:
    def __init__(self, graph, data, positives, unlabeled):
        self.graph = graph
        self.data = data
        self.positives = positives
        self.unlabeled = unlabeled



    def train(self):
        self.a = np.zeros(len(self.unlabeled) + len(self.positives))
        
        if not nx.is_connected(self.graph):
            adj = nx.to_scipy_sparse_array(self.graph)  # Convert to sparse matrix
            adj_aux = mst_graph(self.data).toarray()
            
            rows, cols = np.where((adj.toarray() == 0) & (adj_aux == 1))
            
            for i, j in zip(rows, cols):
                self.graph.add_edge(i, j)

        d = np.zeros(len(self.unlabeled) + len(self.positives))

        for p in self.positives:
            for u in self.unlabeled:
                #print(f'computing shortest path length {u}/{p}')
                d_u = nx.shortest_path_length(self.graph,p,u)
                d[u] = d_u
            self.a += d
            d = np.zeros(len(self.unlabeled) + len(self.positives))

        
        self.a = self.a / len(self.positives)
        

    def negative_inference(self, num_neg):
        RN = [x for _, x in sorted(zip(self.a, range(len(self.data))), reverse=True)][:num_neg]
        return RN

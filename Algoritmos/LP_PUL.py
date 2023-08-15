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
        self.distance_vector = np.zeros(len(self.unlabeled))
        # Verificando se o grafo é conexo, se não for, cria uma MST e soma as arestas novas no grafo
        if not nx.is_connected(self.graph):
            adj = nx.adjacency_matrix(self.graph)
            adj_aux = mst_graph(self.data).toarray()
            aux_graph = nx.DiGraph()
            aux_graph.add_nodes_from(range(len(adj_aux)))
            for i in range(len(adj_aux)):
                for j in range(len(adj_aux)):
                    if adj[i,j] == 0 and adj_aux[i,j] == 1:
                        self.graph.add_edge(i,j)
        
        d = list()

        for p in self.positives:
            for u in self.unlabeled:
                #print(f'computing shortest path length {u}/{p}')
                d_u = nx.shortest_path_length(self.graph,p,u)
                d.append(d_u)
            self.distance_vector += d
            d = list()

        for i in range(len(self.distance_vector)):
            self.distance_vector[i] = self.distance_vector[i] / len(self.positives)
        

    def negative_inference(self, num_neg):
        RN = [x for _, x in sorted(zip(self.distance_vector, self.unlabeled), reverse=True)][:num_neg]
        return RN

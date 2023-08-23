# Funções auxiliares
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
from networkx import shortest_path_length
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score


# def matrizPeso(G, P):
#     '''
#     G: Graph
#     P: List of positive nodes

#     return the adjacency matrix of G with weighted edges in the shortest path between every two nodes of P
#     '''
#     A = nx.adjacency_matrix(G, nodelist=range(len(G.nodes()))).todense()
#     for index, value in enumerate(P):
#         for j in P[index + 1:]:  # Avoids re-computing paths and self-loops
#             try:
#                 node_list = nx.dijkstra_path(G, value, j)
#                 for u, v in zip(node_list[:-1], node_list[1:]):
#                     A[u, v] += 1
#             except nx.NetworkXNoPath:
#                 # Handle the case when there is no path between value and j
#                 pass

#     return torch.tensor(A, dtype=torch.float64)

# def degree_matrix(G):
#     D = torch.zeros((len(G.nodes()),len(G.nodes())), dtype = torch.float64)

#     for i in G.nodes():
#         D[i][i] = G.degree[i]

#     return D

# def inverse_sqroot_matrix(D):
#     D_clone = D.clone()
#     D_clone[D_clone != 0] = D_clone[D_clone != 0].pow(-0.5)
#     return D_clone
    

# def message_passing_PUL(D_tilde, A_tilde, X, C, function):
#     result = function(torch.matmul(torch.matmul(torch.matmul(D_tilde, (A_tilde + C)), D_tilde), X))
#     return result

# def relu(A):
#     return torch.maximum(A, torch.tensor(0.0))

# def feature_matrix(G, P, y):
#     # Criando a matriz de características
#     n_features = 10
#     n_elements = len(G.nodes())

#     X = torch.zeros((n_elements, n_features))

#     for i in range(n_elements):
#         if y[i] == 1:
#             X[i] = torch.rand(1)  # Valores entre 0 e 1 (distribuição uniforme)
#         else:
#             X[i] = -torch.rand(1)  # Valores entre -1 e 0 (distribuição uniforme)

#     # Adicionando ruído às entradas
#     noise = torch.normal(mean=0, std=0.1, size=(n_elements, n_features))  # Distribuição normal com média 0 e desvio padrão 0.1
#     X += noise
#     return X

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

def graph_from_adjacency_matrix(adj_matrix):
    """
    Gera um grafo a partir de uma matriz de adjacência.

    Parâmetros:
        adj_matrix (numpy.ndarray): Matriz de adjacência do grafo.

    Retorna:
        networkx.Graph: O grafo gerado a partir da matriz de adjacência.
    """
    # Certifique-se de que a matriz de adjacência seja do tipo numpy.ndarray
    if not isinstance(adj_matrix, np.ndarray):
        raise ValueError("A matriz de adjacência deve ser um numpy.ndarray.")

    # Verifique se a matriz de adjacência é quadrada
    num_rows, num_cols = adj_matrix.shape
    if num_rows != num_cols:
        raise ValueError("A matriz de adjacência deve ser quadrada (mesmo número de linhas e colunas).")

    # Crie o grafo a partir da matriz de adjacência
    graph = nx.Graph()
    for i in range(num_rows):
        for j in range(num_cols):
            if adj_matrix[i, j] != 0:
                graph.add_edge(i, j, weight=adj_matrix[i, j])

    return graph

# def ordenar_lista_de_acordo_com_outra(elements, loss):
#     # Combine as duas listas usando zip
#     lista_combinada = list(zip(loss, elements))

#     # Classifique a lista combinada com base na ordem da primeira lista
#     lista_combinada.sort(key=lambda x: x[0])

#     # Extraia a lista ordenada originalmente
#     lista_ordenada = [item[1] for item in lista_combinada]

#     return lista_ordenada


def nearest_nodes(P, i, G, k = 3):
    '''
    For every i in P, compute the distance using dijkstra algorithm for every j in P. Return a tuple with the 3 nearest elements (lowest values of dijkstra), from i to the result.
    if there is no path between i and j, the distance is infinity
    '''
    if k >= len(P):
        k = len(P) - 1
    distances = []
    for j in P:
        if i != j:
            try:
                distances.append((nx.dijkstra_path_length(G, i, j), j))
            except:
                distances.append((np.inf, j))
    distances.sort()
    return [distances[i][1] for i in range(k)]

def connect_nearest_nodes(G, P, k = 3):
    '''
    For every node in P, compute the nearest node in P and connect them using G.add_nodes_from
    '''
    if k >= len(P):
        k = len(P) - 1
    for i in P:
        nearest = nearest_nodes(P, i, G, k)
        for j in nearest:
            G.add_edge(i, j)
    return G

def strong_connect_positives(P, edge_index, m):
    '''
    for every node i,j if i and j are positives, then, the edge_weight between i and j is m
    '''
    edge_weight = torch.zeros(edge_index.shape[1])
    for index, src, tgt in zip(range(len(edge_index[0])), edge_index[0], edge_index[1]):
        if src in P and tgt in P:
            edge_weight[index] = m
    return edge_weight

def compute_accuracy(y, infered_elements):
    y_pred = np.zeros(len(infered_elements))
    y_true = np.array([y[i] for i in infered_elements])[:len(infered_elements)]
    # print(y_true)
    # print(y_pred)

    return accuracy_score(y_true, y_pred)

def compute_f1_score(y, infered_elements):
    y_pred = np.zeros(len(infered_elements))
    y_true = np.array([y[i] for i in infered_elements])[:len(infered_elements)]
    # print(y_true)
    # print(y_pred)

    return f1_score(y_true, y_pred,pos_label = 0)

def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum(tensor1 - tensor2) ** 2)
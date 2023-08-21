import warnings

warnings.filterwarnings('ignore')

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *
from Algoritmos.AE_PUL import autoencoder_PUL_model
from Algoritmos.GAE_PUL import graphautoencoder_PUL_model
from torch_geometric.nn import GCNConv


# def edge_weight_positives(G, edge_index, P):
#     edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
#     for indexi, i in enumerate(P):
#         for indexj, j in enumerate(P):
#             print(f'{indexi} / {len(P)} -- {indexj} / {len(P)}')
#             if i != j:
#                 short_path = nx.shortest_path(G, i, j)
#                 for index in range(len(short_path) - 1):
#                     for e_idx, (src, tgt) in enumerate(edge_index.T):
#                         #if (src == short_path[index] and tgt == short_path[index + 1]) or (src == short_path[index + 1] and tgt == short_path[index]):
#                         if (src == short_path[index] and tgt == short_path[index + 1]):
#                             edge_weight[e_idx] += 1
#     return edge_weight

def edge_weight_positives(G, edge_index, P):
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    paths = dict(nx.all_pairs_shortest_path(G))  # Convert generator to dictionary

    for i in P:
        for j in P:
            if i != j:
                short_path = paths[i][j]
                for index in range(len(short_path) - 1):
                    # Find the edge index in one of the directions
                    src = short_path[index]
                    tgt = short_path[index + 1]
                    mask = (edge_index[0] == src) & (edge_index[1] == tgt)
                    edge_weight[mask] += 1

                    # If the graph is undirected, also consider the reverse direction
                    if not G.is_directed():
                        mask = (edge_index[0] == tgt) & (edge_index[1] == src)
                        edge_weight[mask] += 1

    return edge_weight


def connect_graph(data, G):
    if not nx.is_connected(G):
        n = len(G.nodes())
        A = mst_graph(data).toarray()
        adj = nx.adjacency_matrix(G).toarray()
        # for every u,v in A, if A[u,v] == 1 and adj[u,v] == 0 then, adj[u,v] = 1
        for i in range(n):
            for j in range(n):
                if A[i][j] == 1 and adj[i][j] == 0:
                    adj[i][j] = 1
        adj = np.matrix(adj)
        _G = nx.DiGraph()
        num_nodes = len(adj)
        for node in range(num_nodes):
            G.add_node(node)  # Add nodes to the graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] == 1:
                    _G.add_edge(i, j)
        _G = _G.to_undirected()

        return _G
    return G

def connect_positives(edge_index, P):
    src_list = edge_index[0].tolist()
    tgt_list = edge_index[1].tolist()
    for i in P:
        for j in P:
            if i != j:
                src_list.append(i)
                tgt_list.append(j)
                # src_list.append(j)
                # tgt_list.append(i)
    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    return edge_index

def strong_connect_positives(edge_index, P):
    edge_weight = torch.ones((edge_index.shape[1],), dtype=torch.float32)
    
    # Create masks for source and target nodes in P
    src_mask = torch.tensor([src in P for src in edge_index[0]])
    tgt_mask = torch.tensor([tgt in P for tgt in edge_index[1]])
    
    # Combine masks to find edges with both source and target in P
    positive_mask = src_mask & tgt_mask
    
    # Update edge weights for positive edges
    edge_weight[positive_mask] += np.sqrt(5) - 1
    
    return edge_weight


# edge_index = np.array([
#     [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6],
#     [1, 2, 0, 2, 3, 0, 1, 3, 6, 0, 2, 4, 3, 4, 5, 4, 2, 4]
# ])

# G = nx.DiGraph()
# for src, tgt in edge_index.T:
#     G.add_edge(src, tgt)

# P = [0, 6]
# edge_weights = strong_connect_positives(connect_positives(edge_index, P), P)
# print("Edge Weights:", edge_weights)

dataset = Planetoid(root = 'Datasets', name = "Cora", transform=NormalizeFeatures())
data = dataset[0]

G = to_networkx(data, to_undirected=True)
#adj = nx.adjacency_matrix(G).toarray()
X = data.x.double()
Y = data.y

G = connect_graph(X, G)
edges = list(G.edges())

nodes_from_edges = [edge[0] for edge in edges]
nodes_to_edges = [edge[1] for edge in edges]

edge_index = torch.tensor([nodes_from_edges, nodes_to_edges], dtype=torch.long)

# CORA
pul_label = [0,1,2,4]
# pul_label = [3,5,6]
# CiteSeer
#pul_label = [2,3,4]

# pul_label = [1]

Y = torch.tensor([1 if x in pul_label else 0 for x in Y])

all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]

class Regularized_GAE(torch.nn.Module):
    def __init__(self, in_channel, hid_channel1, hid_channel2):
        super(Regularized_GAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channel, hid_channel1),
            nn.ReLU(),
            nn.Linear(hid_channel1, hid_channel2),
            nn.ReLU()
        )
        self.conv1 = GCNConv(hid_channel2, hid_channel2)
        self.conv2 = GCNConv(hid_channel2, hid_channel2)
        self.decoder = nn.Sequential(
            nn.Linear(hid_channel2, hid_channel1),
            nn.ReLU(),
            nn.Linear(hid_channel1, in_channel),
            nn.ReLU()
        )    

    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.decoder(x)
        return x
      
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

epochs = list(range(1,40))

print(nx.is_directed(G))
positive_rate = 0.05
for i in range(10):
    positives = random.sample(all_positives, int(positive_rate * len(all_positives)))
    unlabeled = list(set(range(len(G.nodes()))) - set(positives))   

    # print('conectando positivos')
    # edge_index = connect_positives(edge_index, positives)
    # # print('calculando strong connecting')
    # edge_weights = strong_connect_positives(edge_index, positives)
    # print('finalizado')
    # edge_weights = edge_weight_positives(G, edge_index, positives)
    # edge_weights = torch.sqrt(edge_weights)
    model1 = Regularized_GAE(in_channel = X.shape[1], hid_channel1=256, hid_channel2=128)
    model2 = Autoencoder(input_size = X.shape[1], hidden_size1 = 256, hidden_size2 = 128)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr = 0.01)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr = 0.01)

    print(len(positives))
    for epoch in epochs:
        GAE_classifier = graphautoencoder_PUL_model(model = model1,
                                                    optimizer = optimizer1,
                                                    epochs = epoch,
                                                    data = X,
                                                    positives = positives,
                                                    unlabeled = unlabeled,
                                                    edge_index=edge_index,
                                                    edge_weight=None)
        GAE_classifier.train()
        RN_GAE = GAE_classifier.negative_inference(num_neg = 200)
        # print(f'GAE: quantidade de epocas de treinamento {epoch} \t acurácia {compute_accuracy(Y, RN_GAE)}')
        model1 = Regularized_GAE(in_channel = X.shape[1], hid_channel1=256, hid_channel2=128)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr = 0.001)

        AE_classifier = autoencoder_PUL_model(model = model2, optimizer = optimizer2, epochs = epoch, data = X, positives = positives, unlabeled = unlabeled)
        AE_classifier.train()
        RN_AE = AE_classifier.negative_inference(num_neg = 200)
        # print(f'AE : quantidade de epocas de treinamento {epoch} \t acurácia {compute_accuracy(Y, RN_AE)}')
        model2 = Autoencoder(input_size = X.shape[1], hidden_size1 = 256, hidden_size2 = 128)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr = 0.01)

        print(f'Quantidade de epocas de treinamento: {epoch}, \t acurária GAE: {compute_accuracy(Y, RN_GAE)} \t acurácia AE: {compute_accuracy(Y, RN_AE)}')




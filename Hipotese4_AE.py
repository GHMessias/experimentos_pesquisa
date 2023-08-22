import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *
from Algoritmos.AE_PUL import autoencoder_PUL_model
from Algoritmos.GAE_PUL import graphautoencoder_PUL_model

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        # Initialize the Linear layer with the identity matrix
        self.lin = Linear(in_channels, out_channels, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(in_channels, out_channels))
        for param in self.lin.parameters():
            param.requires_grad = False  # Make the parameters non-trainable
            # print(param)
        
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_weight):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class GAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(GAutoencoder, self).__init__()
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
        
        # Create an instance of GCNConv for message passing
        self.gcn_layer = GCNConv(hidden_size2, hidden_size2)
    
    def forward(self, x, edge_index, edge_weight):
        encoded = self.encoder(x)
        # messaged = self.gcn_layer(encoded, edge_index, edge_weight)
        decoded = self.decoder(encoded)
        # decoded = self.decoder(messaged)       
        return decoded
    
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

Y = torch.tensor([1 if x in pul_label else 0 for x in Y])

all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]

positive_rate = 0.05

positives = random.sample(all_positives, int(positive_rate * len(all_positives)))
unlabeled = list(set(range(len(G.nodes()))) - set(positives))  

model = GAutoencoder(X.shape[1], 256, 128)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 4

print('conectando positivos')
edge_index = connect_positives(edge_index, positives)
print('calculando strong connecting')
edge_weights = strong_connect_positives(edge_index, positives)


GAE_classifier = graphautoencoder_PUL_model(model = model,
                                            optimizer = optimizer,
                                            epochs = epochs,
                                            data = X,
                                            positives = positives,
                                            unlabeled = unlabeled,
                                            edge_index=edge_index,
                                            edge_weight=edge_weights)
GAE_classifier.train()
RN_GAE = GAE_classifier.negative_inference(num_neg = 200)
print(f'GAE: quantidade de epocas de treinamento {epochs} \t acurácia {compute_accuracy(Y, RN_GAE)}')
import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        
        # Initialize the Linear layer with the identity matrix
        self.lin = Linear(in_channels, out_channels, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(in_channels, out_channels))
        for param in self.lin.parameters():
            param.requires_grad = False  # Make the parameters non-trainable
            print(param)
        
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, x, edge_index):
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

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            # nn.Linear(hidden_size1, hidden_size2),
            # nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.Sigmoid(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        )
        
        # Create an instance of GCNConv for message passing
        self.gcn_layer = GCNConv(hidden_size1, hidden_size1)
    
    def forward(self, x, edge_index):
        encoded = self.encoder(x)
        # messaged = self.gcn_layer(encoded, edge_index)
        # messaged = F.sigmoid(messaged)
        decoded = self.decoder(messaged)       
        return decoded


# Lista de arestas (pares de nós)
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
edge_list = torch.tensor(edges, dtype=torch.long).t()

# Criando um grafo não direcionado
G = nx.Graph()
G.add_edges_from(edges)

# Desenhando o grafo
# nx.draw(G, with_labels=True, node_size=1000, node_color="skyblue", font_color="black")

# plt.show()
 
# X = torch.tensor((6 * [0],
#                   6 * [1],
#                   6 * [2],
#                   6 * [3],
#                   6 * [4],
#                   6 * [5])).double()

# mean = X.mean(dim=0)  # Calculate mean along rows
# std = X.std(dim=0)    # Calculate standard deviation along rows

# X = (X - mean) / std

X = torch.rand((6,6))
print(X)

model = Autoencoder(6, 4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    encoded = model.encoder(X)
    messaged = model.gcn_layer(encoded, edge_list)
    decoded = model.decoder(messaged)
    loss = F.mse_loss(X, decoded)
    loss.backward()
    # print(f'loss = {F.mse_loss(X, model(X, edge_list)).item()}')
    optimizer.step()

print(model(X, edge_list))

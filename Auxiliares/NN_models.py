import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(GraphAutoEncoder, self).__init__()
        self.encode1 = nn.Linear(input_dim, hidden_dim1)
        self.encode2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.message_passing1 = GCNConv(hidden_dim2, hidden_dim2)
        self.message_passing2 = GCNConv(hidden_dim2, hidden_dim2)
        self.decode1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.decode2 = nn.Linear(hidden_dim1, input_dim)

    def forward(self, x, edge_index, edge_weight = None):
        x = self.encode1(x)
        x = nn.functional.relu(x)
        x = self.encode2(x)
        x = nn.functional.relu(x)
        x = self.message_passing1(x, edge_index, edge_weight)
        x = nn.functional.relu(x)
        x = self.message_passing2(x, edge_index, edge_weight)
        x = nn.functional.relu(x)
        x = self.decode1(x)
        x = nn.functional.relu(x)
        x = self.decode2(x)
        return nn.functional.sigmoid(x)
    

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(AutoEncoder, self).__init__()
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


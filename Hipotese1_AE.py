from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *
from Algoritmos.AE_PUL import autoencoder_PUL_model

dataset = Planetoid(root = 'Datasets', name = "Cora", transform=NormalizeFeatures())
data = dataset[0]

G = to_networkx(data, to_undirected=True)
adj = nx.adjacency_matrix(G).toarray()
X = data.x.double()
Y = data.y
# CORA
pul_label = [0,1,2,4]
# CiteSeer
# pul_label = [2,3,4]

Y = torch.tensor([1 if x in pul_label else 0 for x in Y])

all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]

A = torch.tensor(nx.adjacency_matrix(G).todense())
D = degree_matrix(G)
A_tilde = A + torch.eye(len(G.nodes()))
A_tilde = A_tilde.double()
D_tilde = inverse_sqroot_matrix(D)



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
    

class Regularized_GAE(torch.nn.Module):
    def __init__(self, in_channel, hid_channel1, hid_channel2, D_tilde, C):
        super(Regularized_GAE, self).__init__()
        self.D_tilde = D_tilde
        self.C = C
        self.encoder = nn.Sequential(
            nn.Linear(in_channel, hid_channel1),
            nn.ReLU(),
            nn.Linear(hid_channel1, hid_channel2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_channel2, hid_channel1),
            nn.ReLU(),
            nn.Linear(hid_channel1, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # Message Passing considering the shortest path of a pair of positive nodes
        x = torch.matmul(torch.matmul(torch.matmul(self.D_tilde, self.C), self.D_tilde), x)
        x = F.relu(x)
        x = torch.matmul(torch.matmul(torch.matmul(self.D_tilde, self.C), self.D_tilde), x)
        x = F.relu(x)
        # x = torch.matmul(torch.matmul(torch.matmul(self.D_tilde, self.C), self.D_tilde), x)
        # x = F.relu(x)
        x = self.decoder(x)
        return x


epochs = [1,2,3,4, 5, 10]


positive_rate = 0.25
for i in range(10):
    positives = random.sample(all_positives, int(positive_rate * len(all_positives)))
    unlabeled = list(set(range(len(G.nodes()))) - set(positives))   
    print('aaaaaaaaaaaaaaaa')
    C = matrizPeso(G, positives)
    C = torch.sqrt(C)

    model1 = Regularized_GAE(in_channel = X.shape[1], hid_channel1=256, hid_channel2=128, D_tilde = D_tilde, C = C)
    model2 = Autoencoder(input_size = X.shape[1], hidden_size1 = 256, hidden_size2 = 128)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr = 0.01)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr = 0.01)

    print(len(positives))
    for epoch in epochs:
        GAE_classifier = autoencoder_PUL_model(model = model1, optimizer = optimizer1, epochs = epoch, data = X, positives = positives, unlabeled = unlabeled)
        GAE_classifier.train()
        RN_GAE = GAE_classifier.negative_inference(num_neg = 200)
        print(f'GAE: quantidade de epocas de treinamento {epoch} \t acurácia {compute_accuracy(Y, RN_GAE)}')
        model1 = Regularized_GAE(in_channel = X.shape[1], hid_channel1=256, hid_channel2=128, D_tilde = D_tilde, C = C)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr = 0.01)

        AE_classifier = autoencoder_PUL_model(model = model2, optimizer = optimizer2, epochs = epoch, data = X, positives = positives, unlabeled = unlabeled)
        AE_classifier.train()
        RN_AE = AE_classifier.negative_inference(num_neg = 200)
        print(f'AE : quantidade de epocas de treinamento {epoch} \t acurácia {compute_accuracy(Y, RN_AE)}')
        model2 = Autoencoder(input_size = X.shape[1], hidden_size1 = 256, hidden_size2 = 128)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr = 0.01)




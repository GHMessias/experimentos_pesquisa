from Auxiliares.auxiliar_functions import *
from Auxiliares.requirements import *


df = pd.read_csv('Datasets/Ionosphere/ionosphere.data')

X = torch.tensor(df.iloc[:, :-1].values, dtype = torch.float32)
Y = torch.tensor([1 if x == 'g' else 0 for x in df.iloc[:, -1].values])

pul_labels = [1]
Y = torch.tensor([1 if x in pul_labels else 0 for x in Y])

# Criando um grafo a partir da matriz de proximidade Kneighbors
G = graph_from_adjacency_matrix(kneighbors_graph(X, 3, mode='distance', include_self=False).todense())

# Gerando os dados positivos e negativos
all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]

positives = random.sample(all_positives, int(0.01 * len(all_positives)))
unlabeled = list(set(range(len(Y))) - set(positives))

pul_mask = torch.zeros(len(Y))
for i in positives:
    pul_mask[i] = 1
pul_mask = pul_mask.to(dtype = torch.bool)

# Modelo AutoEncoder Padrão

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

model = Autoencoder(X.shape[1], 4)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


def train(model):
    # model.double()
    model.train()
    optimizer.zero_grad()
    F.mse_loss(X[pul_mask], model(X)[pul_mask]).backward()
    optimizer.step

output_ = model(X)
loss = [F.mse_loss(X[i], output_[i]) for i in unlabeled]
pares = list(zip(loss, unlabeled))
pares_ordenados = sorted(pares, reverse=True)
RN = [par[1] for par in pares_ordenados][:50]
print(compute_accuracy(Y, RN))



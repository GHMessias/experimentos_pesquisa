from Auxiliares.auxiliar_functions import *
from Auxiliares.requirements import *


df = pd.read_csv('Datasets/Ionosphere/ionosphere.data')

X = torch.tensor(df.iloc[:, :-1].values, dtype = torch.float64)
Y = torch.tensor([1 if x == 'g' else 0 for x in df.iloc[:, -1].values], dtype = torch.float64)

train_mask = torch.zeros(len(Y))
train_examples = random.sample(range(len(X)), int(0.7 * len(X)))
for i in train_examples:
    train_mask[i] = 1
train_mask = train_mask.to(dtype = torch.bool)
test_mask = torch.tensor([not x for x in train_mask])


# Use o método "bincount" do torch para contar ocorrências de cada valor
# count = torch.bincount(Y)

# # O índice do array "count" representa o valor e o valor no índice representa a contagem
# for value, occurrences in enumerate(count):
#     print(f"Valor {value}: {occurrences} ocorrências")
#     print(f'porcentagem de {value}: {occurrences / len(Y)}')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(NeuralNet, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )
    
    def forward(self, x):
        feedforward = self.feedforward(x)
        return feedforward
    
model = NeuralNet(X.shape[1], 8, 1)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

def train(model):
    model.double()
    model.train()
    optimizer.zero_grad()
    F.mse_loss(Y[train_mask], model(X)[train_mask]).backward()
    optimizer.step

train(model)
Y = Y.to(int)
print(accuracy_score(Y[test_mask].detach().numpy(), model(X)[test_mask].to(int).detach().numpy()))
import sys
sys.path.append("G:\Meu Drive\Mestrado\Pesquisa\paper - AE como medida de similaridade\EXPERIMENTOS_PESQUISA_FINAL")

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *

from Algoritmos.LP_PUL import LP_PUL
from Algoritmos.AE_PUL import autoencoder_PUL_model
from Algoritmos.CCRNE import CCRNE
from Algoritmos.MCLS import MCLS
from Algoritmos.PU_LP import PU_LP
from Algoritmos.RCSVM_RN import RCSVM_RN

import warnings
warnings.filterwarnings("ignore")
print('warnings ignorados')

################################################################################
# CORA dataset #
################################################################################


dataset = Planetoid(root = "Datasets", name = "Cora", transform=NormalizeFeatures())
data = dataset[0]

# transformando o arquivo data em um grafo networkx
G = to_networkx(data, to_undirected=True)
adj = nx.adj_matrix(G).toarray()

# Criando uma variável para armazenar as features
X = data.x.double()
Y = data.y
pul_label = [0,1,2,4]
print(type(Y))

# Use o método "bincount" do torch para contar ocorrências de cada valor
count = torch.bincount(Y)

# O índice do array "count" representa o valor e o valor no índice representa a contagem
for value, occurrences in enumerate(count):
    print(f"Valor {value}: {occurrences} ocorrências")

Y = torch.tensor([1 if label in pul_label else 0 for label in Y])
print('shape dos dados', X.shape)


# Gerando os dados positivos e negativos
all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]

#print(f' positives : {positives}')
#print(unlabeled)

# Definindo os elementos da regularização

A = torch.tensor(nx.adjacency_matrix(G).todense())
D = degree_matrix(G)
A_tilde = A + torch.eye(len(G.nodes()))
A_tilde = A_tilde.double()
D_tilde = inverse_sqroot_matrix(D)


# Criando os modelos de regularização para grafos e autoencoder padrão
# Modelo Graph AutoEncoder com regularização
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
        x = self.decoder(x)
        return x
    

# Modelo AutoEncoder Padrão

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

# Instanciando os algoritmos (ainda é necessário fazer o teste paramétrico dos outros modelos)
    
positive_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.20, 0.25]
auto_inference_algorithms = ['CCRNE', 'PU_LP', 'RCSVM_RN']
num_neg = 200

algoritmo_list = []
iteracao_list = []
rate_list = []
acc_list = []
f1_list = []
RN_len = []
tempo = []

for rate in positive_rate:
    for i in range(10):
        
        print(f'iteração {i}')
        positives = random.sample(all_positives, int(rate * len(all_positives)))
        unlabeled = list(set(range(len(G.nodes()))) - set(positives))
        C = matrizPeso(G, positives)
        C = torch.sqrt(C)


        pul_mask = torch.tensor([1 if i in positives else 0 for i in range(len(G.nodes()))], dtype = torch.bool)

        model_RGAE = Regularized_GAE(in_channel = X.shape[1], hid_channel1 = 256, hid_channel2 = 128, D_tilde = D_tilde, C = C)
        model_AE = Autoencoder(input_size = X.shape[1], hidden_size1 = 256, hidden_size2 = 128)

        optimizer_RGAE = torch.optim.Adam(model_RGAE.parameters(), lr=0.01)
        optimizer_AE = torch.optim.Adam(model_AE.parameters(), lr = 0.01)


        algorithms_cora = {
        'LP_PUL' : LP_PUL(graph = G, data = X, positives = positives, unlabeled = unlabeled),
        'CCRNE' : CCRNE(data = X, positives = positives, unlabeled = unlabeled),
        'AE_PUL' : autoencoder_PUL_model(model = model_AE, optimizer = optimizer_AE, epochs = 100, data = X, positives = positives, unlabeled = unlabeled),
        'GAE_PUL' : autoencoder_PUL_model(model = model_RGAE, optimizer = optimizer_RGAE, epochs = 100, data = X, positives = positives, unlabeled = unlabeled),
        'MCLS' : MCLS(data = X, positives = positives, k = 7, ratio = 0.1),
        'PU_LP' : PU_LP(graph = G, positives = positives, unlabeled = unlabeled, alpha = 0.3, m = 10, l = 5, k = 5),
        'RCSVM_RN' : RCSVM_RN(data = X, positives = positives, unlabeled = unlabeled, alpha = 0.1, beta = 0.1)
        }
        for algorithm in algorithms_cora:
            start_time = time.time()
            print(f'algoritmo {algorithm}')
            print(f'tamanho do dataset positivo {len(positives)}')
            algorithms_cora[algorithm].train()
            if algorithm in auto_inference_algorithms:
                RN = algorithms_cora[algorithm].negative_inference()
            else:
                RN = algorithms_cora[algorithm].negative_inference(num_neg)
            end_time = time.time()
            tempo.append(end_time - start_time)
            acc = compute_accuracy(Y, RN)
            print(acc)
            f1 = compute_f1_score(Y, RN)
            algoritmo_list.append(algorithm)
            iteracao_list.append(i)
            rate_list.append(rate)
            acc_list.append(acc)
            f1_list.append(f1)
            RN_len.append(len(RN))

            # Criando o DataFrame com as listas preenchidas
            df = pd.DataFrame({
                'Algoritmo': algoritmo_list,
                'Iteração': iteracao_list,
                'Rate': rate_list,
                'acc': acc_list,
                'f1': f1_list,
                'RN len' : RN_len,
                'tempo de execução': tempo
            })

            df.to_csv('Resultados/dataframe_CORA.csv', index = False)
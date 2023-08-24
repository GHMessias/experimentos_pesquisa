import sys
sys.path.append("")

import warnings
warnings.filterwarnings("ignore")

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *
from Auxiliares.NN_models import *

from Algoritmos.LP_PUL import LP_PUL
from Algoritmos.AE_PUL import autoencoder_PUL_model
from Algoritmos.CCRNE import CCRNE
from Algoritmos.MCLS import MCLS
from Algoritmos.PU_LP import PU_LP
from Algoritmos.RCSVM_RN import RCSVM_RN
from Algoritmos.GAE_PUL import graphautoencoder_PUL_model



################################################################################
# Citeseer dataset #
################################################################################


dataset = Planetoid(root = "Datasets", name = "Cora", transform=NormalizeFeatures())
data = dataset[0]

# transformando o arquivo data em um grafo networkx
G = to_networkx(data, to_undirected=True)
edge_index = data.edge_index


# Criando uma variável para armazenar as features
X = data.x.double()
Y = data.y
pul_label = [3]
Y = torch.tensor([1 if label in pul_label else 0 for label in Y])


# Gerando os dados positivos e negativos
all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]
    
positive_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.20, 0.25]
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

        G = connect_nearest_nodes(G, positives, 5)
        edge_index1 = torch.tensor(list(G.edges)).t().contiguous()
        edge_weight = strong_connect_positives(positives, edge_index1, 3)

        pul_mask = torch.tensor([1 if i in positives else 0 for i in range(len(G.nodes()))], dtype = torch.bool)

        model_AE = AutoEncoder(input_size=X.shape[1], hidden_size1=256, hidden_size2=128)
        model_GAE1 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)
        model_GAE2 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)
        model_GAE3 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)

        optimizer_AE = optim.Adam(model_AE.parameters(), lr=0.001)
        optimizer_GAE1 = optim.Adam(model_GAE1.parameters(), lr=0.001)
        optimizer_GAE2 = optim.Adam(model_GAE2.parameters(), lr=0.001)
        optimizer_GAE3 = optim.Adam(model_GAE3.parameters(), lr=0.001)
        

        algorithms = {
        'LP_PUL' : LP_PUL(graph = G, data = X, positives = positives, unlabeled = unlabeled),
        'CCRNE' : CCRNE(data = X, positives = positives, unlabeled = unlabeled),
        'AE_PUL' : autoencoder_PUL_model(model = model_AE, optimizer = optimizer_AE, epochs = 5, data = X, positives = positives, unlabeled = unlabeled),
        'GAE_PUL1' : graphautoencoder_PUL_model(model = model_GAE1, optimizer = optimizer_GAE1, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index, edge_weight = None),
        'GAE_PUL2' : graphautoencoder_PUL_model(model = model_GAE2, optimizer = optimizer_GAE2, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index1, edge_weight = None),
        'GAE_PUL3' : graphautoencoder_PUL_model(model = model_GAE3, optimizer = optimizer_GAE3, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index1, edge_weight = edge_weight),
        'MCLS' : MCLS(data = X, positives = positives, k = 7, ratio = 0.1),
        'PU_LP' : PU_LP(data = X, positives = positives, unlabeled = unlabeled, alpha = 0.3, m = 3, l = 1),
        'RCSVM_RN' : RCSVM_RN(data = X, positives = positives, unlabeled = unlabeled, alpha = 0.1, beta = 0.9)
        }
        for algorithm in algorithms:
            print('dataset: Cora')
            start_time = time.time()
            print(f'algoritmo {algorithm}, porcentagem do dataset positivo {rate}')
            algorithms[algorithm].train()
            RN = algorithms[algorithm].negative_inference(num_neg)
            end_time = time.time()
            tempo.append(end_time - start_time)
            acc = compute_accuracy(Y, RN)
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

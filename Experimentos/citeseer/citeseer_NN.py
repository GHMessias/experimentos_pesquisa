import sys
sys.path.append("")

import warnings
warnings.filterwarnings("ignore")

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *

from Algoritmos.GAE_PUL import *
from Algoritmos.AE_PUL import *
from Auxiliares.NN_models import *



dataset = Planetoid(root = "Datasets", name = "CiteSeer", transform=NormalizeFeatures())
data = dataset[0]

# transformando o arquivo data em um grafo networkx
G = to_networkx(data, to_undirected=True)

# Criando uma variável para armazenar as features
X = data.x.double()
Y = data.y
pul_label = [2]
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
        positives = random.sample(all_positives, int(rate * len(all_positives)))
        unlabeled = list(set(range(len(G.nodes()))) - set(positives))
        
        model_AE = AutoEncoder(input_size=X.shape[1], hidden_size1=256, hidden_size2=128)
        optimizer_AE = optim.Adam(model_AE.parameters(), lr=0.001)
        
        model_GAE1 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)
        optimizer_GAE1 = optim.Adam(model_GAE1.parameters(), lr=0.001)
        
        model_GAE2 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)
        optimizer_GAE2 = optim.Adam(model_GAE2.parameters(), lr=0.001)
        
        model_GAE3 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)
        optimizer_GAE3 = optim.Adam(model_GAE3.parameters(), lr=0.001)
        
        model_GAE4 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=256, hidden_dim2=128)
        optimizer_GAE4 = optim.Adam(model_GAE4.parameters(), lr=0.001)
        
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        
        G1 = connect_nearest_nodes(G, positives, 5) # GAE_PUL3
        edge_index1 = torch.tensor(list(G1.edges)).t().contiguous() # GAE_PUL3
        
        edge_weight1 = strong_connect_positives(positives, edge_index1, 3) # GAE_PUL4
        edge_weight2 = dijkstra_n_weight(G, edge_index, positives, 5, 3) # GAE_PUL2        
        
        algorithms = {
                'AE_PUL' : autoencoder_PUL_model(model = model_AE, optimizer = optimizer_AE, epochs = 5, data = X, positives = positives, unlabeled = unlabeled),
                'GAE_PUL1' : graphautoencoder_PUL_model(model = model_GAE1, optimizer = optimizer_GAE1, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index, edge_weight = None),
                'GAE_PUL2' : graphautoencoder_PUL_model(model = model_GAE2, optimizer = optimizer_GAE2, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index, edge_weight = edge_weight2),
                'GAE_PUL3' : graphautoencoder_PUL_model(model = model_GAE3, optimizer = optimizer_GAE3, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index1, edge_weight = None),
                'GAE_PUL4' : graphautoencoder_PUL_model(model = model_GAE4, optimizer = optimizer_GAE4, epochs = 5, data = X, positives = positives, unlabeled = unlabeled, edge_index = edge_index1, edge_weight = edge_weight1)
        }
        
        for algorithm in algorithms:
            print('dataset: CiteSeer')
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

df.to_csv('Resultados/dataframe_citeseer_NN.csv', index = False)
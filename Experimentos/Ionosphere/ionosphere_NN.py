import sys
sys.path.append("")

import warnings
warnings.filterwarnings("ignore")

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *

from Algoritmos.GAE_PUL import *
from Algoritmos.AE_PUL import *
from Auxiliares.NN_models import *



df = pd.read_csv('Datasets/Ionosphere/ionosphere.data')
X = torch.tensor(df.iloc[:, :-1].values, dtype = torch.float64)
Y = [1 if x == 'b' else 0 for x in df.iloc[:, -1].values]

pul_labels = [1]
Y = torch.tensor([1 if x in pul_labels else 0 for x in Y])

# Criando um grafo a partir da matriz de proximidade Kneighbors
G = graph_from_adjacency_matrix(kneighbors_graph(X, 3, mode='distance', include_self=False).todense())
edge_index = torch.tensor(list(G.edges)).t().contiguous()
# Gerando os dados positivos e negativos
all_positives = [index for index in range(len(Y)) if Y[index] == 1]
all_negatives = [index for index in range(len(Y)) if Y[index] == 0]

    
positive_rate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.20, 0.25]
num_neg = 50

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
        
        model_AE = AutoEncoder(input_size=X.shape[1], hidden_size1=8, hidden_size2=2)
        optimizer_AE = optim.Adam(model_AE.parameters(), lr=0.001)
        
        model_GAE1 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=8, hidden_dim2=2)
        optimizer_GAE1 = optim.Adam(model_GAE1.parameters(), lr=0.001)
        
        model_GAE2 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=8, hidden_dim2=2)
        optimizer_GAE2 = optim.Adam(model_GAE2.parameters(), lr=0.001)
        
        model_GAE3 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=8, hidden_dim2=2)
        optimizer_GAE3 = optim.Adam(model_GAE3.parameters(), lr=0.001)
        
        model_GAE4 = GraphAutoEncoder(input_dim=X.shape[1], hidden_dim1=8, hidden_dim2=2)
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

df.to_csv('Resultados/dataframe_ionosphere_NN.csv', index = False)
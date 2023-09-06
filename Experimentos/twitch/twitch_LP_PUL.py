import sys
sys.path.append("")

import warnings
warnings.filterwarnings("ignore")

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *

from Algoritmos.LP_PUL import LP_PUL

from torch_geometric.datasets import Twitch

dataset = Twitch(root = "Datasets", name = "PT")
data = dataset[0]

# transformando o arquivo data em um grafo networkx
G = to_networkx(data, to_undirected=True)

# Criando uma variável para armazenar as features
X = data.x.double()
Y = data.y
pul_label = [1]
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
        
        algorithm = LP_PUL(graph = G, data = X, positives = positives, unlabeled = unlabeled)
        
        print('dataset: twitch')
        start_time = time.time()
        print(f'algoritmo {algorithm}, porcentagem do dataset positivo {rate}')
        algorithm.train()
        RN = algorithm.negative_inference(num_neg)
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

df.to_csv('Resultados/dataframe_twitch_LP_PUL.csv', index = False)
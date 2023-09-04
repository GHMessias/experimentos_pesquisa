import sys
sys.path.append("")

import warnings
warnings.filterwarnings("ignore")

from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *

from Algoritmos.MCLS import MCLS

# Define a transform to preprocess the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download the MNIST dataset
full_dataset = torchvision.datasets.MNIST(root='Datasets', train=True, transform=transform, download=True)

# Select 300 examples (300 per class)
num_examples_per_class = 300
selected_indices = []
for class_label in range(10):  # There are 10 classes (0 to 9)
    class_indices = np.where(np.array(full_dataset.targets) == class_label)[0]
    selected_indices.extend(class_indices[:num_examples_per_class])

selected_dataset = torch.utils.data.Subset(full_dataset, selected_indices)

# Create a data loader for the selected dataset
batch_size = 1
data_loader = torch.utils.data.DataLoader(dataset=selected_dataset, batch_size=batch_size, shuffle=True)

# Extract features (X) and labels from the data loader
X = []
labels = []
for images, batch_labels in data_loader:
    X.append(images.view(images.size(0), -1))  # Flatten the images
    labels.append(batch_labels)

X = torch.cat(X, dim=0).double()
Y = torch.cat(labels, dim=0)

pul_labels = [0,2,4,6,8]
Y = torch.tensor([1 if x in pul_labels else 0 for x in Y])

# Criando um grafo a partir da matriz de proximidade Kneighbors
# Criando um grafo a partir da matriz de proximidade Kneighbors
G = graph_from_adjacency_matrix(kneighbors_graph(X, 3, mode='distance', include_self=False).todense())
edge_index = torch.tensor(list(G.edges)).t().contiguous()
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
        
        algorithm = MCLS(data = X, positives = positives, k = 7, ratio = 0.1)
        
        print('dataset: MNIST')
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

df.to_csv('Resultados/dataframe_MNIST_MCLS.csv', index = False)
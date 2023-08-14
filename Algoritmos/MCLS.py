from Auxiliares.requirements import *

def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum(tensor1 - tensor2) ** 2)

def cluster_signal_ratio(cluster, positives, ratio = 0.5):
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > ratio * len(cluster):
        return 1
    else:
        return 0
    
def cluster_signal_abs(cluster, positives, k):
    pos = 0
    for i in cluster:
        if i in positives:
            pos += 1
    if pos > len(positives) / k:
        return 1
    else:
        return 0

class MCLS:
    def __init__(self, data, positives, k, ratio):
        self.positives = positives
        self.k = k
        self.ratio = ratio
        self.data = data
        self.distance = dict()


        
    def train(self):
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(self.data)
        clusters_labels = kmeans.labels_

        clusters = {}

        for indice, rotulo in enumerate(clusters_labels):
            if rotulo not in clusters:
                clusters[rotulo] = []
            clusters[rotulo].append(indice)

        cluster_signals = {}

        for cluster in clusters:
            sig = cluster_signal_ratio(clusters[cluster], self.positives,ratio = self.ratio)
            cluster_signals[cluster] = sig
        if np.sum(list(cluster_signals.values())) == 0:
            cluster_signals = {}
            for cluster in clusters:
                sig = cluster_signal_abs(clusters[cluster], self.positives, self.k)
                cluster_signals[cluster] = sig

        print(cluster_signals)
        cluster_centroids = {}

        centroids = kmeans.cluster_centers_

        for i, center in enumerate(centroids):
            cluster_centroids[i] = center
        
        # lista com os clusters positivos
        positive_clusters = [cluster for cluster in clusters if cluster_signals[cluster] == 1]
        negative_clusters = [cluster for cluster in clusters if cluster_signals[cluster] == 0]

        positive_centroids = torch.tensor([cluster_centroids[i] for i in positive_clusters])

        for cluster in negative_clusters:
            for element in clusters[cluster]:
                distances = [euclidean_distance(self.data[element], centroid) for centroid in positive_centroids]
                mean_distance = torch.mean(torch.stack(distances))
                self.distance[element] = mean_distance

    def negative_inference(self, num_neg):
        RN = sorted(self.distance, key=self.distance.get, reverse=True)[:num_neg]
        return RN
    
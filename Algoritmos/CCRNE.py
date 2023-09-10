from Auxiliares.requirements import *

def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum(tensor1 - tensor2) ** 2)

def phy(m):
    return torch.log2(m) + 1
    # return torch.log10(m) + 1

def return_index(centroids, O_j):
    for index,value in enumerate(centroids):
        if torch.equal(value, O_j):
            return index
        
class CCRNE:
    def __init__(self, data, positives, unlabeled):
        self.clusters = dict()
        self.data = data
        self.r_p = 0
        self.positives = positives
        self.unlabeled = unlabeled
        self.ratio = 0.6

    def train(self):
        self.pul_mask = torch.zeros(len(self.data))
        for i in self.positives:
            self.pul_mask[i] = 1
        self.pul_mask = self.pul_mask.bool()

        O_p = self.data[self.pul_mask].mean(dim = 0)
        r = torch.max(torch.tensor([euclidean_distance(x_k, O_p) for x_k in self.data[self.pul_mask]]))

        m = torch.tensor(len(self.positives))
        self.r_p = (r * phy(m))/(phy(m) + 1)


        self.clusters[0] = {'cluster' : [self.positives[0]],
                    'centroid' : self.data[self.positives[0]]}

        Z = self.positives[1:]
        n = 1

        for x_i in Z:
            lista_distancia = torch.tensor([euclidean_distance(self.data[x_i], O_k) for O_k in [self.clusters[i]['centroid'] for i in range(len(self.clusters))]])
            centroids = torch.tensor([self.clusters[i]['centroid'].tolist() for i in range(len(self.clusters))])
            order_idnex = torch.argsort(lista_distancia)
            centroids_ordenado = centroids[order_idnex]

            O_j = centroids_ordenado[0]
            j = return_index(centroids, O_j)

            if euclidean_distance(self.data[x_i], O_j) < self.r_p:
                self.clusters[j]['cluster'].append(x_i)
                O_j = (n * O_j + self.data[x_i]) / (n + 1)
                n += 1
                self.clusters[j]['centroid'] = O_j
            
            else:
                self.clusters[(len(self.clusters))] = dict()
                self.clusters[len(self.clusters) - 1]['cluster'] = [x_i]
                self.clusters[len(self.clusters) - 1]['centroid'] = self.data[x_i]

    
    def negative_inference(self, num_neg):
        RN = self.unlabeled

        for i in range(len(self.clusters)):
            for x_i in RN:
                    if euclidean_distance(self.data[x_i], self.clusters[i]['centroid']) < self.ratio * self.r_p:
                        RN.remove(x_i)
        
        return RN[:num_neg]

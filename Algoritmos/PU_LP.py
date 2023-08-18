from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *


class PU_LP:
    def __init__(self, data, positives, unlabeled, alpha, m, l):
        self.positives = positives
        self.data = data
        self.unlabeled = unlabeled
        self.alpha = alpha
        self.m = m
        self.l = l

    def train(self):
        pul_mask = torch.tensor([1 if x in self.positives else 0 for x in range(len(self.data))])
        A = kneighbors_graph(self.data, 3, mode='connectivity', metric='minkowski', include_self=False).todense()
        eigenvalues, eigenvectors = np.linalg.eig(A)
        largest_eigenvalue = np.max(eigenvalues)
        if self.alpha < largest_eigenvalue:
            W = torch.inverse(torch.eye(len(A)) - self.alpha * A) - torch.eye(len(A))
        else:
            print(self.alpha, largest_eigenvalue)
            raise Exception('alpha > largest_eigenvalue')
        
        self.RP = list()
        positives_ = self.positives.copy()
        unlabeled_ = self.unlabeled.copy()
        S = list()

        for k in range(self.m):
            for v_i in unlabeled_:
                #Sv_i = torch.sum(W[v_i][pul_mask]) / len(positives_)
                Sv_i = torch.mean(W[v_i][pul_mask])
                S.append(Sv_i.item())
            _, RP_ = zip(*sorted(zip(S, unlabeled_), reverse=True))
            RP_ = list(RP_)[:int(self.l / self.m)]
            positives_ = positives_ + RP_
            unlabeled_ = [element for element in unlabeled_ if element not in RP_]
            self.RP = self.RP + RP_
        
        P_RP_mask = [1 if x in self.positives + self.RP else 0 for x in range(len(self.data))]
        S = list()
        for v_i in [element for element in self.unlabeled if element not in self.RP]:
            #Sv_i = torch.sum(W[v_i][P_RP_mask]) / len(positives_)
            Sv_i = torch.mean(W[v_i][P_RP_mask])
            S.append(Sv_i.item())
        _, self.RN = zip(*sorted(zip(S, [element for element in self.unlabeled if element not in self.RP]), reverse=True))
        self.RN = list(self.RN)

    def negative_inference(self, num_neg):
        return self.RN[-num_neg:]


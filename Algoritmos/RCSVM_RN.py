from Auxiliares.requirements import *
from Auxiliares.auxiliar_functions import *

class RCSVM_RN:
    def __init__(self, data, positives, unlabeled, alpha, beta):
        self.data = data
        self.positives = positives
        self.unlabeled = unlabeled
        self.alpha = alpha
        self.beta = beta


    def train(self):
        positive_mask = torch.zeros(len(self.data))
        for i in self.positives:
            positive_mask[i] = 1
        positive_mask = torch.tensor(positive_mask, dtype = torch.bool)
        unlabeled_mask = torch.tensor([not x for x in positive_mask], dtype = torch.bool)

        self.c_positive = self.alpha * 1/len(self.positives) * torch.mean(self.data[positive_mask]) - self.beta * 1/len(self.unlabeled) * torch.mean(self.data[unlabeled_mask])

        self.c_negative = self.alpha * 1/len(self.unlabeled) * torch.mean(self.data[unlabeled_mask]) - self.beta * 1/len(self.positives) * torch.mean(self.data[positive_mask])


    def negative_inference(self, similarity = euclidean_distance):
        RN = list()
        for element in self.unlabeled:
            if similarity(self.c_positive, element) <= similarity(self.c_negative, element):
                RN.append(element)
        
        return torch.tensor(RN)
    
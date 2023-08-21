from Auxiliares.requirements import *

class graphautoencoder_PUL_model:
    def __init__(self, model, optimizer, epochs, data, positives, unlabeled, edge_index, edge_weight = None):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.data = data
        self.positives = positives
        self.unlabeled = unlabeled
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        if edge_weight is not None:
            self.edge_weight = edge_weight
        


    def train(self):
        self.pul_mask = torch.zeros((len(self.data)))
        for i in self.positives:
            self.pul_mask[i] = 1
        self.pul_mask = self.pul_mask.bool()

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            self.model.double()
            F.mse_loss(self.data[self.pul_mask], self.model(self.data, self.edge_index, self.edge_weight)[self.pul_mask]).backward()
            #print(F.mse_loss(self.data[self.pul_mask], self.model(self.data)[self.pul_mask]))
            self.optimizer.step()

    def negative_inference(self, num_neg):
        output_ = self.model(self.data, self.edge_index, self.edge_weight)
        loss_rank = [F.mse_loss(self.data[i], output_[i]).item() for i in self.unlabeled]

        RN = [x for _, x in sorted(zip(loss_rank, self.unlabeled), reverse = True)][:num_neg]
        # for loss, element in sorted(zip(loss_rank, self.unlabeled), reverse = False):
        #     print(loss, element, y[element])
        # print(RN)
        return RN
        
        
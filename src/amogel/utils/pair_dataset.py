import torch 

class PairDataset(torch.utils.data.Dataset):
    def __init__(self , datasetA , datasetB , datasetC):
        super().__init__()
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.datasetC = datasetC

        assert len(self.datasetA) == len(self.datasetB)
        assert len(self.datasetC) == len(self.datasetA)

    def __len__(self):
        return len(self.datasetA)
        
    def __getitem__(self , idx):
        return self.datasetA[idx] , self.datasetB[idx] , self.datasetC[idx]
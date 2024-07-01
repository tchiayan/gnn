from typing import Any
import pytorch_lightning as pl 
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
import torch.nn.functional as F
from torch.utils.data import DataLoader , Dataset
import pandas as pd
import torch
from sklearn.metrics import classification_report , roc_auc_score
import numpy as np

class DNN(pl.LightningModule):
    
    def __init__(self , input_dimension,  num_classes) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dimension , 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024 , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512 , 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32 , num_classes)
        )
        
        self.loss = nn.CrossEntropyLoss()
        self.acc = MulticlassAccuracy(num_classes=num_classes)
        self.actual = []
        self.predict = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x , y = batch
        y_hat = self.model(x)
        
        # calculate loss
        loss = self.loss(y_hat , y)
        acc = self.acc(F.softmax(y_hat , dim=1) , y)

        self.log("train_loss" , loss , on_step=False , on_epoch=True , prog_bar=True )
        self.log("train_acc" , acc , on_step=False , on_epoch=True , prog_bar=True )
        
        return loss
    
    def validation_step(self, batch , batch_idx):
        x , y = batch
        y_hat = self.forward(x)
        
        # calculate loss
        loss = self.loss(y_hat , y)
        acc = self.acc(F.softmax(y_hat , dim=1) , y)
        
        self.actual.extend(y.cpu().numpy())
        self.predict.extend(F.softmax(y_hat , dim=1).cpu().numpy())
        
        self.log("val_loss" , loss  , on_epoch=True , prog_bar=True)
        self.log("val_acc" , acc  , on_epoch=True , prog_bar=True)
        
    def on_validation_epoch_end(self) -> None:
        
        if self.current_epoch == self.trainer.max_epochs - 1:
            output = classification_report(self.actual , np.stack(self.predict, axis=0).argmax(axis=1) , digits=4)
            if self.num_classes == 2:
                output += f"roc_auc: {roc_auc_score(self.actual , np.stack(self.predict , axis=0)[:,1] , multi_class='ovr')}\n"
            else:
                output += f"roc_auc: {roc_auc_score(self.actual , np.stack(self.predict , axis=0) , multi_class='ovr'):.4f}\n"
            
            with open("./artifacts/compare/traditional/dnn_report.txt" , "w") as f:
                f.write(output)
                
            print(output)
            
        self.actual = []
        self.predict = []
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters() , lr=1e-5 , weight_decay=0.01)
        return optimizer
        
    
class OmicDataset(Dataset):
    
    def __init__(self , data: pd.DataFrame , label: pd.DataFrame) -> None:
        
        assert len(data) == len(label) , "Data and label must have the same length"
        assert isinstance(data , pd.DataFrame) , "Data must be a pandas DataFrame"
        assert isinstance(label , pd.DataFrame) , "Label must be a pandas DataFrame"
        
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = torch.tensor(self.data.iloc[idx].values , dtype=torch.float32)
        label = torch.tensor(self.label.iloc[idx,0] , dtype=torch.long)
        
        return data , label
    

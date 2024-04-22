# build simple GCN model for graph classification 
from torch_geometric.loader import DataLoader   
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy , Precision , Recall , AUROC , ConfusionMatrix 
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , BatchNorm 
from torch_geometric.nn import global_mean_pool

class GCN(pl.LightningModule):
    def __init__(self, num_features ,  hidden_channels , output_class , lr=0.0001):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=output_class)
        self.precision = Precision(task="multiclass" , num_classes=output_class)
        self.recall = Recall(task="multiclass" , num_classes=output_class)
        self.auroc = AUROC(task="multiclass" ,num_classes=output_class)
        self.cfm_training = ConfusionMatrix(task="multiclass", num_classes=output_class)
        self.cfm_testing = ConfusionMatrix(task="multiclass", num_classes=output_class)
        self.lr = lr

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x , edge_index , batch , y = batch.x , batch.edge_index , batch.batch , batch.y
        out = self(x, edge_index, batch)
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        self.cfm_training(out , y)
        
        self.log('train_loss' , loss , prog_bar=True)
        self.log('train_acc' , acc , prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x , edge_index , batch , y = batch.x , batch.edge_index , batch.batch , batch.y
        out = self(x, edge_index, batch)
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        preci = self.precision(out, y)
        rec = self.recall(out, y)
        auroc = self.auroc(out, y)
        cfm = self.cfm_testing(out, y)  
        
        self.log('val_loss' , loss , prog_bar=True, on_epoch=True)
        self.log('val_acc' , acc , prog_bar=True, on_epoch=True)
        self.log('val_preci' , preci , prog_bar=True, on_epoch=True)
        self.log('val_rec' , rec , prog_bar=True, on_epoch=True)
        self.log('val_auroc' , auroc , prog_bar=True , on_epoch=True)
    
    def on_train_epoch_end(self) -> None:
        
        cfm = self.cfm_training.compute().cpu().numpy()
        print("")
        print("-------- Confusion Matrix [Training] --------")
        print(cfm)
        
        self.cfm_training.reset()
        
    def on_validation_epoch_end(self):
        
        cfm = self.cfm_testing.compute().cpu().numpy()
        print("")
        print("-------- Confusion Matrix [Testing] --------")
        print(cfm)
        
        self.cfm_testing.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
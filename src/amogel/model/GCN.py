# build simple GCN model for graph classification 
from torch_geometric.loader import DataLoader   
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy , Precision , Recall , AUROC , ConfusionMatrix , F1Score , Specificity
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , BatchNorm , GATConv 
from torch_geometric.nn import global_mean_pool , SAGPooling , TopKPooling
import mlflow
from sklearn.metrics import classification_report , roc_auc_score

class GCN(pl.LightningModule):
    def __init__(self, in_channels ,  hidden_channels , num_classes , lr=0.0001 , drop_out=0.0, weight=None, pooling_ratio=0 ,mlflow:mlflow = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(GCN, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin_hidden = Linear(hidden_channels, hidden_channels)
        self.pooling = SAGPooling(hidden_channels, ratio=self.pooling_ratio)
        self.lin = Linear(hidden_channels, num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout1d(drop_out),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        self.weight = weight if weight is None else torch.tensor(weight, device=device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.mlflow = mlflow
        
        # [[ 79   0  14  10   0] =  79+0+14+10+0 = 103 , 614 / (103) = 5.96 , 1 - 103 / 614 = 0.83
        # [ 14   0  18   9   0] = 14+0+18+9+0 = 41 , 614 /  41 = 14.97 , 1 - 41 / 614 = 0.93
        # [  1   0 327   7   0] = 1+0+327+7+0 = 335 , 614 / 335 = 1.83 , 1 - 335 / 614 = 0.45
        # [  5   0  83  19   0] = 5+0+83+19+0 = 107 , 614 / 107 = 5.73 , 1 - 107 / 614 = 0.82
        # [  0   0  27   1   0]] = 0+0+27+1+0 = 28 , 614 / 28 = 21.93 , 1 - 28 / 614 = 0.95

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass" , num_classes=num_classes)
        self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        self.recall = Recall(task="multiclass" , num_classes=num_classes , average="macro")
        self.auroc = AUROC(task="multiclass" ,num_classes=num_classes)
        self.f1score = F1Score(task="multiclass" , num_classes=num_classes , average="macro")
        self.cfm_training = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.cfm_testing = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.actual = []
        self.predicted = []
        self.predicted_proba = []
        self.lr = lr
        self.drop_out = drop_out

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index , edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index , edge_attr)
        x = self.bn2(x)
        x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        if self.pooling_ratio > 0:
            x , edge_index , edge_attr , batch , perm , score = self.pooling(x , edge_index , edge_attr , batch)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=self.drop_out , training=self.training)
        # x = self.lin_hidden(x).relu()
        # x = self.lin(x)
        x = self.mlp(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x , edge_index , edge_attr, batch , y = batch.x , batch.edge_index , batch.edge_attr , batch.batch , batch.y
        out = self(x, edge_index, edge_attr, batch)
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        self.cfm_training(out , y)
        
        self.log('train_loss' , loss , prog_bar=True, on_epoch=True , on_step=False)
        self.log('train_acc' , acc , prog_bar=True , on_epoch=True , on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x , edge_index , edge_attr, batch , y = batch.x , batch.edge_index , batch.edge_attr , batch.batch , batch.y
        out = self(x, edge_index, edge_attr, batch)
        
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        preci = self.precision(out, y)
        rec = self.recall(out, y)
        auroc = self.auroc(out, y)
        spe = self.specificity(out, y)
        f1score = self.f1score(out, y)
        cfm = self.cfm_testing(out, y)  
        self.actual.extend(y.cpu().numpy())
        self.predicted.extend(out.argmax(dim=1).cpu().numpy())
        self.predicted_proba.extend(F.softmax(out , dim=-1).cpu().numpy())  
        
        self.log('val_loss' , loss , prog_bar=True, on_epoch=True)
        self.log('val_acc' , acc , prog_bar=True, on_epoch=True)
        self.log('val_preci' , preci , prog_bar=True, on_epoch=True)
        self.log('val_rec' , rec , prog_bar=True, on_epoch=True)
        self.log('val_auroc' , auroc , prog_bar=True , on_epoch=True)
        self.log('val_f1score' , f1score , on_epoch=True)
        self.log('val_spe' , spe , on_epoch=True)
    
    def on_train_epoch_end(self) -> None:
        
        if self.current_epoch == self.trainer.max_epochs - 1:
            cfm = self.cfm_training.compute().cpu().numpy()
            print("")
            print("-------- Confusion Matrix [Training] --------")
            print(cfm)
        self.cfm_training.reset()
        
    def on_validation_epoch_end(self):
        
        if self.current_epoch == self.trainer.max_epochs - 1:
            report = classification_report(self.actual , self.predicted , digits=4)
            auc = roc_auc_score(self.actual , self.predicted_proba , multi_class="ovr")
            report += f"roc_auc_score: {auc:.4f}"
            if self.mlflow is not None:
                self.mlflow.log_text(report , f"calssification_report_val_{self.current_epoch:04d}.txt")
            print(report)
        self.actual = []
        self.predicted = []
        self.predicted_proba = []
        self.cfm_testing.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
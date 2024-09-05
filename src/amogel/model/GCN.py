# build simple GCN model for graph classification 
from torch_geometric.loader import DataLoader   
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy , Precision , Recall , AUROC , ConfusionMatrix , F1Score , Specificity
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv , BatchNorm , GATConv 
from torch_geometric.nn import global_mean_pool , SAGPooling , TopKPooling
from torch_geometric.utils import to_dense_batch
import mlflow
from sklearn.metrics import classification_report , roc_auc_score
import numpy as np

class GCN(pl.LightningModule):
    def __init__(self, in_channels ,  hidden_channels , num_classes , no_of_nodes ,  dnn_hidden_nodes=1024, lr=0.0001 , drop_out=0.0, weight=None, pooling_ratio=0 ,mlflow:mlflow = None , decay=0.0 , GNN=True , DNN=True, save_dir="./artifacts/amogel"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(GCN, self).__init__()
        self.save_dir = save_dir
        self.DNN = DNN
        self.GNN = GNN
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.decay = decay
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin_hidden = Linear(hidden_channels, hidden_channels)
        self.pooling = SAGPooling(hidden_channels, ratio=self.pooling_ratio)
        self.lin = Linear(hidden_channels, num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 if GNN and DNN else hidden_channels, hidden_channels), # combined feed x and graph x 
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
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(no_of_nodes , dnn_hidden_nodes),
            torch.nn.BatchNorm1d(dnn_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(dnn_hidden_nodes , int(dnn_hidden_nodes/2)),
            torch.nn.BatchNorm1d(int(dnn_hidden_nodes/2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_out),
            torch.nn.Linear(int(dnn_hidden_nodes/2) , hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Dropout(drop_out),
        )
        
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
        self.edge_attn_l1 = []
        self.edge_attn_l2 = []
        self.batches = []
        self.pool_batches = []
        self.pool_perm = []
        self.pool_score = []
        self.embeddings = []

    def forward(self, x, edge_index, edge_attr, batch):
        
        # convert node embeddign to dense batch 
        batch_x , batch_mask = to_dense_batch(x , batch) # dimension ( no_batch , number_of_node )
        batch_x = batch_x.view(batch_x.shape[0] , batch_x.shape[1] * batch_x.shape[2]) 
        
        feed_x  = self.feedforward(batch_x) # [ batch_size  hidden_channels ]
        
        # 1. Obtain node embeddings 
        x , edge_attn_l1 = self.conv1(x, edge_index , edge_attr , return_attention_weights=True)
        x = self.bn1(x)
        x = x.relu()
        x , edge_attn_l2 = self.conv2(x, edge_index , edge_attr , return_attention_weights=True)
        x = self.bn2(x)
        x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        if self.pooling_ratio > 0:
            x , edge_index , edge_attr , batch , perm , score = self.pooling(x , edge_index , edge_attr , batch)
        else:
            perm = None 
            score = None
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        concat_layer = []
        if self.GNN:
            concat_layer.append(x)
        if self.DNN: 
            concat_layer.append(feed_x)
        global_x = torch.concat(concat_layer , dim=1)
        
        # 3. Apply a final classifier
        # x = F.dropout(x, p=self.drop_out , training=self.training)
        # x = self.lin_hidden(x).relu()
        # x = self.lin(x)
        x = self.mlp(global_x)
        
        return x , edge_attn_l1 , edge_attn_l2 , batch , perm , score

    def training_step(self, batch, batch_idx):
        x , edge_index , edge_attr, batch , y = batch.x , batch.edge_index , batch.edge_attr , batch.batch , batch.y
        out , edge_attn_l1 , edge_attn_l2 , _ , _ , _ = self(x, edge_index, edge_attr, batch)
        loss = self.criterion(out, y)
        acc = self.accuracy(out, y)
        self.cfm_training(out , y)
        
        self.log('train_loss' , loss , prog_bar=True, on_epoch=True , on_step=False)
        self.log('train_acc' , acc , prog_bar=True , on_epoch=True , on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x , edge_index , edge_attr, batch , y = batch.x , batch.edge_index , batch.edge_attr , batch.batch , batch.y
        self.batches.append(batch)
        out , edge_attn_l1, edge_attn_l2 , pool_batch , pool_perm , pool_score = self(x, edge_index, edge_attr, batch)
        
        self.edge_attn_l1.append(edge_attn_l1)
        self.edge_attn_l2.append(edge_attn_l2)
        self.pool_batches.append(pool_batch)
        self.pool_perm.append(pool_perm)
        self.pool_score.append(pool_score)
        
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
        self.edge_attn_l2 = []
        self.edge_attn_l1 = []
        self.batches = []
        self.cfm_training.reset()
        
    def on_validation_epoch_end(self):
        
        if (self.current_epoch+1) % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            report = classification_report(self.actual , self.predicted , digits=4)
            if self.num_classes == 2:
                auc = roc_auc_score(self.actual ,  np.stack(self.predicted_proba , axis=0)[: , 1] , multi_class="ovr")
                report += f"roc_auc_score: {auc:.4f}"
            else:
                auc = roc_auc_score(self.actual , self.predicted_proba , multi_class="ovr")
                report += f"roc_auc_score: {auc:.4f}"
            if mlflow is not None:
                mlflow.log_text(report , f"calssification_report_val_{(self.current_epoch+1):04d}.txt")
                
            if self.current_epoch == self.trainer.max_epochs - 1:
                print(report)
            
                torch.save(self.edge_attn_l1 , f"{self.save_dir}/edge_attn_l1_{self.current_epoch+1}.pt")
                torch.save(self.edge_attn_l2 , f"{self.save_dir}/edge_attn_l2_{self.current_epoch+1}.pt")
                torch.save(self.batches , f"{self.save_dir}/batches_{self.current_epoch+1}.pt")
                torch.save(self.pool_batches , f"{self.save_dir}/pool_batches_{self.current_epoch+1}.pt")
                torch.save(self.pool_perm , f"{self.save_dir}/pool_perm_{self.current_epoch+1}.pt")
                torch.save(self.pool_score , f"{self.save_dir}/pool_score_{self.current_epoch+1}.pt")
            
        self.edge_attn_l2 = []
        self.edge_attn_l1 = []
        self.batches = []
        self.pool_batches = []
        self.pool_perm = []
        self.pool_score = []
        self.actual = []
        self.predicted = []
        self.predicted_proba = []
        self.cfm_testing.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr , weight_decay=self.decay)
        
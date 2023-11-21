from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch_geometric.nn as geom_nn
import torch_geometric.utils as geom_utils
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Batch
import torch 
import lightning as pl 
from torch import optim
from torchmetrics import Accuracy , AUROC , F1Score
import os 
from utils import  generate_graph , read_features_file
import mlflow 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback 
from sklearn.model_selection import StratifiedKFold
from math import ceil
import argparse

gnn = {
    'GCNConv': geom_nn.DenseGCNConv , 
    'GraphConv': geom_nn.DenseGraphConv , 
    'GATConv': geom_nn.DenseGATConv , 
    'SAGEConv': geom_nn.DenseSAGEConv , 
}
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
    

class GCN(torch.nn.Module):
    
    def __init__(self , in_channels , hidden_channels , out_channels):
        super().__init__()
        self.conv1 = geom_nn.SAGEConv(in_channels , hidden_channels)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = geom_nn.SAGEConv(hidden_channels , hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = geom_nn.SAGEConv(hidden_channels , out_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(out_channels)
        
    def forward(self , x , edge_index , edge_attr , mask=None):
        
        x = self.conv1( x , edge_index ).relu()
        x = self.batch_norm1(x)
        
        x = self.conv2( x , edge_index ).relu()
        x = self.batch_norm2(x)
        
        x = self.conv3( x , edge_index ).relu()
        x = self.batch_norm3(x)
        
        return x
    
class DenseGCN(torch.nn.Module):
    
    def __init__(self , in_channels , hidden_channels , out_channels , skip_connection = True , lin=True , type='SAGEConv'):
        super().__init__()
        self.conv1 = gnn[type](in_channels , hidden_channels)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = gnn[type](hidden_channels , hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = gnn[type](hidden_channels , out_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(out_channels)
        self.skip_connection = skip_connection
        if lin:
            self.lin = torch.nn.Linear(
                2 * hidden_channels + out_channels , out_channels
            )
        else: 
            self.lin = None
    
    def bn(self , i , x):
        batch_size , nodes , num_channels = x.size()
        
        x = x.view(-1 , num_channels)
        x = getattr(self , 'batch_norm{}'.format(i))(x)
        x = x.view(batch_size , nodes , num_channels)
        return x 
    
    def forward(self , x , edge_index , edge_attr , mask=None):
        #print("Shape of x: " , x.size())
        #print("Shape of edge: " , edge_index.size())
        x0 = x
        x1 = self.bn(1 , self.conv1(x0 , edge_index , mask).relu())
        x2 = self.bn(2 , self.conv2(x1 , edge_index , mask).relu())
        x3 = self.bn(3 , self.conv3(x2 , edge_index , mask).relu())
        
        if not self.skip_connection:
            return x3
        else: 
            x = torch.concat([x1 , x2 , x3] , dim = -1) # each output node have node_embedding dimension * 3
            if self.lin is not None:
                x = self.lin(x).relu()
            return x
        
class GeneralPooling(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr = 1e-3):
        super().__init__()
        
        self.graph_model = GCN( in_channels , hidden_channels , hidden_channels)
        self.lr = lr
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels , hidden_channels),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels , hidden_channels),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels , num_classes), 
            torch.nn.Softmax()
        )
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass" , num_classes=num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        
    def forward(self , x , edge_index , edge_attr , batch_idx):
        
        # Graph Message Passing
        x = self.graph_model(x , edge_index , edge_attr)
        # print("Graph model shape: ", x.size())
        
        # Graph Pooling 
        x = geom_nn.global_mean_pool(x , batch_idx)
        # print("Graph pooling shape: " , x.size())
        
        # Dense layer 
        x = self.head(x)
        return x 
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , lr=1e-3)
        return optimizer 
    
    def training_step(self , batch , batch_idx):
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        
        output = self.forward(batch.x , batch.edge_index , batch.edge_attr , batch.batch) 
        #print("Output dimension: ", output.size())
        
        loss = self.loss(output , batch.y)
        acc = self.acc(output , batch.y)
        
        self.log("train_acc" , acc , prog_bar=True)
        self.log("train_loss" , loss , prog_bar=True)
        
        return loss 
    
    def validation_step(self , batch , batch_idx):
        output = self.forward(batch.x , batch.edge_index , batch.edge_attr , batch.batch) 
        loss = self.loss(output , batch.y)
        acc = self.acc(output , batch.y)
        
        f1 = self.f1(output , batch.y)
        auc = self.auc(output , batch.y)
        self.log("val_acc" , acc , prog_bar=True)
        self.log("val_loss" , loss , prog_bar=True)
        self.log('val_auc' , auc , prog_bar=True)
        self.log('val_f1' , f1 , prog_bar=True)

class DmonGraphPooling(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , input_size , lr=1e-4):
        super().__init__() 
        self.lr = lr
        
        self.conv1 = geom_nn.GCNConv(in_channels , hidden_channels)
        num_nodes = ceil(0.5 * input_size)
        self.pool1 = geom_nn.DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv2 = geom_nn.DenseGraphConv(hidden_channels, hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = geom_nn.DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv3 = geom_nn.DenseGraphConv(hidden_channels, hidden_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels)

        #self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels) , 
            torch.nn.Dropout(0.1), 
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels,hidden_channels), 
            torch.nn.Dropout(0.1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels,hidden_channels), 
            torch.nn.Dropout(0.1), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels , num_classes) , 
            torch.nn.Softmax(dim=-1)
        )
        #self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
        #self.softmax = torch.nn.Softmax(dim=-1)
        
        #self.loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.NLLLoss()
        self.acc = Accuracy(task='multiclass' , num_classes = num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        
    def bn(self , i , x):
        batch_size , nodes , num_channels = x.size()
        
        x = x.view(-1 , num_channels)
        x = getattr(self , 'batch_norm{}'.format(i))(x)
        x = x.view(batch_size , nodes , num_channels)
        return x 
    
    def forward(self , x , edge_index , batch):
        
        x = self.conv1(x , edge_index).relu()
        
        x , mask = geom_utils.to_dense_batch(x , batch)
        adj = geom_utils.to_dense_adj(edge_index , batch)
        
        _ , x , adj , sp1 , o1 , c1 = self.pool1(x , adj , mask)
        
        #x = self.conv2(x , adj).relu()
        x = self.bn(2 , self.conv2(x , adj).relu())
        
        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)
        
        #x = self.conv3(x , adj)
        x = self.bn(3 , x = self.conv3(x , adj))
        
        x = x.mean(dim=1)
        #x = self.lin1(x).relu()
        # = self.lin2(x)
        x = self.head(x)
        
        return x , sp1 + sp2 + o1 + o2 + c1 + c2
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , lr=self.lr)
        return optimizer 
    
    
    def training_step(self , batch ,  batch_idx):
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 = batch
        
        output , dmonpool_loss = self.forward(batch_1.x , batch_1.edge_index , batch_1.batch ) 
        #print("Output dimension: ", output.size())
        
        loss = self.loss(output , batch_1.y) + dmonpool_loss 
        acc = self.acc(output , batch_1.y)
        f1 = self.f1(output , batch_1.y)
        auc = self.auc(output , batch_1.y)
        
        self.log("train_acc" , acc , prog_bar=True , on_epoch=True)
        self.log("train_loss" , loss , prog_bar=True , on_epoch=True)
        self.log('train_auc' , auc)
        self.log('train_f1' , f1)
        return loss 
    
    def validation_step(self , batch , batch_idx):
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 = batch
        
        output , dmonpool_loss = self.forward(batch_1.x , batch_1.edge_index , batch_1.batch ) 
        #print("Output dimension: ", output.size())
        
        # print(output.argmax(dim=-1))
        # print(batch_1.y)
        loss = self.loss(output , batch_1.y) + dmonpool_loss
        acc = self.acc(output , batch_1.y)
        f1 = self.f1(output , batch_1.y)
        auc = self.auc(output , batch_1.y)
        
        self.log("val_acc" , acc , prog_bar=True, on_epoch=True)
        self.log("val_loss" , loss , prog_bar=True, on_epoch=True)
        self.log('val_auc' , auc)
        self.log('val_f1' , f1)
        
        
class SingleGraphDiffPooling(pl.LightningModule):
    def __init__(self , in_channels , hidden_channels , num_classes , input_size , skip_connection=None , lr=1e-3 , type='SAGEConv'):
        super().__init__()
        
        self.skip_connection = skip_connection 
        self.lr = lr  
        
        # Pooling layer 1 
        node_size = min( 100 , ceil(input_size * 0.5))
        self.graph_em_1 = DenseGCN( in_channels , hidden_channels , hidden_channels , skip_connection , lin=False , type=type)
        self.graph_pl_1 = DenseGCN( in_channels , hidden_channels , node_size , skip_connection  , type=type)
        
        # Pooling layer 2
        node_size = ceil(node_size * 0.5)
        input_hidden_channels = 3 * hidden_channels if skip_connection  else hidden_channels
        self.graph_em_2 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels  , skip_connection , lin=False, type=type)
        self.graph_pl_2 = DenseGCN( input_hidden_channels , hidden_channels , node_size , skip_connection , type=type)
        
        # Pooling layer 3 
        node_size = ceil(node_size * 0.5)
        input_hidden_channels = 3 * hidden_channels if skip_connection else hidden_channels
        self.graph_em_3 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_3 = DenseGCN( input_hidden_channels , hidden_channels , 1 , skip_connection, type=type)
        
        # Last layer of graph convolution
        input_hidden_channels = 3 * hidden_channels if skip_connection  else hidden_channels
        self.graph_em_4 = DenseGCN( input_hidden_channels , hidden_channels , num_classes , lin=False, type=type)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_hidden_channels , hidden_channels*2),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*2), 
            torch.nn.Linear(hidden_channels*2 , hidden_channels*2),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*2), 
            torch.nn.Linear(hidden_channels*2 , hidden_channels),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels , num_classes), 
            torch.nn.Softmax()
        )
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass' , num_classes = num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        
    def forward(self , x1 , edge_index1 , batch1_idx ):
        
        #print("Shape of edge_index1 before dense:", edge_index1.size())
        x1 , _ = geom_utils.to_dense_batch(x1 , batch1_idx)
        edge_index1 = geom_utils.to_dense_adj(edge_index1, batch1_idx)
        #print("Shape of edge_index1 after dense:", edge_index1.size())
        #print(edge_index1)
        
        #x2 , _ = geom_utils.to_dense_batch(x2 , batch2_idx)
        #edge_index2 = geom_utils.to_dense_adj(edge_index2 , batch2_idx)
        
        #print("0 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # first layer pooling 
        s1 = self.graph_pl_1(x1 , edge_index1 , None)
        x1 = self.graph_em_1(x1 , edge_index1 , None)
        
        #s2 = self.graph_pl_1(x2 , edge_index2 , None)
        #x2 = self.graph_em_1(x2 , edge_index2 , None)
        
        #print("1 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("1 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        
        x1 , edge_index1 , l11 , _ = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        #x2 , edge_index2 , l12 , _ = geom_nn.dense_diff_pool(x2 , edge_index2 , s2)
        
        #print("1 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # second layer pooling 
        s1 = self.graph_pl_2(x1 , edge_index1 , None)
        x1 = self.graph_em_2(x1 , edge_index1 , None)
        
        #s2 = self.graph_pl_2(x2 , edge_index2 , None)
        #x2 = self.graph_em_2(x2 , edge_index2 , None)
        
        
        #print("2 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("2 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        
        x1 , edge_index1 , l21 , _ = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        #x2 , edge_index2 , l22 , _ = geom_nn.dense_diff_pool(x2 , edge_index2 , s2)
        
        # print("2 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x1.size())
        
        # final layer pooling 
        s1 = self.graph_pl_3(x1 , edge_index1 , None)
        x1 = self.graph_em_3(x1 , edge_index1 , None)
        
        #s2 = self.graph_pl_3(x2 , edge_index2 , None)
        #x2 = self.graph_em_3(x2 , edge_index2 , None)
        
        # print("3 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x1.size())
        # print("3 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s1.size())
        
        x1 , edge_index1 , l31 , _ = geom_nn.dense_diff_pool(x1 , edge_index1 , s1) # x1 => number_of_sample , 1 , embedding_size
        #x2 , edge_index2 , _ , _ = geom_nn.dense_diff_pool(x2 , edge_index2 , s2) # x2 => number_of_sample , 1 , embedding_size
        
        # print("3 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x1.size()) 
        
        # final layer GNN 
        #x = torch.cat([x1 , x2] , dim=-1).squeeze(1)
        x = x1.squeeze(1)
        # print("Final dimension: ", x.size())
        x = self.head(x)
        return x , l11+l21+l31
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , lr=self.lr)
        return optimizer 
    
    def training_step(self , batch , batch_idx):
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 = batch
        
        
        output , diffpool_loss = self.forward(batch_1.x , batch_1.edge_index , batch_1.batch ) 
        #print("Output dimension: ", output.size())
        
        # print(output.argmax(dim=-1))
        # print(batch_1.y)
        
        loss = self.loss(output , batch_1.y) + diffpool_loss
        acc = self.acc(output , batch_1.y)
        f1 = self.f1(output , batch_1.y)
        auc = self.auc(output , batch_1.y)
        
        self.log("train_acc" , acc , prog_bar=True , on_epoch=True)
        self.log("train_loss" , loss , prog_bar=True , on_epoch=True)
        self.log('train_auc' , auc)
        self.log('train_f1' , f1)
        return loss 
    
    def validation_step(self , batch , batch_idx):
        batch_1 , batch_2 = batch
        
        output , diffpool_loss = self.forward(batch_1.x , batch_1.edge_index , batch_1.batch ) 
        
        # print(output.size())
        # print(batch_1.y)
        # print(batch_1)
        loss = self.loss(output , batch_1.y) + diffpool_loss
        acc = self.acc(output , batch_1.y)
        
        f1 = self.f1(output , batch_1.y)
        auc = self.auc(output , batch_1.y)
        self.log("val_acc" , acc , prog_bar=True , on_epoch=True )
        self.log("val_loss" , loss , prog_bar=True , on_epoch=True)
        self.log('val_auc' , auc , prog_bar=True)
        self.log('val_f1' , f1 , prog_bar=True) 
        
class MultiGraphDiffPooling(pl.LightningModule):
    
        
    def __init__(self , in_channels , hidden_channels , num_classes , input_size , skip_connection=False , lr=1e-3, type='SAGEConv'):
        super().__init__()
        
        self.skip_connection = skip_connection 
        self.lr = lr
        
        # Pooling layer 1 
        node_size = min( 100 , ceil(input_size * 0.5))
        self.graph_em_11 = DenseGCN( in_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_11 = DenseGCN( in_channels , hidden_channels , node_size , skip_connection , type=type)
        
        self.graph_em_12 = DenseGCN( in_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_12 = DenseGCN( in_channels , hidden_channels , node_size , skip_connection , type=type)
        
        self.graph_em_13 = DenseGCN( in_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_13 = DenseGCN( in_channels , hidden_channels , node_size , skip_connection , type=type)
        
        # Pooling layer 2
        node_size = ceil(node_size * 0.5)
        input_hidden_channels = 3 * hidden_channels if skip_connection  else hidden_channels
        self.graph_em_21 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_21 = DenseGCN( input_hidden_channels , hidden_channels , node_size , skip_connection, type=type)
        
        self.graph_em_22 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_22 = DenseGCN( input_hidden_channels , hidden_channels , node_size , skip_connection, type=type)
        
        self.graph_em_23 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type)
        self.graph_pl_23 = DenseGCN( input_hidden_channels , hidden_channels , node_size , skip_connection, type=type)
        
        # Pooling layer 3 
        node_size = ceil(node_size * 0.5)
        input_hidden_channels = 3 * hidden_channels if skip_connection  else hidden_channels
        self.graph_em_31 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels  , lin=True, type=type)
        self.graph_pl_31 = DenseGCN( input_hidden_channels , hidden_channels , 1  , skip_connection, type=type)
        
        self.graph_em_32 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels  , lin=True, type=type)
        self.graph_pl_32 = DenseGCN( input_hidden_channels , hidden_channels , 1  , skip_connection, type=type)
        
        self.graph_em_33 = DenseGCN( input_hidden_channels , hidden_channels , hidden_channels  , lin=True, type=type)
        self.graph_pl_33 = DenseGCN( input_hidden_channels , hidden_channels , 1  , skip_connection, type=type)
        
        self.graph_em_4 = DenseGCN( hidden_channels , hidden_channels , hidden_channels  , lin=True, type=type)
        self.graph_pl_4 = DenseGCN( hidden_channels , hidden_channels , 1 , skip_connection , type='GATConv')
        
        # Last layer of graph convolution
        # input_hidden_channels = 2 * 3 * hidden_channels if skip_connection  else 2 * hidden_channels
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels , 1024),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(1024), 
            torch.nn.Linear(1024 , 1024),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(1024), 
            torch.nn.Linear(1024 , 512),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(512), 
            torch.nn.Linear(512 , num_classes), 
        )
        
        # self.sofmax = torch.nn.Softmax()
        self.loss = torch.nn.CrossEntropyLoss()
        self.x1loss = torch.nn.CrossEntropyLoss()
        self.x2loss = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass' , num_classes = num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        
    
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx , view_edge):
        
        #print("Shape of edge_index1 before dense:", edge_index1.size())
        x1 , _ = geom_utils.to_dense_batch(x1 , batch1_idx)
        edge_index1 = geom_utils.to_dense_adj(edge_index1, batch1_idx  , edge_attr1).squeeze(-1)
        #print("Shape of edge_index1 after dense:", edge_index1.size())
        #print(edge_index1)
        
        x2 , _ = geom_utils.to_dense_batch(x2 , batch2_idx)
        edge_index2 = geom_utils.to_dense_adj(edge_index2 , batch2_idx , edge_attr2).squeeze(-1)
        
        x3 , _ = geom_utils.to_dense_batch(x3 , batch3_idx)
        edge_index3 = geom_utils.to_dense_adj(edge_index3 , batch3_idx , edge_attr3).squeeze(-1)
        
        #print("0 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # first layer pooling 
        s1 = self.graph_pl_11(x1 , edge_index1 , None)
        x1 = self.graph_em_11(x1 , edge_index1 , None)
        
        s2 = self.graph_pl_12(x2 , edge_index2 , None)
        x2 = self.graph_em_12(x2 , edge_index2 , None)
        
        s3 = self.graph_pl_13(x3 , edge_index3 , None)
        x3 = self.graph_em_13(x3 , edge_index3 , None)
        
        #print("1 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("1 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        
        x1 , edge_index1 , l11 , e11 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        x2 , edge_index2 , l12 , e12 = geom_nn.dense_diff_pool(x2 , edge_index2 , s2)
        x3 , edge_index3 , l13 , e13 = geom_nn.dense_diff_pool(x3 , edge_index3 , s3)
        
        #print("1 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # second layer pooling 
        s1 = self.graph_pl_21(x1 , edge_index1 , None)
        x1 = self.graph_em_21(x1 , edge_index1 , None)
        
        s2 = self.graph_pl_22(x2 , edge_index2 , None)
        x2 = self.graph_em_22(x2 , edge_index2 , None)
        
        s3 = self.graph_pl_23(x3 , edge_index3 , None)
        x3 = self.graph_em_23(x3 , edge_index3 , None)
        
        #print("2 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("2 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        
        x1 , edge_index1 , l21 , e21 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        x2 , edge_index2 , l22 , e22 = geom_nn.dense_diff_pool(x2 , edge_index2 , s2)
        x3 , edge_index3 , l23 , e23 = geom_nn.dense_diff_pool(x3 , edge_index3 , s3)
        
        #print("2 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # final layer pooling 
        s1 = self.graph_pl_31(x1 , edge_index1 , None)
        x1 = self.graph_em_31(x1 , edge_index1 , None)
        
        s2 = self.graph_pl_32(x2 , edge_index2 , None)
        x2 = self.graph_em_32(x2 , edge_index2 , None)
        
        s3 = self.graph_pl_33(x3 , edge_index3 , None)
        x3 = self.graph_em_33(x3 , edge_index3 , None)
        
        #print("3 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("3 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        
        x1 , edge_index1 , l31 , e31 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1) # x1 => number_of_sample , 1 , embedding_size
        x2 , edge_index2 , l32 , e32 = geom_nn.dense_diff_pool(x2 , edge_index2 , s2) # x2 => number_of_sample , 1 , embedding_size
        x3 , edge_index3 , l33 , e33 = geom_nn.dense_diff_pool(x3 , edge_index3 , s3) # x3 => number_of_sample , 1 , embedding_size
        
        #print("3 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x2.size()) 
        x = torch.cat([x1 , x2 , x3] , dim=1)
        s = self.graph_pl_4(x , view_edge , None)
        x = self.graph_em_4(x , view_edge , None)
        
        x , _ , lf , le = geom_nn.dense_diff_pool(x , view_edge , s)
        
        
        # final layer GNN 
        # x = torch.cat([ x1 , x2 , x3 ] , dim=-1)
        #print("Final size: " , x.size())
        # mean
        # x_mean = torch.mean(torch.stack([x1.squeeze(1),x2.squeeze(1)], dim=-1) , dim=-1)
        
        x = self.head(x.squeeze(1))
        return x , l11+l12+l13+l21+l22+l23+l31+l32+l33+lf , x1.squeeze(1) , x2.squeeze(1) , e11+e12+e13+e21+e22+e23+e31+e32+e33+le
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , lr=self.lr)
        return optimizer 
    
    def training_step(self , batch , batch_idx):
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 , batch_3 , view_edge = batch
        
        output , diffpool_loss , output_x1 , output_x2 , entropy_loss = self.forward(
            batch_1.x , batch_1.edge_index , batch_1.edge_attr , 
            batch_2.x , batch_2.edge_index , batch_2.edge_attr , 
            batch_3.x , batch_3.edge_index , batch_3.edge_attr , 
            batch_1.batch , batch_2.batch , batch_3.batch , 
            view_edge
        ) 
        #print("Output dimension: ", output.size())
        
        loss = self.loss(output , batch_1.y)  + diffpool_loss + entropy_loss #+ self.x1loss(output_x1 , batch_1.y) + self.x2loss(output_x2 , batch_2.y) 
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
        f1 = self.f1(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
        auc = self.auc(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
        
        self.log("train_acc" , acc , prog_bar=True, on_epoch=True)
        self.log("train_loss" , loss , prog_bar=True, on_epoch=True)
        self.log('train_auc' , auc)
        self.log('train_f1' , f1)
        return loss 
    
    def validation_step(self , batch , batch_idx): 
        batch_1 , batch_2 , batch_3 , view_edge = batch
        
        output , diffpool_loss , output_x1 , output_x2 , entropy_loss = self.forward(
            batch_1.x , batch_1.edge_index , batch_1.edge_attr , 
            batch_2.x , batch_2.edge_index , batch_2.edge_attr , 
            batch_3.x , batch_3.edge_index , batch_3.edge_attr , 
            batch_1.batch , batch_2.batch , batch_3.batch , 
            view_edge
        ) 
        # print(output.size())
        # print(batch_1.y)
        # print(batch_1)
        loss = self.loss(output , batch_1.y) + diffpool_loss + entropy_loss  #+ self.x1loss(output_x1 , batch_1.y) + self.x2loss(output_x2 , batch_2.y) 
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1) , batch_1.y)
        
        f1 = self.f1(torch.nn.functional.softmax(output , dim=-1) , batch_1.y)
        auc = self.auc(torch.nn.functional.softmax(output , dim=-1) , batch_1.y)
        self.log("val_acc" , acc , prog_bar=True, on_epoch=True)
        self.log("val_loss" , loss , prog_bar=True, on_epoch=True)
        self.log('val_auc' , auc , prog_bar=True)
        self.log('val_f1' , f1 , prog_bar=True) 
        
class MultiGraphGeneralPooling(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes):
        super().__init__()
        
        self.graph_model_1 = GCN( in_channels , hidden_channels , hidden_channels)
        self.graph_model_2 = GCN( in_channels , hidden_channels , hidden_channels)
        
        
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(4*hidden_channels , hidden_channels),
            torch.nn.Dropout(0.2),
            torch.nn.LeakyReLU(0.02), 
            torch.nn.BatchNorm1d(hidden_channels), 
            # torch.nn.Linear(2*hidden_channels , hidden_channels),
            # torch.nn.Dropout(0.2),
            # torch.nn.LeakyReLU(0.02), 
            # torch.nn.BatchNorm1d(hidden_channels), 
            # torch.nn.Linear(hidden_channels , hidden_channels),
            # torch.nn.Dropout(0.2),
            # torch.nn.LeakyReLU(0.02), 
            # torch.nn.BatchNorm1d(hidden_channels), 
            torch.nn.Linear(hidden_channels , num_classes), 
            torch.nn.Softmax()
        )
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass" , num_classes=num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , batch1_idx , batch2_idx):
        
        # Graph Message Passing
        x1 = self.graph_model_1(x1 , edge_index1 , edge_attr1)
        x2 = self.graph_model_1(x2 , edge_index2 , edge_attr2)
        # print("Graph model shape: ", x.size())
        
        # Graph Pooling 
        x1_mean = geom_nn.global_mean_pool(x1 , batch1_idx)
        x1_max = geom_nn.global_max_pool(x1 , batch1_idx)
        x2_mean = geom_nn.global_mean_pool(x2 , batch2_idx)
        x2_max = geom_nn.global_max_pool(x2 , batch2_idx)
        x = torch.cat([x1_mean , x1_max , x2_mean , x2_max] , dim=-1)
        
        # print("Graph pooling shape: " , x.size())
        
        # Dense layer 
        x = self.head(x)
        return x 
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , lr=1e-4)
        return optimizer 
    
    def training_step(self , batch , batch_idx):
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 , batch_3  = batch
        
        output = self.forward(batch_1.x , batch_1.edge_index , batch_1.edge_attr , batch_2.x , batch_2.edge_index , batch_2.edge_attr , batch_1.batch , batch_2.batch) 
        #print("Output dimension: ", output.size())
        
        loss = self.loss(output , batch_1.y)
        acc = self.acc(output , batch_1.y)
        f1 = self.f1(output , batch_1.y)
        auc = self.auc(output , batch_1.y)
        
        self.log("train_acc" , acc , prog_bar=True)
        self.log("train_loss" , loss , prog_bar=True)
        self.log('train_auc' , auc)
        self.log('train_f1' , f1)
        return loss 
    
    def validation_step(self , batch , batch_idx):
        batch_1 , batch_2 , batch_3 = batch
        
        output = self.forward(batch_1.x , batch_1.edge_index , batch_1.edge_attr , batch_2.x , batch_2.edge_index , batch_2.edge_attr , batch_1.batch , batch_2.batch) 
        loss = self.loss(output , batch_1.y)
        acc = self.acc(output , batch_1.y)
        
        f1 = self.f1(output , batch_1.y)
        auc = self.auc(output , batch_1.y)
        self.log("val_acc" , acc , prog_bar=True)
        self.log("val_loss" , loss , prog_bar=True)
        self.log('val_auc' , auc , prog_bar=True)
        self.log('val_f1' , f1 , prog_bar=True)

class BestModelTracker(Callback):
    
    def __init__(self) -> None:
        self.best_model = None 
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        outputs = trainer.logged_metrics
        
        if self.best_model == None: 
            self.best_model = {
                'best_val_acc': outputs['val_acc'].item(), 
                'best_epoch': trainer.current_epoch, 
                'best_val_auc': outputs['val_auc'].item(), 
                'best_val_f1': outputs['val_f1'].item(),
            }
        elif self.best_model['best_val_acc'] < outputs['val_acc'].item() :
            self.best_model = {
                'best_val_acc': outputs['val_acc'].item(), 
                'best_epoch': trainer.current_epoch, 
                'best_val_auc': outputs['val_auc'].item(), 
                'best_val_f1': outputs['val_f1'].item(),
            }

def main():
    
    parser = argparse.ArgumentParser("Multi/Single GNN")
    parser.add_argument("--model" , default='multigraph_diffpool' , choices=['multigraph_diffpool' , 'singlegraph_diffpool' , 'dmongraph_pool'])
    parser.add_argument("--hidden_embedding" , default=32 , type=int)
    parser.add_argument("--max_epoch" , type=int , default=100 , help="Maximum epochs")
    parser.add_argument("--lr" , type=float , default=1e-3 , help="Learning rate of the experiment")
    parser.add_argument("--build_graph", type=str , default='PPI' , choices=['PPI' , 'pearson'] )
    parser.add_argument("--edge_threshold" , type=float , default=1 , help="Edge threshold")
    parser.add_argument("--convolution" , type=str , default='SAGEConv' , choices=list(gnn.keys()) )
    parser.add_argument("--batch_size" , type=int, default=50)
    parser.add_argument("--runkfold"  , type=int , default=10)
    parser.add_argument("--disable_early_stopping" , action="store_true")
    
    args = parser.parse_args()
    
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")
    
    # read labels
    labels = os.path.join(base_path, "labels_tr.csv")
    df_labels = read_features_file(labels) 

    
    feature_info = {}
    
    ## mRNA Features
    feature1 = os.path.join(base_path, "1_tr.csv")
    df1 = read_features_file(feature1)
    name1 = os.path.join(base_path, "1_featname.csv")
    df1_header = read_features_file(name1)
    gp1 = generate_graph(df1 , df1_header , df_labels[0].tolist(), threshold=args.edge_threshold, rescale=True, integration=args.build_graph)
    
    # print(gp1[0])
    # degree = geom_utils.degree()
    # print(gp1[0].has_isolated_nodes()) 
    _ , _ , mask = geom_utils.remove_isolated_nodes(gp1[0].edge_index)
    feature_info.update({
        "feature1_isolated_node": gp1[0].x.shape[0] - mask.sum().item(), 
        "feature1_network_number_of_edge": gp1[0].edge_index.shape[1]
    })
    #print("Number of isolated node: " , gp1[0].x.shape[0] - mask.sum().item())
    
    ## DNA Methylation
    feature2 = os.path.join(base_path, "2_tr.csv")
    df2 = read_features_file(feature2)
    name2 = os.path.join(base_path, "2_featname.csv")
    df2_header = read_features_file(name2)
    gp2 = generate_graph(df2 , df2_header , df_labels[0].tolist(), threshold=args.edge_threshold, rescale=True , integration=args.build_graph)
    _ , _ , mask = geom_utils.remove_isolated_nodes(gp2[0].edge_index)
    feature_info.update({
        "feature2_isolated_node": gp2[0].x.shape[0] - mask.sum().item(), 
        "feature2_network_number_of_edge": gp2[0].edge_index.shape[1]
    })
    
    ## miRNA Feature
    feature3 = os.path.join(base_path , "3_tr.csv")
    df3 = read_features_file(feature3)
    name3 = os.path.join(base_path , "3_featname.csv")
    df3_header = read_features_file(name3)
    gp3 = generate_graph(df3 , None , df_labels[0].tolist() , integration= 'GO&KEGG' if args.build_graph == 'PPI' else 'pearson')
    # gp3 = generate_graph(df3 , df3_header , df_labels[0].tolist(), threshold=args.edge_threshold, rescale=True , integration='pearson')
    _ , _ , mask = geom_utils.remove_isolated_nodes(gp3[0].edge_index)
    feature_info.update({
        "feature3_isolated_node": gp3[0].x.shape[0] - mask.sum().item(), 
        "feature3_network_number_of_edge": gp3[0].edge_index.shape[1]
    })
    
    kf = StratifiedKFold(n_splits=10 , shuffle=True)
    batch_size = args.batch_size
    
    # model = MultiGraphGeneralPooling(1 , 32 , 5)
    mlflow.set_experiment("biomarker_detection")
    mlflow.pytorch.autolog(
        log_every_n_epoch=1, 
        log_every_n_step=0
    )
    
    def collate(data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])
        batchC = Batch.from_data_list([data[2] for data in data_list])
        view_edge = torch.ones(( len(data_list) , 3 , 3 ))
        return batchA, batchB , batchC , view_edge

    
    # with mlflow.start_run() as run:
        
    #     for arg in vars(args):
    #         mlflow.log_param(arg , getattr(args , arg))
    #         mlflow.log_params(feature_info)
        
    #     parent_run_metrics = {}
        
    for i , (train_index , test_index) in enumerate(kf.split(df2.values , df_labels[0].values)):
        
        datasetA_tr = [gp1[idx] for idx in train_index] 
        datasetB_tr = [gp2[idx] for idx in train_index]
        datasetC_tr = [gp3[idx] for idx in train_index]
        pair_dataset_tr = PairDataset(datasetA_tr , datasetB_tr , datasetC_tr)
        dataloader_tr = torch.utils.data.DataLoader(pair_dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
        
        datasetA_te = [gp1[idx] for idx in test_index] 
        datasetB_te = [gp2[idx] for idx in test_index]
        datasetC_te = [gp3[idx] for idx in test_index]
        pair_dataset_te = PairDataset(datasetA_te , datasetB_te , datasetC_te)
        dataloader_te = torch.utils.data.DataLoader(pair_dataset_te, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)
        
        # batch_size
        # Define model 
        
        if args.model == 'multigraph_diffpool':
            model = MultiGraphDiffPooling(1 , args.hidden_embedding , 5 , 1000, skip_connection=True , lr=args.lr)
            #mlflow.set_experiment("multigraph_diff_pooling")
        elif args.model == 'dmongraph_pool':
            model = DmonGraphPooling(1 , args.hidden_embedding , 5 ,1000 , args.lr)
        else: 
            model = SingleGraphDiffPooling(1 , args.hidden_embedding , 5 , 1000 , skip_connection=True , lr=args.lr)
            #mlflow.set_experiment("singlegraph_diff_pooling")
        
        mode = {
            'val_loss': 'min', 
            'val_accuracy': 'max'
        }
        
        callbacks = []
        if not args.disable_early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss' , patience=10 , mode='min')
            callbacks.append(early_stopping)
            
        # model checkpoint 
        checkpoint = ModelCheckpoint(monitor='val_acc' , mode='max' , save_top_k=1)
        callbacks.append(checkpoint)
        
        # model tracker 
        modelTracker = BestModelTracker()
        callbacks.append(modelTracker)
        
        # train model 
        trainer = pl.Trainer(
            max_epochs=args.max_epoch , 
            callbacks=callbacks, 
            # accumulate_grad_batches=3, 
            gradient_clip_val=0.5
        )
        
        
        with mlflow.start_run() as child_run:
            
            for arg in vars(args):
                mlflow.log_param(arg , getattr(args , arg))
                
            # mlflow.log_param("kfold" , i)
            mlflow.log_params(feature_info)
            
            trainer.fit(
                model=model , 
                train_dataloaders=dataloader_tr, 
                val_dataloaders=dataloader_te
            )
            
            #trainer.logged_metrics
            #run_data = mlflow.get_run(child_run.info.run_id)
            print(modelTracker.best_model)
            mlflow.log_metrics(modelTracker.best_model)
            # subrun_metrics = mlflow.get_run(child_run.info.run_id).data.metrics
            # for key , value in subrun_metrics.items():
            #     if not key in parent_run_metrics:
            #         parent_run_metrics[key] = [ value ]
            #     else:
            #         parent_run_metrics[key].append(value)
            
        if i+1 == args.runkfold :
            break
            
        # for key , value in parent_run_metrics.items():
        #     parent_run_metrics[key] = sum(parent_run_metrics[key])/len(parent_run_metrics[key])
        # mlflow.log_metrics(parent_run_metrics)
    
if __name__ == "__main__":
    main()
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
from utils import  generate_graph , read_features_file , get_omic_graph
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
    'GINConv': geom_nn.DenseGINConv , 
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
    
    def __init__(self , in_channels , hidden_channels , out_channels , skip_connection = True , lin=True , type='SAGEConv' , **args):
        super().__init__()
        self.conv1 = gnn[type](in_channels , hidden_channels , **args)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = gnn[type](hidden_channels , hidden_channels, **args)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = gnn[type](hidden_channels , out_channels, **args)
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
        
class ShortDenseGCN(torch.nn.Module):
    
    def __init__(self , in_channels , hidden_channels , out_channels , skip_connection = True , lin=True , type='SAGEConv' , **args):
        super().__init__()
        self.conv1 = gnn[type](in_channels , hidden_channels , **args)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = gnn[type](hidden_channels , out_channels, **args)
        self.batch_norm2 = torch.nn.BatchNorm1d(out_channels)
        
        self.skip_connection = skip_connection
        if lin:
            self.lin = torch.nn.Linear(
                hidden_channels + out_channels , out_channels
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
        
        if not self.skip_connection:
            return x2
        else: 
            x = torch.concat([x1 , x2] , dim = -1) # each output node have node_embedding dimension * 3
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

class DMonGraphConv(torch.nn.Module):
    
    def __init__(self, in_channels , hidden_channels , input_size , classes) -> None:
        super().__init__()
        
        self.conv1 = geom_nn.GCNConv(in_channels , hidden_channels)
        num_nodes = ceil(0.5 * input_size)
        self.pool1 = geom_nn.DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv2 = geom_nn.DenseGraphConv(hidden_channels, hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = geom_nn.DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv3 = geom_nn.DenseGraphConv(hidden_channels, hidden_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels)
        self.pool3  = geom_nn.DMoNPooling([hidden_channels , hidden_channels] , 1)
        
        self.conv4 = geom_nn.DenseGraphConv(hidden_channels , classes)
        self.batch_norm4 = torch.nn.BatchNorm1d(classes)
        
    
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
        
        _ , x , adj , sp3 , o3 , c3 = self.pool3(x , adj)
        
        x = self.bn(4 , x = self.conv4(x , adj))
        
        return x  , sp1+sp2+sp3 , o1+o2+o3 , c1+c2+c3
    
class DmonGraphPooling(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , input_size , lr=1e-4):
        super().__init__() 
        self.lr = lr
        self.automatic_optimization = False
        self.dmon_pool1 = DMonGraphConv(in_channels , hidden_channels , input_size , classes=num_classes)
        self.dmon_pool2 = DMonGraphConv(in_channels , hidden_channels , input_size , classes=num_classes)
        self.dmon_pool3 = DMonGraphConv(in_channels , hidden_channels , input_size , classes=num_classes)

        #self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*3, hidden_channels) , 
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
        self.loss1 = torch.nn.NLLLoss()
        self.loss2 = torch.nn.NLLLoss()
        self.loss3 = torch.nn.NLLLoss()
        self.acc = Accuracy(task='multiclass' , num_classes = num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
    
    def forward(self , x1 , edge_index1, batch1 , x2 , edge_index2, batch2 , x3 , edge_index3 , batch3 ):
        
        
        x1 , sp1 , o1 , c1 = self.dmon_pool1(x1 , edge_index1 , batch1)
        x2 , sp2 , o2 , c2 = self.dmon_pool2(x2 , edge_index2 , batch2)
        x3 , sp3 , o3 , c3 = self.dmon_pool3(x3 , edge_index3 , batch3)
        
        #x = self.lin1(x).relu()
        # = self.lin2(x)
        # print("x1 shape" , x1.size())
        # print("x2 shape" , x2.size())
        # print("x3 shape" , x3.size())
        
        #x = torch.concat([x1 , x2 , x3 ] , dim=-1).squeeze(1)
        #print("Output shape" , x.size())
        #x = self.head(x)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)
        #print(f"Dimension for x1 : {x1.size()}")
        #print(f"Dimesnion for x2 : {x2.size()}")
        #print(f"Dimesnion for x3 : {x3.size()}")
        return x1 , x2 , x3 , sp1+o1+c1 , sp2+o2+c2 , sp3+o3+c3 
     
    def configure_optimizers(self):
        optimizer_1 = optim.Adam(self.dmon_pool1.parameters() , lr=self.lr)
        optimizer_2 = optim.Adam(self.dmon_pool2.parameters() , lr=self.lr)
        optimizer_3 = optim.Adam(self.dmon_pool3.parameters() , lr=self.lr)
        return [ optimizer_1 , optimizer_2 , optimizer_3 ] 
    
    
    def training_step(self , batch ,  batch_idx):
        opt1 , opt2 , opt3 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()
        opt3.zero_grad()
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 , batch_3 , view_edge = batch
        
        o1 , o2 , o3 , dl1 , dl2 , dl3 = self.forward(batch_1.x , batch_1.edge_index , batch_1.batch  , batch_2.x , batch_2.edge_index , batch_2.batch , batch_3.x , batch_3.edge_index , batch_3.batch ) 
        #print("Output dimension: ", output.size())
        
        loss1 = self.loss1(o1 , batch_1.y) + dl1 
        acc1 = self.acc(o1 , batch_1.y)
        
        loss2 = self.loss2(o2 , batch_2.y) + dl2 
        acc2 = self.acc(o2 , batch_2.y)
        
        loss3 = self.loss3(o3 , batch_3.y) + dl3 
        acc3 = self.acc(o3 , batch_3.y)
        #f1 = self.f1(output , batch_1.y)
        #auc = self.auc(output , batch_1.y)
        self.log("train_acc_1" , acc1 , prog_bar=True , on_epoch=True)
        self.log("train_loss_1" , loss1 , prog_bar=True , on_epoch=True)
        self.log("train_acc_2" , acc2 , prog_bar=True , on_epoch=True)
        self.log("train_loss_2" , loss2 , prog_bar=True , on_epoch=True)
        self.log("train_acc_3" , acc3 , prog_bar=True , on_epoch=True)
        self.log("train_loss_3" , loss3 , prog_bar=True , on_epoch=True)
        
        self.manual_backward(loss1)
        self.manual_backward(loss2)
        self.manual_backward(loss3)
        
        opt1.step()
        opt2.step()
        opt3.step()
        #self.log('train_auc' , auc)
        #self.log('train_f1' , f1)
        #return loss 
    
    def validation_step(self , batch , batch_idx):
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 , batch_3 , view_edge = batch
        
        o1 , o2 , o3 , dl1 , dl2 , dl3 = self.forward(batch_1.x , batch_1.edge_index , batch_1.batch  , batch_2.x , batch_2.edge_index , batch_2.batch , batch_3.x , batch_3.edge_index , batch_3.batch ) 
        #print("Output dimension: ", output.size())
        
        loss1 = self.loss1(o1 , batch_1.y) + dl1 
        acc1 = self.acc(o1 , batch_1.y)
        
        loss2 = self.loss2(o2 , batch_2.y) + dl2 
        acc2 = self.acc(o2 , batch_2.y)
        
        loss3 = self.loss3(o3 , batch_3.y) + dl3 
        acc3 = self.acc(o3 , batch_3.y)
        
        self.log("val_acc_1" , acc1 , prog_bar=True , on_epoch=True)
        self.log("val_loss_1" , loss1 , prog_bar=True , on_epoch=True)
        self.log("val_acc_2" , acc2 , prog_bar=True , on_epoch=True)
        self.log("val_loss_2" , loss2 , prog_bar=True , on_epoch=True)
        self.log("val_acc_3" , acc3 , prog_bar=True , on_epoch=True)
        self.log("val_loss_3" , loss3 , prog_bar=True , on_epoch=True)
        #self.log('train_auc' , auc)
        #self.log('train_f1' , f1)
        #return loss 

dense_gcn = {
    'DenseGCN': DenseGCN, 
    'ShortDenseGCN': ShortDenseGCN, 
}

class GraphDiffPoolConv(torch.nn.Module): 
    
    def __init__(self , in_channels , hidden_channels , num_classes , input_size , skip_connection=False, type='SAGEConv' , linear=False , gcn_conv='DenseGCN' , **args):
        super().__init__()
        
        self.skip_connection = skip_connection
        self.mode = "pretrain"
        
        # Pooling layer 1 
        node_size = min( 100 , ceil(input_size * 0.5))
        self.graph_em_11 = dense_gcn[gcn_conv]( in_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type , **args)
        self.graph_pl_11 = dense_gcn[gcn_conv]( in_channels , hidden_channels , node_size , skip_connection , type=type , **args)
        
        # Pooling layer 2
        node_size = ceil(node_size * 0.5)
        multiplyer = 3 if gcn_conv == 'DenseGCN' else 2
        input_hidden_channels = multiplyer * hidden_channels if skip_connection  else hidden_channels
        self.graph_em_21 = dense_gcn[gcn_conv]( input_hidden_channels , hidden_channels , hidden_channels , skip_connection , lin=False, type=type , **args)
        self.graph_pl_21 = dense_gcn[gcn_conv]( input_hidden_channels , hidden_channels , node_size , skip_connection, type=type , **args)
        
        # Pooling layer 3 
        node_size = ceil(node_size * 0.5)
        multiplyer = 3 if gcn_conv == 'DenseGCN' else 2
        input_hidden_channels = multiplyer * hidden_channels if skip_connection  else hidden_channels
        self.graph_em_31 = dense_gcn[gcn_conv]( input_hidden_channels , hidden_channels , num_classes if not linear else hidden_channels  , lin=True, type=type, **args)
        self.graph_pl_31 = dense_gcn[gcn_conv]( input_hidden_channels , hidden_channels , 1  , skip_connection, type=type , **args)
        
        if linear:
            self.linear = torch.nn.Linear(hidden_channels , num_classes)
        else: 
            self.linear = None 
            
    def forward(self , x1 , edge_index1 , edge_attr1 , batch1_idx):
        
        # Convert to dense batch
        x1 , _ = geom_utils.to_dense_batch(x1 , batch1_idx)
        edge_index1 = geom_utils.to_dense_adj(edge_index1, batch1_idx  , edge_attr1).squeeze(-1)
        
        
        # first layer pooling 
        #print("0 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        s1 = self.graph_pl_11(x1 , edge_index1 , None)
        x1 = self.graph_em_11(x1 , edge_index1 , None)
        #print("1 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("1 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        x1 , edge_index1 , l11 , e11 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        #print("1 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # second layer pooling 
        s1 = self.graph_pl_21(x1 , edge_index1 , None)
        x1 = self.graph_em_21(x1 , edge_index1 , None)
        #print("2 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("2 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        x1 , edge_index1 , l21 , e21 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        #print("2 ---- New shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        
        # final layer pooling 
        s1 = self.graph_pl_31(x1 , edge_index1 , None)
        x1 = self.graph_em_31(x1 , edge_index1 , None)
        
        x1 , edge_index1 , l31 , e31 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1)
        
        if self.linear is not None: 
            x_linear = self.linear(x1.squeeze(1))
        else: 
            x_linear = None
        
        return x_linear , x1 , s1 , edge_index1 , l11+l21+l31 , e11+e21+e31

class GraphGeneralPooling(torch.nn.Module):
    def __init__(self , hidden_channels , num_classes ):
        super().__init__()
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3*hidden_channels , 1024),
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
        
    def forward(self , x1 , x2 , x3 ):
        x_concat = torch.cat([x1 , x2 , x3] , dim = -1).squeeze(1) # number_of_sample , 3 * hidden_embedding
        
        x = self.head(x_concat)
        
        return x
    
class GraphAttentionDifferencePooling(torch.nn.Module): 
    
    def __init__(self , in_channels , hidden_channels , num_classes , skip_connection , type='SAGEConv'):
        super().__init__()
        
        self.graph_em = DenseGCN(in_channels , hidden_channels , hidden_channels , lin=True , type=type)
        self.graph_pl = DenseGCN(in_channels , hidden_channels , 1 , skip_connection , type='GATConv')
        
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
        
    def forward(self , x1, x2, x3  , view_edge):
        
        # x1 , edge_index1 , l31 , e31 = geom_nn.dense_diff_pool(x1 , edge_index1 , s1) # x1 => number_of_sample , 1 , embedding_size
        # x2 , edge_index2 , l32 , e32 = geom_nn.dense_diff_pool(x2 , edge_index2 , s2) # x2 => number_of_sample , 1 , embedding_size
        # x3 , edge_index3 , l33 , e33 = geom_nn.dense_diff_pool(x3 , edge_index3 , s3) # x3 => number_of_sample , 1 , embedding_size
        
        #print(f"X size: {x1.size()}")
        x = torch.cat([x1 , x2 , x3] , dim=1)
        #print(f"X size: {x.size()} | S size: {s.size()}")
        s = self.graph_pl(x , view_edge , None)
        x = self.graph_em(x , view_edge , None)
        #print(f"s : {s.size()} | x: {x.size()}")
        x , _ , lf , le = geom_nn.dense_diff_pool(x , view_edge , s)
        #print(f"X size: {x.size()}")
        x = self.head(x.squeeze(1))
        
        return x , lf+le
        
class MultiGraphDiffPooling(pl.LightningModule):
    
        
    def __init__(self , in_channels , hidden_channels , num_classes , input_size , skip_connection=False , lr=1e-3, type='SAGEConv', pretrain_epoch = 0 , decay=0 , linear=False , slr=1e-4 , tlr=1e-4 , gcn_conv='DenseGCN' , last_pooling = 'general' ,**args):
        super().__init__()
        
        self.skip_connection = skip_connection 
        self.lr = lr
        self.mode = "pretrain"
        self.automatic_optimization = False
        self.pretrain_epoch = pretrain_epoch
        self.decay = decay
        self.linear = linear
        
        self.graph_diff_pool1 = GraphDiffPoolConv(in_channels , hidden_channels , num_classes , input_size , skip_connection , type , linear=linear , gcn_conv=gcn_conv , **args)
        self.graph_diff_pool2 = GraphDiffPoolConv(in_channels , hidden_channels , num_classes , input_size , skip_connection , type , linear=linear , gcn_conv=gcn_conv , **args)
        self.graph_diff_pool3 = GraphDiffPoolConv(in_channels , hidden_channels , num_classes , input_size , skip_connection , type , linear=linear , gcn_conv=gcn_conv , **args)
        
        if last_pooling == 'general':
            self.graph_general_pool = GraphGeneralPooling(hidden_channels,num_classes)
        else:
            self.graph_attn_pool = GraphAttentionDifferencePooling(num_classes if not self.linear else hidden_channels , hidden_channels , num_classes , skip_connection , 'GATConv' )
        self.last_pooling = last_pooling
        
        # self.sofmax = torch.nn.Softmax()
        self.loss = torch.nn.CrossEntropyLoss()
        # self.x1loss = torch.nn.CrossEntropyLoss()
        # self.x2loss = torch.nn.CrossEntropyLoss()
        # self.x3loss = torch.nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass' , num_classes = num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx , view_edge):
        
        l_output1 , output1 , s1 , edge_index1 , lp1 , le1 = self.graph_diff_pool1( x1 , edge_index1 , edge_attr1 , batch1_idx  ) # Expected dimension for output -> [ Batch , 1 , number_of_classes ]
        l_output2 , output2 , s2 , edge_index2 , lp2 , le2 = self.graph_diff_pool2( x2 , edge_index2 , edge_attr2 , batch2_idx  ) # Expected dimension for output -> [ Batch , 1 , number_of_classes ]
        l_output3 , output3 , s3 , edge_index3 , lp3 , le3 = self.graph_diff_pool3( x3 , edge_index3 , edge_attr3 , batch3_idx  ) # Expected dimension for output -> [ Batch , 1 , number_of_classes ]
        #print("3 ---- Shape of x1: " , x1.size() , "| Shape of x2: " , x2.size())
        #print("3 ---- Shape of s1: " , s1.size() , "| Shape of s2: " , s2.size())
        #print(f"Size: {output1.size()} {output2.size()} {output3.size()}")
        # if self.last_pooling == 'general':
        #     x = self.graph_general_pool(output1 , output2 , output3)
        #     if self.linear:
        #         return x , None , l_output1 , l_output2 , l_output3 , lp1+le1 , lp2+le2 , lp3+le3
        #     else: 
        #         return x , None , output1.squeeze(1) , output2.squeeze(1) , output3.squeeze(1) , lp1+le1 , lp2+le2 , lp3+le3
        # else: 
        #     x , xloss = self.graph_attn_pool(output1 , output2 , output3 , view_edge)
        
        #     if self.linear:
        #         return x , xloss , l_output1 , l_output2 , l_output3 , lp1+le1 , lp2+le2 , lp3+le3
        #     else: 
        #         return x , xloss , output1.squeeze(1) , output2.squeeze(1) , output3.squeeze(1) , lp1+le1 , lp2+le2 , lp3+le3
        return l_output1 , l_output2 , l_output3 , output1 , output2 , output3 , lp1+le1 , lp2+le2 , lp3+le3
        
    def configure_optimizers(self):
        optimizer1 = optim.Adam(self.graph_diff_pool1.parameters() , lr=self.lr , weight_decay=self.decay)
        optimizer2 = optim.Adam(self.graph_diff_pool2.parameters() , lr=self.lr , weight_decay=self.decay)
        optimizer3 = optim.Adam(self.graph_diff_pool3.parameters() , lr=self.lr , weight_decay=self.decay)
        if self.last_pooling == 'general':
            optimizer = optim.Adam(self.graph_general_pool.parameters() , lr=self.lr , weight_decay=self.decay)
        else:
            optimizer = optim.Adam(self.graph_attn_pool.parameters() , lr=self.lr , weight_decay=self.decay)
        return [ optimizer1 , optimizer2 , optimizer3 , optimizer ] 
    
    def training_step(self , batch , batch_idx):
        
        opt1 , opt2 , opt3 , opt = self.optimizers()
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 , batch_3 , view_edge = batch
        
        output_x1 , output_x2 , output_x3 , houtput_x1 , houtput_x2 , houtput_x3 , loss_x1 , loss_x2 , loss_x3 = self.forward(
            batch_1.x , batch_1.edge_index , batch_1.edge_attr , 
            batch_2.x , batch_2.edge_index , batch_2.edge_attr , 
            batch_3.x , batch_3.edge_index , batch_3.edge_attr , 
            batch_1.batch , batch_2.batch , batch_3.batch , 
            view_edge
        ) 
        
        # optimize loss per omic data 
        opt1.zero_grad()
        loss1 = self.loss(output_x1 , batch_1.y) + loss_x1 
        self.manual_backward(loss1)
        opt1.step() 
        
        opt2.zero_grad()
        loss2 = self.loss(output_x2 , batch_2.y) + loss_x2 
        self.manual_backward(loss2)
        opt2.step() 
         
        opt3.zero_grad()
        loss3 = self.loss(output_x3 , batch_3.y) + loss_x3 
        self.manual_backward(loss3)
        opt3.step() 
        
        # calculate acc per omic data 
        acc1 = self.acc(torch.nn.functional.softmax(output_x1 , dim=-1)  , batch_1.y)
        acc2 = self.acc(torch.nn.functional.softmax(output_x2 , dim=-1)  , batch_2.y)
        acc3 = self.acc(torch.nn.functional.softmax(output_x3 , dim=-1)  , batch_3.y)
        
        
        self.log("train_loss_x1" , loss1 , on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log("train_loss_x2" , loss2 , on_epoch=True , batch_size=batch_2.batch.shape[0])
        self.log("train_loss_x3" , loss3 , on_epoch=True , batch_size=batch_3.batch.shape[0])
        self.log("train_acc_x1" , acc1 , on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log("train_acc_x2" , acc2 , on_epoch=True , batch_size=batch_2.batch.shape[0])
        self.log("train_acc_x3" , acc3 , on_epoch=True , batch_size=batch_3.batch.shape[0])
        
        
        if not self.mode == 'pretrain':
            opt.zero_grad()
            output_x1 , output_x2 , output_x3 , houtput_x1 , houtput_x2 , houtput_x3 , loss_x1 , loss_x2 , loss_x3 = self.forward(
                batch_1.x , batch_1.edge_index , batch_1.edge_attr , 
                batch_2.x , batch_2.edge_index , batch_2.edge_attr , 
                batch_3.x , batch_3.edge_index , batch_3.edge_attr , 
                batch_1.batch , batch_2.batch , batch_3.batch , 
                view_edge
            ) 
            if self.last_pooling == 'general':
                output = self.graph_general_pool(houtput_x1 , houtput_x2 , houtput_x3)
                loss = self.loss(output , batch_1.y)    
            else:
                output , diffpool_entropy_loss = self.graph_general_pool(houtput_x1 , houtput_x2 , houtput_x3 , view_edge)
                loss = self.loss(output , batch_1.y) + diffpool_entropy_loss # + self.x1loss(output_x1 , batch_1.y) + self.x2loss(output_x2 , batch_2.y) 
            acc = self.acc(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
            f1 = self.f1(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
            auc = self.auc(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
            
            self.manual_backward(loss)
            opt.step()
        else: 
            loss = torch.tensor(0 , dtype=torch.float)
            acc = torch.tensor(0 , dtype=torch.float)
            f1 = torch.tensor(0 , dtype=torch.float)
            auc = torch.tensor(0 , dtype=torch.float)
            # opt1.step()
            # opt2.step()
            # opt3.step()
        
       
        self.log("train_acc" , acc , prog_bar=True, on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log("train_loss" , loss , prog_bar=True, on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log('train_auc' , auc , batch_size=batch_1.batch.shape[0])
        self.log('train_f1' , f1 , batch_size=batch_1.batch.shape[0])
        
    
    def validation_step(self , batch , batch_idx): 
        
        #print("x shape:", batch.x.size() , "| egdge_index shape: " , batch.edge_index.size() , "| edge_attr shape: ", batch.edge_attr.size() , "| batch shape: ", batch.batch.size())
        batch_1 , batch_2 , batch_3 , view_edge = batch
        
        output_x1 , output_x2 , output_x3 , houtput_x1 , houtput_x2 , houtput_x3 , loss_x1 , loss_x2 , loss_x3 = self.forward(
            batch_1.x , batch_1.edge_index , batch_1.edge_attr , 
            batch_2.x , batch_2.edge_index , batch_2.edge_attr , 
            batch_3.x , batch_3.edge_index , batch_3.edge_attr , 
            batch_1.batch , batch_2.batch , batch_3.batch , 
            view_edge
        ) 
        
        # optimize loss per omic data 
        loss1 = self.loss(output_x1 , batch_1.y) + loss_x1 
        loss2 = self.loss(output_x2 , batch_2.y) + loss_x2 
        loss3 = self.loss(output_x3 , batch_3.y) + loss_x3 
        
        # calculate acc per omic data 
        acc1 = self.acc(torch.nn.functional.softmax(output_x1 , dim=-1)  , batch_1.y)
        acc2 = self.acc(torch.nn.functional.softmax(output_x2 , dim=-1)  , batch_2.y)
        acc3 = self.acc(torch.nn.functional.softmax(output_x3 , dim=-1)  , batch_3.y)
        
        
        self.log("val_loss_x1" , loss1 , on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log("val_loss_x2" , loss2 , on_epoch=True , batch_size=batch_2.batch.shape[0])
        self.log("val_loss_x3" , loss3 , on_epoch=True , batch_size=batch_3.batch.shape[0])
        self.log("val_acc_x1" , acc1 , on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log("val_acc_x2" , acc2 , on_epoch=True , batch_size=batch_2.batch.shape[0])
        self.log("val_acc_x3" , acc3 , on_epoch=True , batch_size=batch_3.batch.shape[0])
        
        
        if not self.mode == 'pretrain':
            if self.last_pooling == 'general':
                output = self.graph_general_pool(houtput_x1 , houtput_x2 , houtput_x3)
                loss = self.loss(output , batch_1.y)    
            else:
                output , diffpool_entropy_loss = self.graph_general_pool(houtput_x1 , houtput_x2 , houtput_x3 , view_edge)
                loss = self.loss(output , batch_1.y) + diffpool_entropy_loss # + self.x1loss(output_x1 , batch_1.y) + self.x2loss(output_x2 , batch_2.y) 
            acc = self.acc(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
            f1 = self.f1(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
            auc = self.auc(torch.nn.functional.softmax(output , dim=-1)  , batch_1.y)
        else: 
            loss = torch.tensor(0 , dtype=torch.float)
            acc = torch.tensor(0 , dtype=torch.float)
            f1 = torch.tensor(0 , dtype=torch.float)
            auc = torch.tensor(0 , dtype=torch.float)
        
       
        self.log("val_acc" , acc , prog_bar=True, on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log("val_loss" , loss , prog_bar=True, on_epoch=True , batch_size=batch_1.batch.shape[0])
        self.log('val_auc' , auc , batch_size=batch_1.batch.shape[0])
        self.log('val_f1' , f1 , batch_size=batch_1.batch.shape[0])
        
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

class UpdateOptimizer(Callback):
    
    def __init__(self , epoch=10 , decay=0 , slr=1e-5 , tlr=1e-6 ) -> None:
        super().__init__()
        self.epoch = epoch
        self.decay = decay 
        self.slr = slr 
        self.tlr = tlr
        
    def on_train_epoch_start(self , trainer:pl.Trainer, pl_module: pl.LightningModule) -> None :
        
        if trainer.model.mode == 'pretrain' and trainer.current_epoch == self.epoch: 
            trainer.model.mode = 'train'
            
            optimizer1 = optim.Adam(trainer.model.graph_diff_pool1.parameters() , lr=self.slr , weight_decay=self.decay)
            optimizer2 = optim.Adam(trainer.model.graph_diff_pool2.parameters() , lr=self.slr , weight_decay=self.decay)
            optimizer3 = optim.Adam(trainer.model.graph_diff_pool3.parameters() , lr=self.slr , weight_decay=self.decay)
            if trainer.model.last_pooling == 'general':
                optimizer = optim.Adam(trainer.model.graph_general_pool.parameters() , lr=self.tlr , weight_decay=self.decay)
            else:
                optimizer = optim.Adam(trainer.model.graph_attn_pool.parameters() , lr=self.tlr , weight_decay=self.decay)
            trainer.optimizers = [ optimizer1 , optimizer2 , optimizer3 , optimizer ]
        
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
    parser.add_argument("--model" , default='multigraph_diffpool' , choices=['multigraph_diffpool' , 'dmongraph_pool'])
    parser.add_argument('--gcn_conv' , default='DenseGCN', choices=dense_gcn.keys())
    parser.add_argument("--hidden_embedding" , default=32 , type=int)
    parser.add_argument("--max_epoch" , type=int , default=100 , help="Maximum epochs")
    parser.add_argument("--lr" , type=float , default=1e-3 , help="Learning rate of the experiment")
    parser.add_argument("--build_graph", type=str , default='PPI' , choices=['PPI' , 'pearson','cosine_similarity' , 'dynamic'] )
    parser.add_argument("--edge_threshold" , type=float , default=1 , help="Edge threshold")
    parser.add_argument("--convolution" , type=str , default='SAGEConv' , choices=list(gnn.keys()) )
    parser.add_argument("--batch_size" , type=int, default=50)
    parser.add_argument("--runkfold"  , type=int , default=10)
    parser.add_argument("--pretrain_epoch" , type=int , default=0)
    parser.add_argument("--disable_early_stopping" , action="store_true")
    parser.add_argument("--use_quantile" , action="store_true")
    parser.add_argument("--decay" , type=float , default=0.0)
    parser.add_argument("--linear" , action="store_true")
    parser.add_argument("--conv_dropout" , type=float , default=0.0)
    parser.add_argument("--slr" , type=float , default=1e-4)
    parser.add_argument("--tlr" , type=float , default=1e-4)
    parser.add_argument("--last_pooling" , type=str , default='general' , choices=['general', 'diff_pooling'])
    parser.add_argument("--filter_ppi" , type=int , default=None)
    parser.add_argument("--filter_p_value" , type=float , default=None)
    parser.add_argument("--weight" , action='store_true')
    
    args = parser.parse_args()
    
    ## Checking 
    if args.model == 'multigraph_diffpool':
        if args.pretrain_epoch >= args.max_epoch:
            print("Pretrain epoch must be smaller than max_epoch. Exiting")
            exit()
    
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")
    
    # read labels
    labels_train = os.path.join(base_path, "labels_tr.csv")
    df_labels_train = read_features_file(labels_train)
    
    labels_test = os.path.join(base_path, "labels_tr.csv")
    df_labels_test = read_features_file(labels_test)  

    
    feature_info = {}
    
    ## mRNA Features
    if args.build_graph == 'dynamic':
        gp1_train , feat1_n , feat1_e , feat1_d = get_omic_graph('1_tr.csv' , '1_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
        gp2_train , feat2_n , feat2_e , feat2_d = get_omic_graph('2_tr.csv' , '2_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
        gp3_train , feat3_n , feat3_e , feat3_d = get_omic_graph('3_tr.csv' , '3_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
        gp1_test , _ , _ , _ = get_omic_graph('1_te.csv' , '1_featname_conversion.csv' , 'labels_te.csv')
        gp2_test , _ , _ , _ = get_omic_graph('2_te.csv' , '2_featname_conversion.csv' , 'labels_te.csv')
        gp3_test , _ , _ , _= get_omic_graph('3_te.csv' , '3_featname_conversion.csv' , 'labels_te.csv')
        
        feature_info.update({
            "feature1_isolated_node": feat1_n, 
            "feature1_network_number_of_edge": feat1_e,
            "feature1_isolated_node": feat2_n, 
            "feature1_network_number_of_edge": feat2_e,
            "feature1_isolated_node": feat3_n, 
            "feature1_network_number_of_edge": feat3_e
        })
        
    else:
        name1 = os.path.join(base_path, "1_featname.csv")
        df1_header = read_features_file(name1)
        
        feature1_train = os.path.join(base_path, "1_tr.csv")
        df1_train = read_features_file(feature1_train)
        gp1_train = generate_graph(df1_train , df1_header , df_labels_train[0].tolist(), threshold=args.edge_threshold, rescale=True, integration=args.build_graph , use_quantile=args.use_quantile)
        
        feature1_test = os.path.join(base_path, "1_te.csv")
        df1_test = read_features_file(feature1_test)
        gp1_test = generate_graph(df1_test , df1_header , df_labels_test[0].tolist(), threshold=args.edge_threshold, rescale=True, integration=args.build_graph , use_quantile=args.use_quantile)
        
        _ , _ , mask = geom_utils.remove_isolated_nodes(gp1_train[0].edge_index)
        feature_info.update({
            "feature1_isolated_node": gp1_train[0].x.shape[0] - mask.sum().item(), 
            "feature1_network_number_of_edge": gp1_train[0].edge_index.shape[1]
        })
        
        
        ## DNA Methylation
        name2 = os.path.join(base_path, "2_featname.csv")
        df2_header = read_features_file(name2)
        
        feature2_train = os.path.join(base_path, "2_tr.csv")
        df2_train = read_features_file(feature2_train)
        gp2_train = generate_graph(df2_train , df2_header , df_labels_train[0].tolist(), threshold=args.edge_threshold, rescale=True , integration=args.build_graph, use_quantile=args.use_quantile)
        
        feature2_test = os.path.join(base_path, "2_te.csv")
        df2_test = read_features_file(feature2_test)
        gp2_test = generate_graph(df2_test , df2_header , df_labels_test[0].tolist(), threshold=args.edge_threshold, rescale=True , integration=args.build_graph, use_quantile=args.use_quantile)
        
        _ , _ , mask = geom_utils.remove_isolated_nodes(gp2_train[0].edge_index)
        feature_info.update({
            "feature2_isolated_node": gp2_train[0].x.shape[0] - mask.sum().item(), 
            "feature2_network_number_of_edge": gp2_train[0].edge_index.shape[1]
        })
        
        ## miRNA Feature
        name3 = os.path.join(base_path , "3_featname.csv")
        df3_header = read_features_file(name3)
        
        feature3_train = os.path.join(base_path , "3_tr.csv")
        df3_train = read_features_file(feature3_train)
        gp3_train = generate_graph(df3_train , df3_header , df_labels_train[0].tolist() , threshold=args.edge_threshold, rescale=True , integration= 'GO&KEGG' if args.build_graph == 'PPI' else 'pearson' , use_quantile=args.use_quantile)
        
        feature3_test = os.path.join(base_path , "3_te.csv")
        df3_test = read_features_file(feature3_test)
        gp3_test = generate_graph(df3_test , df3_header , df_labels_test[0].tolist() , threshold=args.edge_threshold, rescale=True , integration= 'GO&KEGG' if args.build_graph == 'PPI' else 'pearson' , use_quantile=args.use_quantile)
        
        # gp3 = generate_graph(df3 , df3_header , df_labels[0].tolist(), threshold=args.edge_threshold, rescale=True , integration='pearson')
        _ , _ , mask = geom_utils.remove_isolated_nodes(gp3_train[0].edge_index)
        feature_info.update({
            "feature3_isolated_node": gp3_train[0].x.shape[0] - mask.sum().item(), 
            "feature3_network_number_of_edge": gp3_train[0].edge_index.shape[1]
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
    
    datasetA_tr = gp1_train
    datasetB_tr = gp2_train
    datasetC_tr = gp3_train
    pair_dataset_tr = PairDataset(datasetA_tr , datasetB_tr , datasetC_tr)
    dataloader_tr = torch.utils.data.DataLoader(pair_dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    
    datasetA_te = gp1_test 
    datasetB_te = gp2_test
    datasetC_te = gp3_test
    pair_dataset_te = PairDataset(datasetA_te , datasetB_te , datasetC_te)
    dataloader_te = torch.utils.data.DataLoader(pair_dataset_te, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)
    
    # batch_size
    # Define model 
    if args.model == 'multigraph_diffpool':
        if args.pretrain_epoch >= args.max_epoch:
            print("Pretrain epoch must be smaller than max_epoch. Exiting")
            exit()
        
        model = MultiGraphDiffPooling(
            1 , args.hidden_embedding , 5 , 1000 if args.build_graph != 'dynamic' else 500, 
            skip_connection=True , 
            lr=args.lr , 
            pretrain_epoch=args.pretrain_epoch,
            decay=args.decay, 
            linear=args.linear ,
            slr=args.slr, 
            tlr=args.tlr, 
            gcn_conv=args.gcn_conv, 
            last_pooling=args.last_pooling
        )
        #mlflow.set_experiment("multigraph_diff_pooling")
    elif args.model == 'dmongraph_pool':
        model = DmonGraphPooling(1 , args.hidden_embedding , 5 ,1000 , args.lr)
    
    mode = {
        'val_loss': 'min', 
        'val_accuracy': 'max'
    }
    
    callbacks = []
    if not args.disable_early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss' , patience=10 , mode='min')
        callbacks.append(early_stopping)
        
    # model checkpoint 
    # checkpoint = ModelCheckpoint(monitor='val_acc' , mode='max' , save_top_k=1)
    # callbacks.append(checkpoint)
    
    # model tracker 
    modelTracker = BestModelTracker()
    callbacks.append(modelTracker)
    
    if args.model == 'multigraph_diffpool': 
        dynamicOptim = UpdateOptimizer(epoch=args.pretrain_epoch,decay=args.decay, slr=args.slr, tlr=args.tlr)
        callbacks.append(dynamicOptim)
    
    # train model 
    trainer = pl.Trainer(
        max_epochs=args.max_epoch , 
        callbacks=callbacks, 
        # accumulate_grad_batches=3, 
        # gradient_clip_val=0.5
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
            
            
        # for key , value in parent_run_metrics.items():
        #     parent_run_metrics[key] = sum(parent_run_metrics[key])/len(parent_run_metrics[key])
        # mlflow.log_metrics(parent_run_metrics)
    
if __name__ == "__main__":
    main()
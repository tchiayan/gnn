from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
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


# Get the pooling layer 
geom_model = {
    'GATConv': geom_nn.GATConv , 
}

# implement GAT Convolution and SAGPool
class GrapConvolution(torch.nn.Module):
    def __init__(self , in_channels , hidden_channels , out_channels , jump = True , **args):
        super().__init__() 
        
        # Graph Convolution
        self.graph_conv1 = geom_nn.GATConv(in_channels=in_channels , out_channels=hidden_channels, **args)
        self.graph_conv2 = geom_nn.GATConv(in_channels=hidden_channels , out_channels=hidden_channels, **args)
        self.graph_conv3 = geom_nn.GATConv(in_channels=hidden_channels , out_channels=out_channels, **args)
        
        
        self.batch_norm1 = geom_nn.BatchNorm(in_channels=hidden_channels)
        self.batch_norm2 = geom_nn.BatchNorm(in_channels=hidden_channels)
        self.batch_norm3 = geom_nn.BatchNorm(in_channels=out_channels)
        
        self.jump = jump
        
    def forward(self , x , edge_index , edge_attr = None): 
        
        if edge_attr is not None:
            x1 = self.graph_conv1(x , edge_index , edge_attr).relu() # batch
            x1 = self.batch_norm1(x1)
            x2 = self.graph_conv2(x1 , edge_index , edge_attr).relu()
            x2 = self.batch_norm2(x2)
            x3 = self.graph_conv3(x2 , edge_index , edge_attr).relu()
            x3 = self.batch_norm3(x3)
        else: 
            x1 = self.graph_conv1(x , edge_index).relu() # batch
            x1 = self.batch_norm1(x1)
            x2 = self.graph_conv2(x1 , edge_index).relu()
            x2 = self.batch_norm2(x2)
            x3 = self.graph_conv3(x2 , edge_index).relu()
            x3 = self.batch_norm3(x3)
            
        # Jumping knowledge 
        # x = torch.stack([x1 , x2 , x3], dim=-1).mean(dim=-1)
        if self.jump:
            x = torch.concat([x1,x2,x3] , dim=-1)
            return x 
        else: 
            return x3
        

class GraphClassification(pl.LightningModule):
    
    def __init__(self, in_channels , hidden_channels , num_classes , lr=0.0001) -> None:
        super().__init__()
        
        self.lr = lr
        
        # Graph Convolution
        self.graph_conv1 = GrapConvolution(in_channels , hidden_channels , hidden_channels)
        
        # Graph Pooling 
        self.pooling = geom_nn.TopKPooling(in_channels=hidden_channels*3 , ratio=0.5) # TopK pooling much stable than SAGPooling
        # self.pooling = geom_nn.DMoNPooling(channels=hidden_channels*3 , k=100)
        
        # Graph Convolution
        self.graph_conv2 = GrapConvolution(hidden_channels*3 , hidden_channels , hidden_channels)
        
        # Graph Pooling 
        self.pooling2 = geom_nn.TopKPooling(in_channels=hidden_channels*3 , ratio=0.5) # TopK pooling much stable than SAGPooling
        # self.pooling = geom_nn.DMoNPooling(channels=hidden_channels*3 , k=10)
        
        # Graph Convolution 
        # self.graph_conv3 = GrapConvolution(hidden_channels*3 , hidden_channels , hidden_channels)
        
        # Graph Pooling 
        # self.pooling3 = geom_nn.TopKPooling(in_channels=hidden_channels*3 , ratio=0.5)
        
        ## MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*3 , hidden_channels*4), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*4), 
            torch.nn.Linear(hidden_channels*4 , hidden_channels*2), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*2), 
            torch.nn.Linear(hidden_channels*2 , num_classes)
        )
        
        self.acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        
    def forward(self , x , edge_index , edge_attr , batch):
        
        # First layer graph convolution
        x = self.graph_conv1(x , edge_index , edge_attr)
        x , edge_index , _ , batch ,  _ , _ = self.pooling(x , edge_index , edge_attr , batch)
        
        # Second layer graph convolution
        x = self.graph_conv2(x , edge_index)
        x , edge_index , _ , batch ,  _ , _ = self.pooling2(x , edge_index , batch=batch)
        
        # Third layer graph convolution
        # x = self.graph_conv3(x , edge_index)
        # x , edge_index , _ , batch ,  _ , _ = self.pooling3(x , edge_index , batch=batch)
        
        x = geom_nn.global_mean_pool(x , batch)

        x = self.mlp(x)
        
        return x
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters() , lr= self.lr , weight_decay=0.0001) 
    
    def training_step(self , batch):
        x , edge_index , edge_attr , batch_idx ,  y = batch.x , batch.edge_index , batch.edge_attr , batch.batch ,  batch.y
        
        output = self.forward(x , edge_index , edge_attr , batch_idx)
        
        loss = self.loss(output , y)
        acc = self.acc(torch.nn.functional.softmax(output) , y)        
        
        self.log("train_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_idx.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])        
        return loss
    
    def validation_step(self , batch):
        x , edge_index , edge_attr , batch_idx ,  y = batch.x , batch.edge_index , batch.edge_attr , batch.batch ,  batch.y
        
        output = self.forward(x , edge_index , edge_attr , batch_idx)
        
        loss = self.loss(output , y)
        acc = self.acc(torch.nn.functional.softmax(output) , y)        
        auroc = self.auc(torch.nn.functional.softmax(output) , y)
        f1 = self.f1(torch.nn.functional.softmax(output) , y)
        
        self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_idx.shape[0])
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])   
        self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])

class BestModelTracker(Callback):
    
    def __init__(self) -> None:
        self.best_model = None 
        self.best_train_acc = None
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        outputs = trainer.logged_metrics
        
        if self.best_train_acc == None :
            self.best_train_acc = outputs['train_acc'].item() 
        elif self.best_train_acc < outputs['train_acc'].item():
            self.best_train_acc = outputs['train_acc'].item()
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        outputs = trainer.logged_metrics
        
        if self.best_model == None: 
            self.best_model = {
                'best_val_acc': outputs['val_acc'].item(), 
                'best_epoch': trainer.current_epoch, 
                'best_val_auroc': outputs['val_auroc'].item(), 
                'best_val_f1': outputs['val_f1'].item(),
            }
        elif self.best_model['best_val_acc'] < outputs['val_acc'].item() :
            self.best_model = {
                'best_val_acc': outputs['val_acc'].item(), 
                'best_epoch': trainer.current_epoch, 
                'best_val_auroc': outputs['val_auroc'].item(), 
                'best_val_f1': outputs['val_f1'].item(),
            }

  
def main():
    
    parser = argparse.ArgumentParser("GeneGNN")
    parser.add_argument("--lr" , default=0.0001 , type=float , help="Learning rate")
    parser.add_argument("--gene_filter" , default=0.5 , type=float , help="Filter significant gene based on quantile value")
    parser.add_argument("--hidden_embedding" , default=32 , type=int , help="Hidden embedding dimension for convolution and MLP")
    
    args = parser.parse_args()
    
    model = GraphClassification(1 , args.hidden_embedding , 5 , lr=args.lr)
    
    mlflow.set_experiment("basic")
    mlflow.pytorch.autolog(
        log_every_n_epoch=1, 
        log_every_n_step=0
    )
    
    gp1_train , feat1_n , feat1_e , feat1_d = get_omic_graph('1_tr.csv' , '1_featname_conversion.csv' , 'labels_tr.csv' , weighted=False , filter_p_value=0.05 , filter_ppi=300 , significant_q=0)
    # gp2_train , feat2_n , feat2_e , feat2_d = get_omic_graph('2_tr.csv' , '2_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
    # gp3_train , feat3_n , feat3_e , feat3_d = get_omic_graph('3_tr.csv' , '3_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
    gp1_test , _ , _ , _ = get_omic_graph('1_te.csv' , '1_featname_conversion.csv' , 'labels_te.csv' , weighted=False , filter_p_value=0.05 , filter_ppi=300, significant_q=0)
    # gp2_test , _ , _ , _ = get_omic_graph('2_te.csv' , '2_featname_conversion.csv' , 'labels_te.csv')
    # gp3_test , _ , _ , _= get_omic_graph('3_te.csv' , '3_featname_conversion.csv' , 'labels_te.csv')

    train_dataloaders = DataLoader(gp1_train , 30 , True)
    val_dataloaders = DataLoader(gp1_test , 30 , True)
    
    # model tracker 
    modelTracker = BestModelTracker()
    
    trainer = pl.Trainer(
        max_epochs=400, 
        callbacks=[ modelTracker ]
    )
    
    with mlflow.start_run():
        
        for arg in vars(args):
            mlflow.log_param(arg , getattr(args , arg))
            
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloaders, 
            val_dataloaders=val_dataloaders, 
        )
        
        print(modelTracker.best_model , modelTracker.best_train_acc)
        mlflow.log_metrics(modelTracker.best_model)
        mlflow.log_metric('best_train_acc', modelTracker.best_train_acc)
    
    
if __name__ == '__main__':
    main()


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
from torchmetrics import Accuracy , AUROC , F1Score , Specificity , Recall
from torchmetrics.classification import MulticlassConfusionMatrix
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
        
class GraphPooling(torch.nn.Module):
    def __init__(self , in_channels , hidden_channels) -> None: 
        super().__init__()
        
        # Graph Convolution
        self.graph_conv1 = GrapConvolution(in_channels , hidden_channels , hidden_channels)
        
        # Graph Pooling 
        self.pooling = geom_nn.TopKPooling(in_channels=hidden_channels*3 , ratio=0.5) # TopK pooling much stable than SAGPooling
        self.graph_norm1 = geom_nn.GraphNorm(hidden_channels*3)
        # self.pooling = geom_nn.DMoNPooling(channels=hidden_channels*3 , k=100)
        
        # Graph Convolution
        self.graph_conv2 = GrapConvolution(hidden_channels*3 , hidden_channels , hidden_channels)
        
        # Graph Pooling 
        self.pooling2 = geom_nn.TopKPooling(in_channels=hidden_channels*3 , ratio=0.5) # TopK pooling much stable than SAGPooling
        self.graph_norm2 = geom_nn.GraphNorm(hidden_channels*3)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*3 , hidden_channels*4), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*4), 
            torch.nn.Linear(hidden_channels*4 , hidden_channels*2), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*2), 
        )
        
    def forward(self , x , edge_index , edge_attr , batch):
        
        # First layer graph convolution
        x = self.graph_conv1(x , edge_index , edge_attr)
        x , edge_index , _ , batch ,  _ , _ = self.pooling(x , edge_index , edge_attr , batch)
        x = self.graph_norm1(x , batch)
        
        # Second layer graph convolution
        x = self.graph_conv2(x , edge_index)
        x , edge_index , _ , batch ,  _ , _ = self.pooling2(x , edge_index , batch=batch)
        x = self.graph_norm2(x , batch)
        
        x = geom_nn.global_mean_pool(x , batch)

        x = self.mlp(x)
        
        return x
        
class GraphClassification(pl.LightningModule):
    def __init__(self, in_channels , hidden_channels , num_classes , lr=0.0001) -> None:
        super().__init__()
        
        self.lr = lr
        
        # Graph model 
        self.graph = GraphPooling(in_channels , hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels*2 , num_classes)
        
        self.acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        # Sensitivity and specificity 
        self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        self.sensivity = Recall(task="multiclass" , num_classes=num_classes)
        
    def forward(self , x , edge_index , edge_attr , batch):
        # First layer graph convolution
        x = self.graph(x , edge_index , edge_attr)
        x = self.linear(x)
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
        specificity = self.specificity(torch.nn.functional.softmax(output) , y)
        sensivity = self.sensivity(torch.nn.functional.softmax(output) , y)
        
        self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_idx.shape[0])
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])   
        self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        
    def test_step(self , batch):
        x , edge_index , edge_attr , batch_idx ,  y = batch.x , batch.edge_index , batch.edge_attr , batch.batch ,  batch.y
        
        output = self.forward(x , edge_index , edge_attr , batch_idx)
        
        matrix = self.confusion_matrix(output , y)
        self.log("test_confusion_matrix" , matrix.sum(dim=-1) , on_epoch=True)
    
    def predict_step(self , batch):
        x , edge_index , edge_attr , batch_idx ,  y = batch.x , batch.edge_index , batch.edge_attr , batch.batch ,  batch.y
        
        output = self.forward(x , edge_index , edge_attr , batch_idx)
        return output , y
    
class MultiGraphClassification(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr=0.0001) -> None:
        super().__init__() 
        
        self.lr = lr 
        self.graph1 = GraphPooling(in_channels , hidden_channels)
        self.graph2 = GraphPooling(in_channels , hidden_channels)
        self.graph3 = GraphPooling(in_channels , hidden_channels)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*3 , hidden_channels),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Linear(hidden_channels , num_classes),
        )
        
        self.acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        # Sensitivity and specificity 
        self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        self.sensivity = Recall(task="multiclass" , num_classes=num_classes)
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        output1 = self.graph1(x1 , edge_index1 , edge_attr1 , batch1_idx)
        output2 = self.graph2(x2 , edge_index2 , edge_attr2 , batch2_idx)
        output3 = self.graph3(x3 , edge_index3 , edge_attr3 , batch3_idx)
        
        output = torch.stack([output1 , output2 , output3] , dim=-1).mean(dim=-1) # shape -> [ batch , hidden_dimension * 3 ]
        
        output = self.mlp(output)
        return output
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters() , lr= self.lr , weight_decay=0.0001)
    
    def training_step(self , batch):
        batch1 , batch2 , batch3 = batch 
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        loss = self.loss(output , y1)
        acc = self.acc(torch.nn.functional.softmax(output) , y1)
        
        self.log("train_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        return loss
    
    def validation_setp(self , batch):
        batch1 , batch2 , batch3 = batch 
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        loss = self.loss(output , y1)
        acc = self.acc(torch.nn.functional.softmax(output) , y1)
        f1 = self.f1(torch.nn.functional.softmax(output) , y1)
        auroc = self.auc(torch.nn.functional.softmax(output) , y1)
        specificity = self.specificity(torch.nn.functional.softmax(output) , y1)
        sensivity = self.sensivity(torch.nn.functional.softmax(output) , y1)
        
        self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
    
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


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    view_edge = torch.ones(( len(data_list) , 3 , 3 ))
    return batchA, batchB , batchC , view_edge
    
def multiomics(args):
    
    gp_train_x1 , train_avg_node_per_graph_x1 , _ , train_avg_nodedegree_x1 , train_avg_isolate_node_per_graph_x1 = get_omic_graph('1_tr.csv' , '1_featname_conversion.csv' , 'ac_rule_1.tsv' , 'labels_tr.csv' , weighted=False , filter_p_value=None , filter_ppi=None , significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    gp_train_x2 , train_avg_node_per_graph_x2 , _ , train_avg_nodedegree_x2 , train_avg_isolate_node_per_graph_x2 = get_omic_graph('2_tr.csv' , '2_featname_conversion.csv' , 'ac_rule_2.tsv' , 'labels_tr.csv' , weighted=False , filter_p_value=None , filter_ppi=None , significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    gp_train_x3 , train_avg_node_per_graph_x3 , _ , train_avg_nodedegree_x3 , train_avg_isolate_node_per_graph_x3 = get_omic_graph('3_tr.csv' , '3_featname_conversion.csv' , 'ac_rule_3.tsv' , 'labels_tr.csv' , weighted=False , filter_p_value=None , filter_ppi=None , significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    
    gp_test_x1 , test_avg_node_per_graph_x1 , _ , test_avg_nodedegree_x1 , test_avg_isolate_node_per_graph_x1 = get_omic_graph('1_te.csv' , '1_featname_conversion.csv' , 'ac_rule_1.tsv' , 'labels_te.csv' , weighted=False , filter_p_value=None , filter_ppi=None, significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    gp_test_x2 , test_avg_node_per_graph_x2 , _ , test_avg_nodedegree_x2 , test_avg_isolate_node_per_graph_x2 = get_omic_graph('2_te.csv' , '2_featname_conversion.csv' , 'ac_rule_2.tsv' , 'labels_te.csv' , weighted=False , filter_p_value=None , filter_ppi=None, significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    gp_test_x3 , test_avg_node_per_graph_x3 , _ , test_avg_nodedegree_x3 , test_avg_isolate_node_per_graph_x3 = get_omic_graph('3_te.csv' , '3_featname_conversion.csv' , 'ac_rule_3.tsv' , 'labels_te.csv' , weighted=False , filter_p_value=None , filter_ppi=None, significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    
    feature_info  = {
        'train_avg_node_x1': train_avg_node_per_graph_x1 ,
        #'train_avg_edge_x1': train_avg_edge_per_graph_x1 ,
        'train_avg_degree_x1': train_avg_nodedegree_x1 ,
        'train_avg_isolated_x1': train_avg_isolate_node_per_graph_x1 ,
        'train_avg_node_x2': train_avg_node_per_graph_x2 ,
        #'train_avg_edge_x2': train_avg_edge_per_graph_x2 ,
        'train_avg_degree_x2': train_avg_nodedegree_x2 ,
        'train_avg_isolated_x2': train_avg_isolate_node_per_graph_x2 ,
        'train_avg_node_x3': train_avg_node_per_graph_x3 ,
        #'train_avg_edge_x3': train_avg_edge_per_graph_x3 ,
        'train_avg_degree_x3': train_avg_nodedegree_x3 ,
        'train_avg_isolated_x3': train_avg_isolate_node_per_graph_x3 ,
        'test_avg_node_x1': test_avg_node_per_graph_x1 ,
        #'test_avg_edge_x1': test_avg_edge_per_graph_x1 ,
        'test_avg_degree_x1': test_avg_nodedegree_x1 ,
        'test_avg_isolated_x1': test_avg_isolate_node_per_graph_x1 ,
        'test_avg_node_x2': test_avg_node_per_graph_x2 ,
        #'test_avg_edge_x2': test_avg_edge_per_graph_x2 ,
        'test_avg_degree_x2': test_avg_nodedegree_x2 ,
        'test_avg_isolated_x2': test_avg_isolate_node_per_graph_x2 ,
        'test_avg_node_x3': test_avg_node_per_graph_x3 ,
        #'test_avg_edge_x3': test_avg_edge_per_graph_x3 ,
        'test_avg_degree_x3': test_avg_nodedegree_x3 ,
        'test_avg_isolated_x3': test_avg_isolate_node_per_graph_x3 ,
    }
    batch_size = 20 
    
    pair_dataset_tr = PairDataset(gp_train_x1 , gp_train_x2 , gp_train_x3)
    dataloader_tr = torch.utils.data.DataLoader(pair_dataset_tr, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    
    
    pair_dataset_te = PairDataset(gp_test_x1 , gp_test_x2 , gp_test_x3)
    dataloader_te = torch.utils.data.DataLoader(pair_dataset_te, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)
    
    # model tracker 
    modelTracker = BestModelTracker()
    
    trainer = pl.Trainer(
        max_epochs=args.max_epoch, 
        callbacks=[ modelTracker ]
    )
    
    model = MultiGraphClassification(1 , args.hidden_embedding , 5 , lr=args.lr)
    
    if not args.disable_tracking:
        with mlflow.start_run() as run:
            pass
            # for arg in vars(args):
            #     mlflow.log_param(arg , getattr(args , arg))
            
            # mlflow.log_params(train_feature)
            # mlflow.log_params(test_feature)
                
            # trainer.fit(
            #     model=model, 
            #     train_dataloaders=train_dataloaders, 
            #     val_dataloaders=val_dataloaders, 
            # )
            
            # print(modelTracker.best_model , modelTracker.best_train_acc)
            # mlflow.log_metrics(modelTracker.best_model)
            # mlflow.log_metric('best_train_acc', modelTracker.best_train_acc)
            
            # prediction = trainer.predict(model , val_dataloaders)
            # output = torch.concat([ x[0] for x in prediction ])
            # actual = torch.concat([ x[1] for x in prediction ])
            
            # confusion_matrix = MulticlassConfusionMatrix(num_classes=5)
            # confusion_matrix.update(output, actual)
            # #cfm = confusion_matrix(output, actual)
            # #print(cfm)
            
            # fig , ax  = confusion_matrix.plot()
            # #ax.set_fontsize(fs=20)
            # #fig.set_title("Multiclass Confusion Matrix")
            # fig.savefig(f"confusion-matrix-{run.info.run_name}.png")
    else: 
        trainer.fit(
            model=model, 
            train_dataloaders=dataloader_tr, 
            val_dataloaders=dataloader_te, 
        )
            
        # prediction = trainer.predict(model , dataloader_te)
        # output = torch.concat([ x[0] for x in prediction ])
        # actual = torch.concat([ x[1] for x in prediction ])
        
        # confusion_matrix = MulticlassConfusionMatrix(num_classes=5)
        # confusion_matrix.update(output, actual)
        
        # fig , ax  = confusion_matrix.plot()
        # fig.savefig("confusion_matrix.png")
    
def main():
    
    parser = argparse.ArgumentParser("GeneGNN")
    parser.add_argument("--lr" , default=0.0001 , type=float , help="Learning rate")
    parser.add_argument("--gene_filter" , default=0.5 , type=float , help="Filter significant gene based on quantile value")
    parser.add_argument("--hidden_embedding" , default=32 , type=int , help="Hidden embedding dimension for convolution and MLP")
    parser.add_argument("--dataset" , type=str , choices=['miRNA' , 'mRNA' , 'DNA'] , default='mRNA')
    parser.add_argument("--enrichment" , action="store_true")
    parser.add_argument("--noppi" , action="store_true")
    parser.add_argument("--nokegg" , action="store_true")
    parser.add_argument("--corr" , action="store_true")
    parser.add_argument("--topk" , type=int , default=50 )
    parser.add_argument("--max_epoch" , type=int , default=400)
    parser.add_argument("--disable_tracking" , action='store_true')
    parser.add_argument("--multiomics" , action='store_true')
    parser.add_argument("--experiment" , type=str , default="basic" , help="MLFlow expriement name")
    
    args = parser.parse_args()
    
    model = GraphClassification(1 , args.hidden_embedding , 5 , lr=args.lr)
    
    mlflow.set_experiment(args.experiment)
    mlflow.pytorch.autolog(
        log_every_n_epoch=1, 
        log_every_n_step=0
    )
    
    if args.multiomics:
        multiomics(args)
        return
    
    if args.dataset == 'mRNA':
        train_datapath = '1_tr.csv'
        test_datapath = '1_te.csv'
        conversionpath = '1_featname_conversion.csv'
        ac_datapath = 'ac_rule_1.tsv'
    elif args.dataset == 'miRNA':
        train_datapath = '2_tr.csv'
        test_datapath = '2_te.csv'
        conversionpath = '2_featname_conversion.csv'
        ac_datapath = 'ac_rule_2.tsv'
    elif args.dataset == 'DNA':
        train_datapath = '3_tr.csv'
        test_datapath = '3_te.csv'
        conversionpath = '3_featname_conversion.csv'
        ac_datapath = 'ac_rule_3.tsv'
    
    gp_train , train_avg_node_per_graph , train_avg_edge_per_graph , train_avg_nodedegree , train_avg_isolate_node_per_graph = get_omic_graph(train_datapath , conversionpath , ac_datapath , 'labels_tr.csv' , weighted=False , filter_p_value=None , filter_ppi=None , significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    # gp2_train , feat2_n , feat2_e , feat2_d = get_omic_graph('2_tr.csv' , '2_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
    # gp3_train , feat3_n , feat3_e , feat3_d = get_omic_graph('3_tr.csv' , '3_featname_conversion.csv' , 'labels_tr.csv' , weighted=args.weight , filter_p_value=args.filter_p_value , filter_ppi=args.filter_ppi)
    gp_test , test_avg_node_per_graph , test_avg_edge_per_graph , test_avg_nodedegree , test_avg_isolate_node_per_graph = get_omic_graph(test_datapath , conversionpath , ac_datapath , 'labels_te.csv' , weighted=False , filter_p_value=None , filter_ppi=None, significant_q=0 , ac=args.enrichment , k=args.topk , go_kegg=(not args.nokegg) , ppi=(not args.noppi) , correlation=(args.corr))
    # gp2_test , _ , _ , _ = get_omic_graph('2_te.csv' , '2_featname_conversion.csv' , 'labels_te.csv')
    # gp3_test , _ , _ , _= get_omic_graph('3_te.csv' , '3_featname_conversion.csv' , 'labels_te.csv')

    train_feature = { 
        "train_avg_node": train_avg_node_per_graph, 
        "train_avg_edge": train_avg_edge_per_graph, 
        "train_avg_degree": train_avg_nodedegree,
        "train_avg_isolated": train_avg_isolate_node_per_graph                 
    }
    
    test_feature = { 
        "test_avg_node": test_avg_node_per_graph, 
        "test_avg_edge": test_avg_edge_per_graph, 
        "test_avg_degree": test_avg_nodedegree,
        "test_avg_isolated": test_avg_isolate_node_per_graph                 
    }
    
    print("Train feature info: " , train_feature)
    print("Test feature info: " , test_feature)
    train_dataloaders = DataLoader(gp_train , 30 , True)
    val_dataloaders = DataLoader(gp_test , 30 , True)
    
    # model tracker 
    modelTracker = BestModelTracker()
    
    trainer = pl.Trainer(
        max_epochs=args.max_epoch, 
        callbacks=[ modelTracker ]
    )
    
    if not args.disable_tracking:
        with mlflow.start_run() as run:
            
            for arg in vars(args):
                mlflow.log_param(arg , getattr(args , arg))
            
            mlflow.log_params(train_feature)
            mlflow.log_params(test_feature)
                
            trainer.fit(
                model=model, 
                train_dataloaders=train_dataloaders, 
                val_dataloaders=val_dataloaders, 
            )
            
            print(modelTracker.best_model , modelTracker.best_train_acc)
            mlflow.log_metrics(modelTracker.best_model)
            mlflow.log_metric('best_train_acc', modelTracker.best_train_acc)
            
            prediction = trainer.predict(model , val_dataloaders)
            output = torch.concat([ x[0] for x in prediction ])
            actual = torch.concat([ x[1] for x in prediction ])
            
            confusion_matrix = MulticlassConfusionMatrix(num_classes=5)
            confusion_matrix.update(output, actual)
            #cfm = confusion_matrix(output, actual)
            #print(cfm)
            
            fig , ax  = confusion_matrix.plot()
            #ax.set_fontsize(fs=20)
            #fig.set_title("Multiclass Confusion Matrix")
            fig.savefig(f"confusion-matrix-{run.info.run_name}.png")
    else: 
        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloaders, 
            val_dataloaders=val_dataloaders, 
        )
            
        # test = trainer.test(model , val_dataloaders)
        # print(test)
        prediction = trainer.predict(model , val_dataloaders)
        output = torch.concat([ x[0] for x in prediction ])
        actual = torch.concat([ x[1] for x in prediction ])
        
        confusion_matrix = MulticlassConfusionMatrix(num_classes=5)
        confusion_matrix.update(output, actual)
        #cfm = confusion_matrix(output, actual)
        #print(cfm)
        
        fig , ax  = confusion_matrix.plot()
        #ax.set_fontsize(fs=20)
        #fig.set_title("Multiclass Confusion Matrix")
        fig.savefig("confusion_matrix.png")
        
        # output = torch.concat(prediction[:,0] , dim=-1)
        # print(output)
    
    
if __name__ == '__main__':
    main()


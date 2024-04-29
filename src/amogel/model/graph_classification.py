import mlflow
import torch
import pytorch_lightning as pl 
import torch_geometric.nn as geom_nn
import torch_geometric.utils as geom_utils
from torchmetrics import Accuracy , AUROC , F1Score , Specificity , Recall , ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix
from lightning.pytorch.utilities.types import  OptimizerLRScheduler
from torch import optim
from amogel import logger

class GraphConvolution(torch.nn.Module):
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
        
    def forward(self , x , edge_index , edge_attr = None , batch_norm = True ): 
        
        if edge_attr is not None:
            x1 , x1_edge_attr = self.graph_conv1(x , edge_index , edge_attr , return_attention_weights=True)
            x1 = x1.relu() # batch
            if batch_norm:
                x1 = self.batch_norm1(x1)
            x2 , x2_edge_attr = self.graph_conv2(x1 , edge_index , edge_attr , return_attention_weights=True)
            x2 - x2.relu()
            if batch_norm:
                x2 = self.batch_norm2(x2)
            x3 , x3_edge_attr = self.graph_conv3(x2 , edge_index , edge_attr , return_attention_weights=True)
            x3 = x3.relu()
            if batch_norm:
                x3 = self.batch_norm3(x3)
        else: 
            x1 , x1_edge_attr = self.graph_conv1(x , edge_index , return_attention_weights=True)
            x1 = x1.relu() # batch
            if batch_norm:
                x1 = self.batch_norm1(x1)
            x2 , x2_edge_attr = self.graph_conv2(x1 , edge_index , return_attention_weights=True)
            x2 = x2.relu()
            if batch_norm:
                x2 = self.batch_norm2(x2)
            x3 , x3_edge_attr = self.graph_conv3(x2 , edge_index , return_attention_weights=True)
            x3 = x3.relu()
            if batch_norm:
                x3 = self.batch_norm3(x3)
            
        # Jumping knowledge 
        # x = torch.stack([x1 , x2 , x3], dim=-1).mean(dim=-1)
        if self.jump:
            x = torch.concat([x1,x2,x3] , dim=-1)
            return x , x1_edge_attr , x2_edge_attr , x3_edge_attr
        else: 
            return x3 , x1_edge_attr , x2_edge_attr , x3_edge_attr
        
class GraphPooling(torch.nn.Module):
    def __init__(self , in_channels , hidden_channels) -> None: 
        super().__init__()
        
        # Graph Convolution
        self.graph_conv1 = GraphConvolution(in_channels , hidden_channels , hidden_channels)
        
        # Graph Pooling 
        self.pooling = geom_nn.TopKPooling(in_channels=hidden_channels*3 , ratio=0.5) # TopK pooling much stable than SAGPooling
        self.graph_norm1 = geom_nn.GraphNorm(hidden_channels*3)
        # self.pooling = geom_nn.DMoNPooling(channels=hidden_channels*3 , k=100)
        
        # Graph Convolution
        self.graph_conv2 = GraphConvolution(hidden_channels*3 , hidden_channels , hidden_channels)
        
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
        
    def forward(self , x , edge_index , edge_attr , batch , batch_idx = None ,  log=False):
        
        # First layer graph convolution
        x , gc1_k1_edge_attr , gc1_k2_edge_attr , gc1_k3_edge_attr = self.graph_conv1(x , edge_index , edge_attr)
        #print(x)
        x , edge_index , _ , batch1 ,  perm_1 , score_1 = self.pooling(x , edge_index , edge_attr , batch)
        
        x = self.graph_norm1(x , batch1)
        
        # Second layer graph convolution
        x , gc2_k1_edge_attr , gc2_k2_edge_attr , gc2_k3_edge_attr  = self.graph_conv2(x , edge_index)
        x , edge_index , _ , batch2 ,  perm_2 , score_2 = self.pooling2(x , edge_index , batch=batch1)
        x = self.graph_norm2(x , batch2)
        
        x = geom_nn.global_mean_pool(x , batch2)

        
         
        gc1_k1_edge_attr_dense = geom_utils.to_dense_adj(gc1_k1_edge_attr[0] , batch  , edge_attr=gc1_k1_edge_attr[1])
        gc1_k2_edge_attr_dense = geom_utils.to_dense_adj(gc1_k2_edge_attr[0] , batch  , edge_attr=gc1_k2_edge_attr[1])
        gc1_k3_edge_attr_dense = geom_utils.to_dense_adj(gc1_k3_edge_attr[0] , batch  , edge_attr=gc1_k3_edge_attr[1])
        gc1_dense = torch.stack([gc1_k1_edge_attr_dense , gc1_k2_edge_attr_dense , gc1_k3_edge_attr_dense] , dim=-1).mean(dim=-1)
        fea_attr_scr1 = gc1_dense.squeeze(dim=-1).mean(dim=0)
        
        if log: 
            pass
            #print(gc1_dense.shape)
            #print(fea_attr_scr1.shape)
        gc2_k1_edge_attr_dense = geom_utils.to_dense_adj(gc2_k1_edge_attr[0] , batch1 , edge_attr=gc2_k1_edge_attr[1])
        gc2_k2_edge_attr_dense = geom_utils.to_dense_adj(gc2_k2_edge_attr[0] , batch1 , edge_attr=gc2_k2_edge_attr[1])
        gc2_k3_edge_attr_dense = geom_utils.to_dense_adj(gc2_k3_edge_attr[0] , batch1 , edge_attr=gc2_k3_edge_attr[1])
        gc2_dense = torch.stack([gc2_k1_edge_attr_dense , gc2_k2_edge_attr_dense , gc2_k3_edge_attr_dense] , dim=-1).mean(dim=-1)
        fea_attr_scr2 = gc2_dense.squeeze(dim=-1).mean(dim=0)
        
        x = self.mlp(x)
        
        return x , perm_1 , perm_2 , score_1 , score_2 , batch1 , batch2 , fea_attr_scr1 , fea_attr_scr2
    
class GraphClassification(pl.LightningModule):
    def __init__(self, in_channels , hidden_channels , num_classes , lr=0.0001 , drop_out = 0.1 , mlflow:mlflow = None) -> None:
        super().__init__()
        
        self.lr = lr
        self.mlflow = mlflow
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
        self.sensivity = Recall(task="multiclass" , num_classes=num_classes , average="macro")
        
    def forward(self , x , edge_index , edge_attr , batch):
        # First layer graph convolution
        x , perm1 , perm2 , score1 , score2 , batch1 , batch2 , fea_attr_scr1 , fea_att_scr2 = self.graph(x , edge_index , edge_attr , batch)
        x = self.linear(x)
        return x 
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters() , lr= self.lr , weight_decay=0.0001) 
    
    def training_step(self , batch):
        x , edge_index , edge_attr , batch_idx ,  y = batch.x , batch.edge_index , batch.edge_attr , batch.batch ,  batch.y
        
        output = self.forward(x , edge_index , edge_attr , batch_idx)
        
        loss = self.loss(output , y)
        acc = self.acc(torch.nn.functional.softmax(output) , y)   
        self.confusion_matrix.update(output , y)     
        
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
        self.confusion_matrix.update(output , y)    
        
        self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_idx.shape[0])
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])   
        self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
        self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_idx.shape[0])
    
    def on_test_epoch_end(self) -> None:
        if self.current_epoch+1 % 10 == 0:
            if self.mlflow is not None:
                # calculate confusion matrix
                fig , ax = self.confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "training_confusion_matrix_epoch_{}".format(self.current_epoch))
                
    def on_train_epoch_end(self) -> None: 
        if self.current_epoch+1 % 10 == 0:
            if self.mlflow is not None:
                # calculate confusion matrix
                fig , ax = self.confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "training_confusion_matrix_epoch_{}".format(self.current_epoch))
    
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
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr=0.0001 , drop_out = 0.1 , mlflow:mlflow = None , multi_graph_testing=False) -> None:
        super().__init__() 
        
        self.lr = lr 
        self.mlflow = mlflow
        self.multi_graph_testing = multi_graph_testing
        self.num_classes = num_classes
        
        self.graph1 = GraphPooling(in_channels , hidden_channels)
        self.graph2 = GraphPooling(in_channels , hidden_channels)
        self.graph3 = GraphPooling(in_channels , hidden_channels)
        
        self.rank = {}
        self.allrank = {
            'omic1': [] ,
            'omic2': [] ,
            'omic3': [] ,
        }
        self.genes = {
            'omic1_pool1': [],
            'omic1_pool2': [],  
            'omic2_pool1': [], 
            'omic2_pool2': [], 
            'omic3_pool1': [],
            'omic3_pool2': [],
        }
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*3*2 , hidden_channels),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Linear(hidden_channels , num_classes),
        )
        
        self.acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        # Sensitivity and specificity 
        self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        self.sensivity = Recall(task="multiclass" , num_classes=num_classes , average="macro")
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        output1 , perm11 , perm12 , score11 , score12 , batch11 , batch12 , attr_score11 , attr_score12 = self.graph1(x1 , edge_index1 , edge_attr1 , batch1_idx , log=True)
        output2 , perm21 , perm22 , score21 , score22 , batch21 , batch22 , attr_score21 , attr_score22 = self.graph2(x2 , edge_index2 , edge_attr2 , batch2_idx)
        output3 , perm31 , perm32 , score31 , score32 , batch31 , batch32 , attr_score31 , attr_score32 = self.graph3(x3 , edge_index3 , edge_attr3 , batch3_idx)
        
        output = torch.concat([output1 , output2 , output3] , dim=-1) # shape -> [ batch , hidden_dimension * 3 * 2 ]
        
        output = self.mlp(output)
        
        self.rank = {
            'omic1': ( perm11 , perm12 , score11 , score12 , batch11 , batch12 , attr_score11 , attr_score12 ) ,
            'omic2': ( perm21 , perm22 , score21 , score22 , batch21 , batch22 , attr_score21 , attr_score22 ) ,
            'omic3': ( perm31 , perm32 , score31 , score32 , batch31 , batch32 , attr_score31 , attr_score32 ) ,
        }
        
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
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1) , y1)
        self.train_confusion_matrix.update(output , y1)  
        
        self.log("train_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        return loss
    
    def get_rank_genes(self , pooling_info , batch_extra_label , batch_size , num_genes=1000):
        perm_1 , perm2 , score1 , score2 , pollbatch1 , poolbatch2 , _ , _ = pooling_info
        
        #gene_perm1 = batch_extra_label[perm_1].view(batch_size , -1 ) # convert to batch size , gene size ( 0.5 ratio )
        gene_perm1 , mask_pool_1 = geom_utils.to_dense_batch(batch_extra_label[perm_1] , pollbatch1)
        #gene_score1 = score1.view(batch_size , -1) # convert to batch size , gene size ( 0.5 ratio )
        gene_score1 , _ = geom_utils.to_dense_batch(score1 , pollbatch1)
        
        # gene_perm2 = batch_extra_label[perm_1][perm2].view(batch_size , -1) # convert to batch size , gene size (0.5*0.5 ratio)
        gene_perm2 , mask_pool_2 = geom_utils.to_dense_batch(batch_extra_label[perm_1][perm2] , poolbatch2)
        # gene_score2 = score2.view(batch_size , -1) # convert to batch size , gene size (0.5*0.5 ratio)
        gene_score2 , _ = geom_utils.to_dense_batch(score2 , poolbatch2)
        
        pool1 = []
        pool2 = []
        
        for i in range(batch_size):
            _genes1_pool1 = torch.zeros(num_genes)
            _genes1_pool1[gene_perm1[i][mask_pool_1[i]].cpu()] = gene_score1[i][mask_pool_1[i]].cpu()
            pool1.append(_genes1_pool1)
            
            _genes1_pool2 = torch.zeros(num_genes)
            _genes1_pool2[gene_perm2[i][mask_pool_2[i]].cpu()] = gene_score2[i][mask_pool_2[i]].cpu()
            pool2.append(_genes1_pool2)
            
        return pool1 , pool2 
        
    def validation_step(self , batch):
        batch1 , batch2 , batch3 = batch 
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        loss = self.loss(output , y1)
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1) , y1)
        f1 = self.f1(torch.nn.functional.softmax(output , dim=-1) , y1)
        auroc = self.auc(torch.nn.functional.softmax(output , dim=-1) , y1)
        specificity = self.specificity(torch.nn.functional.softmax(output , dim=-1) , y1)
        sensivity = self.sensivity(torch.nn.functional.softmax(output , dim=-1) , y1)
        self.test_confusion_matrix.update(output , y1)  
        
        # if self.current_epoch == self.trainer.max_epochs - 1:
            
        #     omic1_pool1 , omic1_pool2 = self.get_rank_genes(self.rank['omic1'] , batch1.extra_label , batch1.num_graphs , 1000)
        #     omic2_pool1 , omic2_pool2 = self.get_rank_genes(self.rank['omic2'] , batch2.extra_label , batch2.num_graphs , 1000)
        #     omic3_pool1 , omic3_pool2 = self.get_rank_genes(self.rank['omic3'] , batch3.extra_label , batch3.num_graphs , 503)
            
        #     self.genes['omic1_pool1'].extend(omic1_pool1)
        #     self.genes['omic1_pool2'].extend(omic1_pool2)
        #     self.genes['omic2_pool1'].extend(omic2_pool1)
        #     self.genes['omic2_pool2'].extend(omic2_pool2)
        #     self.genes['omic3_pool1'].extend(omic3_pool1)
        #     self.genes['omic3_pool2'].extend(omic3_pool2)
            
        self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
    
    def on_validation_epoch_end(self) -> None: 
        
        if self.current_epoch % 10 == 0:
            
            if self.mlflow is not None:
                logger.info("Logging confusion matrix for test epoch {}".format(self.current_epoch))
                # calculate confusion matrix

                self.test_confusion_matrix.compute()
                fig , ax = self.test_confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "test_confusion_matrix_epoch_{}.png".format(self.current_epoch))
        
        self.test_confusion_matrix.reset()
    
    def on_train_epoch_end(self) -> None:
        
        if self.current_epoch % 10 == 0:
            if self.mlflow is not None:
                logger.info("Logging confusion matrix for training epoch {}".format(self.current_epoch))
                # calculate confusion matrix
                self.train_confusion_matrix.compute()
                fig , ax = self.train_confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "train_confusion_matrix_epoch_{}.png".format(self.current_epoch))
        
        self.train_confusion_matrix.reset()
    
    def single_graph_test(self, batch):
        batch1 , batch2 , batch3 = batch 
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        loss = self.loss(output , y1)
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1) , y1)
        f1 = self.f1(torch.nn.functional.softmax(output , dim=-1) , y1)
        auroc = self.auc(torch.nn.functional.softmax(output , dim=-1) , y1)
        specificity = self.specificity(torch.nn.functional.softmax(output , dim=-1) , y1)
        sensivity = self.sensivity(torch.nn.functional.softmax(output , dim=-1) , y1)
        
        omic1_pool1 , omic1_pool2 = self.get_rank_genes(self.rank['omic1'] , batch1.extra_label , batch1.num_graphs , 1000)
        omic2_pool1 , omic2_pool2 = self.get_rank_genes(self.rank['omic2'] , batch2.extra_label , batch2.num_graphs , 1000)
        omic3_pool1 , omic3_pool2 = self.get_rank_genes(self.rank['omic3'] , batch3.extra_label , batch3.num_graphs , 503)
        
        self.genes['omic1_pool1'].extend(omic1_pool1)
        self.genes['omic1_pool2'].extend(omic1_pool2)
        self.genes['omic2_pool1'].extend(omic2_pool1)
        self.genes['omic2_pool2'].extend(omic2_pool2)
        self.genes['omic3_pool1'].extend(omic3_pool1)
        self.genes['omic3_pool2'].extend(omic3_pool2)
        
        self.allrank['omic1'].append(self.rank['omic1'])
        self.allrank['omic2'].append(self.rank['omic2'])
        self.allrank['omic3'].append(self.rank['omic3'])
        
        self.log("test_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("test_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        
    def multi_graph_test(self , batch):
        batch1 , batch2 , batch3 = batch # batch1 , batch2 , batch3 contains multiple topology of the same omic type
        
        results = []
        for i in range(self.num_classes):
            x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
            x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
            x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
            
            output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
            # loss = self.loss(output , y1)
            output_softmax = torch.nn.functional.softmax(output , dim=-1)
            predicted_class = output_softmax.argmax(dim=-1)
            predicted_prob = output_softmax.max(dim=-1)
            
            with open("multigraph_testing_logs.txt" , "a") as log_file: 
                log_file.write(f"Epoch: {self.current_epoch}\t| Topology: {i}\t| Predicted class: {predicted_class}\t| Predicted probability: {predicted_prob}\t| Actual class: {y1}\n")
            results.append({'topology': i , 'predicted_class': predicted_class , 'predicted_prob': predicted_prob , 'output': output })
        
        
        # get the top predicted probability
        final_prediction = max(results, key=lambda x: x['predicted_prob'])
        print(f"Final prediction: {final_prediction['predicted_class']} | Actual class: {y1}") 
        with open("multigraph_testing_logs.txt" , "a") as log_file: 
            log_file.write(f"Epoch: {self.current_epoch}\t| Final prediction: {final_prediction} | Actual class: {y1}\n")
            log_file.write("\n")
            
        acc = self.acc(torch.nn.functional.softmax(final_prediction['output'] , dim=-1)  , y1)
        self.log('test_acc' , acc , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
    
    def test_step(self , batch): 
        if not self.multi_graph_testing:
            self.single_graph_test(batch)
        else:
            self.multi_graph_test(batch)
        
    def predict_step(self , batch):
        batch1 , batch2 , batch3 = batch 
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        return output , y1
    
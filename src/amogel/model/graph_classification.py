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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , roc_auc_score 
import math

class GraphConvolution(torch.nn.Module):
    def __init__(self , in_channels , hidden_channels , out_channels , jump = True, *args, **kwargs):
        super().__init__() 
        
        multihead = kwargs.get('multihead' , 1)
        concat = kwargs.get('multihead_concat' , False)
        dropout = kwargs.get("gat_dropout" , .0)
        num_layer = kwargs.get("num_layer" , 3)
        
        middle_channels = hidden_channels if not concat else hidden_channels * multihead
        mid_out_channels = out_channels if not concat else out_channels * multihead
        
        self.layers = []
        for i in range(num_layer):
            self.layers.append(
                torch.nn.ModuleList([
                    geom_nn.GATConv(
                        in_channels=in_channels if i == 0 else middle_channels , 
                        out_channels=hidden_channels if i != num_layer-1 else out_channels, 
                        heads=multihead , 
                        concat=concat , 
                        dropout=dropout),
                    geom_nn.BatchNorm(in_channels=middle_channels if i != num_layer-1 else mid_out_channels)
                ])
            )
        self.layers = torch.nn.ModuleList(self.layers)
        # Graph Convolution
        # self.graph_conv1 = geom_nn.GATConv(in_channels=in_channels , out_channels=hidden_channels, heads=multihead , concat=concat , dropout=dropout)
        # self.graph_conv2 = geom_nn.GATConv(in_channels=middle_channels , out_channels=hidden_channels, heads=multihead, concat=concat , dropout=dropout)
        # self.graph_conv3 = geom_nn.GATConv(in_channels=middle_channels , out_channels=out_channels, heads=multihead, concat=concat , dropout=dropout)
        # self.batch_norm1 = geom_nn.BatchNorm(in_channels=middle_channels)
        # self.batch_norm2 = geom_nn.BatchNorm(in_channels=middle_channels)
        # self.batch_norm3 = geom_nn.BatchNorm(in_channels=mid_out_channels)
        
        self.jump = jump
        
    def forward(self , x , edge_index , edge_attr = None , batch_norm = True ): 
        _x , _edges = [], []
        if edge_attr is not None:
            
            for gat_layer , batch_layer in self.layers:
                # print(f"{gat_layer} - {batch_layer}")
                # print(f"{x.device} - {edge_index.device} - {edge_attr.device}")
                # print(f"{x.shape} - {edge_index.shape} - {edge_attr.shape}")
                x , edge = gat_layer(x , edge_index , edge_attr , return_attention_weights=True)
                x = x.relu()
                if batch_norm:
                    x = batch_layer(x)
                _x.append(x)
                _edges.append(edge)
            # x1 , x1_edge_attr = self.graph_conv1(x , edge_index , edge_attr , return_attention_weights=True)
            # x1 = x1.relu() # batch
            # if batch_norm:
            #     x1 = self.batch_norm1(x1)
            # x2 , x2_edge_attr = self.graph_conv2(x1 , edge_index , edge_attr , return_attention_weights=True)
            # x2 - x2.relu()
            # if batch_norm:
            #     x2 = self.batch_norm2(x2)
            # x3 , x3_edge_attr = self.graph_conv3(x2 , edge_index , edge_attr , return_attention_weights=True)
            # x3 = x3.relu()
            # if batch_norm:
            #     x3 = self.batch_norm3(x3)
        else: 
            for gat_layer , batch_layer in self.layers:
                x , edge = gat_layer(x , edge_index , return_attention_weights=True)
                x = x.relu()
                if batch_norm:
                    x = batch_layer(x)
                _x.append(x)
                _edges.append(edge)
            
        # Jumping knowledge 
        # x = torch.stack([x1 , x2 , x3], dim=-1).mean(dim=-1)
        if self.jump:
            x = torch.concat(_x , dim=-1)
            return x , _edges
        else: 
            return x[-1] , _edges # get only layer layer output
        
POOL = {
    "sag" : geom_nn.SAGPooling, 
    "topk" : geom_nn.TopKPooling
}
class GraphPooling(torch.nn.Module):
    def __init__(self , in_channels , hidden_channels , *args, **kwargs) -> None: 
        super().__init__()
        
        multihead = kwargs.get('multihead' , 1)
        concat = kwargs.get('multihead_concat' , False)
        self.num_block = kwargs.get('num_block' , 2)
        pooling_rate = kwargs.get('pooling_rate' , 0.5)
        pooling = kwargs.get('pooling' , 'sag')
        conv_layer = kwargs.get("num_layer" , 3)
        
        assert pooling in ['sag' , 'topk'] , "Invalid pooling method"
        
        if multihead > 1 and concat : 
            graph_conv_output = hidden_channels * multihead * conv_layer
        else: 
            graph_conv_output = hidden_channels * conv_layer
        
        self.block_layers = []
        for i in range(self.num_block):
            self.block_layers.append(
                torch.nn.ModuleList([
                    GraphConvolution(in_channels if i == 0 else graph_conv_output , hidden_channels , hidden_channels , **kwargs),
                    POOL[pooling](in_channels=graph_conv_output , ratio=pooling_rate),
                    geom_nn.GraphNorm(graph_conv_output)
                ])
            )
        self.block_layers = torch.nn.ModuleList(self.block_layers)
        # for i in range(self.num_block):
        #     setattr(self , f'graph_conv{i}' , GraphConvolution(in_channels if i == 0 else graph_conv_output , hidden_channels , hidden_channels , **kwargs))
        #     setattr(self , f'pooling{i}' , POOL[pooling](in_channels=graph_conv_output , ratio=pooling_rate))
        #     setattr(self , f'graph_norm{i}' , geom_nn.GraphNorm(graph_conv_output))
            
        # # Graph Convolution
        # self.graph_conv1 = GraphConvolution(in_channels , hidden_channels , hidden_channels , **kwargs)
        
        # # Graph Pooling 
        # #self.pooling = geom_nn.TopKPooling(in_channels=graph_conv_output , ratio=0.5) # TopK pooling much stable than SAGPooling
        # self.pooling = geom_nn.SAGPooling(in_channels=graph_conv_output , ratio=0.5)
        # self.graph_norm1 = geom_nn.GraphNorm(graph_conv_output)
        # # self.pooling = geom_nn.DMoNPooling(channels=hidden_channels*3 , k=100)
        
        # # Graph Convolution
        # self.graph_conv2 = GraphConvolution(graph_conv_output , hidden_channels , hidden_channels, **kwargs)
        
        # # Graph Pooling 
        # #self.pooling2 = geom_nn.TopKPooling(in_channels=graph_conv_output , ratio=0.5) # TopK pooling much stable than SAGPooling
        # self.pooling2 = geom_nn.SAGPooling(in_channels=graph_conv_output , ratio=0.5)
        # self.graph_norm2 = geom_nn.GraphNorm(graph_conv_output)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(graph_conv_output , hidden_channels*4), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*4), 
            torch.nn.Linear(hidden_channels*4 , hidden_channels*2), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels*2), 
        )
        
    def forward(self , x , edge_index , edge_attr , batch , batch_idx = None ,  log=False):
        
        input_batch = batch
        perms = []
        scores = []
        batches = []
        attrs = []
        for graph_conv , pooling , graph_norm in self.block_layers:
            x , learn_attrs = graph_conv(x , edge_index , edge_attr)
            x , edge_index , edge_attr , batch , perm , score = pooling(x , edge_index , edge_attr , batch)
            x = graph_norm(x , batch)
            
            attrs.append(learn_attrs)
            perms.append(perm)
            scores.append(score)
            batches.append(batch)
            
        # # First layer graph convolution
        # x , gc1_k1_edge_attr , gc1_k2_edge_attr , gc1_k3_edge_attr = self.graph_conv1(x , edge_index , edge_attr)
        # x , edge_index , _ , batch1 ,  perm_1 , score_1 = self.pooling(x , edge_index , edge_attr , batch)
        # x = self.graph_norm1(x , batch1)
        
        # # Second layer graph convolution
        # x , gc2_k1_edge_attr , gc2_k2_edge_attr , gc2_k3_edge_attr  = self.graph_conv2(x , edge_index)
        # x , edge_index , _ , batch2 ,  perm_2 , score_2 = self.pooling2(x , edge_index , batch=batch1)
        # x = self.graph_norm2(x , batch2)
        
        x = geom_nn.global_mean_pool(x , batch)
        gc1_edge_attr_dense = [ geom_utils.to_dense_adj(gc_layer_attr[0] , input_batch , edge_attr=gc_layer_attr[1]) for gc_layer_attr in attrs[0] ]
        gc1_dense = torch.stack(gc1_edge_attr_dense , dim=-1).mean(dim=-1)
        fea_attr_scr1 = gc1_dense.squeeze(dim=-1).mean(dim=0)
        
        # # Get the summary of the edge attention score from first block of graph convolution 
        # gc1_k1_edge_attr_dense = geom_utils.to_dense_adj(gc1_k1_edge_attr[0] , batch  , edge_attr=gc1_k1_edge_attr[1])
        # gc1_k2_edge_attr_dense = geom_utils.to_dense_adj(gc1_k2_edge_attr[0] , batch  , edge_attr=gc1_k2_edge_attr[1])
        # gc1_k3_edge_attr_dense = geom_utils.to_dense_adj(gc1_k3_edge_attr[0] , batch  , edge_attr=gc1_k3_edge_attr[1])
        # gc1_dense = torch.stack([gc1_k1_edge_attr_dense , gc1_k2_edge_attr_dense , gc1_k3_edge_attr_dense] , dim=-1).mean(dim=-1)
        # fea_attr_scr1 = gc1_dense.squeeze(dim=-1).mean(dim=0)
        
        # gc2_k1_edge_attr_dense = geom_utils.to_dense_adj(gc2_k1_edge_attr[0] , batch1 , edge_attr=gc2_k1_edge_attr[1])
        # gc2_k2_edge_attr_dense = geom_utils.to_dense_adj(gc2_k2_edge_attr[0] , batch1 , edge_attr=gc2_k2_edge_attr[1])
        # gc2_k3_edge_attr_dense = geom_utils.to_dense_adj(gc2_k3_edge_attr[0] , batch1 , edge_attr=gc2_k3_edge_attr[1])
        # gc2_dense = torch.stack([gc2_k1_edge_attr_dense , gc2_k2_edge_attr_dense , gc2_k3_edge_attr_dense] , dim=-1).mean(dim=-1)
        # fea_attr_scr2 = gc2_dense.squeeze(dim=-1).mean(dim=0)
        
        x = self.mlp(x)
        
        return x , perms , scores , batches , fea_attr_scr1 
    
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

class MultiGraphConvolution(torch.nn.Module):
    
    def __init__(self , in_channels , hidden_channels , output_channels) -> None:
        super().__init__()
        
        self.graph1 = GraphPooling(in_channels , hidden_channels)
        self.graph2 = GraphPooling(in_channels , hidden_channels)
        self.graph3 = GraphPooling(in_channels , hidden_channels)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*3*2 , hidden_channels),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Linear(hidden_channels , output_channels),
        )
        
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        output1 , perm11 , perm12 , score11 , score12 , batch11 , batch12 , attr_score11 , attr_score12 = self.graph1(x1 , edge_index1 , edge_attr1 , batch1_idx , log=True)
        output2 , perm21 , perm22 , score21 , score22 , batch21 , batch22 , attr_score21 , attr_score22 = self.graph2(x2 , edge_index2 , edge_attr2 , batch2_idx)
        output3 , perm31 , perm32 , score31 , score32 , batch31 , batch32 , attr_score31 , attr_score32 = self.graph3(x3 , edge_index3 , edge_attr3 , batch3_idx)
        
        output = torch.concat([output1 , output2 , output3] , dim=-1) # shape -> [ batch , hidden_dimension * 3 * 2 ]
        
        output = self.mlp(output)
        
        return output

class BinaryLearning(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr=0.0001 , drop_out = 0.1 , mlflow:mlflow = None , multi_graph_testing=False , weight=None , alpha=0.2 , binary = False , *args, **kwargs) -> None: 
        super().__init__()
        self.lr = lr 
        self.mlflow = mlflow
        self.multi_graph_testing = multi_graph_testing
        self.num_classes = num_classes
        
        self.multi_graph_conv = MultiGraphConvolution(in_channels , hidden_channels , 32)
        
        self.classifier = torch.nn.Linear(32 , 1) # binary classification
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.acc = Accuracy(task='binary' , num_classes=2)
        self.multi_acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        self.sensivity = Recall(task="multiclass" , num_classes=num_classes , average="macro")
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.predictions = []
        self.actuals = []
        self.print_results = []
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        x = self.multi_graph_conv(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        x = self.classifier(x)
        return x
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters() , lr= self.lr , weight_decay=0.0001)
    
    def training_step(self , batch):
        batch1 , batch2 , batch3 = batch 
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        loss = self.loss(output.squeeze(dim=-1) , y1.squeeze(dim = -1).float())
        acc = self.acc(torch.nn.functional.sigmoid(output).squeeze(dim=-1) , y1.squeeze(dim=-1))

        self.log("train_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        return loss
    
    def validation_step(self , batch):
        
        
        output , actual_class , batch_shape = self.get_output(batch)
        
        #loss = self.loss(output , actual_class)
        acc = self.multi_acc(torch.nn.functional.softmax(output) , actual_class)
        self.test_confusion_matrix.update(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        # f1 = self.f1(torch.nn.functional.sigmoid(output) , actual_class)
        # auroc = self.auc(torch.nn.functional.sigmoid(output) , actual_class)
        # specificity = self.specificity(torch.nn.functional.sigmoid(output) , actual_class)
        # sensivity = self.sensivity(torch.nn.functional.sigmoid(output) , actual_class)
        
        # self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_shape)
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        
    def get_output(self , batch):
        batch1 , batch2 , batch3 = batch # batch1 , batch2 , batch3 contains multiple topology of the same omic type
        
        print_row = []
        #results = []
        results_1 = []
        for i in range(self.num_classes):
            x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
            x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
            x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
            
            output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
            actual_class = y1
            batch_shape = batch1_idx.shape[0]
            # loss = self.loss(output , y1)
            output_softmax = torch.nn.functional.sigmoid(output)
            
            print_row.append(f"Topology: {i} | P: {output_softmax[0].item()} | A: {actual_class[0].item()}")
            results_1.append(output_softmax[0].item())
        
        self.print_results.append(" , ".join(print_row)) 
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        final_prediction = torch.tensor([results_1], dtype=torch.float32 , device=device)
        return final_prediction , actual_class , batch_shape
    
    def on_validation_epoch_end(self) -> None: 
        
        if self.current_epoch % 10 == 0:
            
            if self.mlflow is not None:
                logger.info("Logging confusion matrix for test epoch {}".format(self.current_epoch))
                # calculate confusion matrix

                self.test_confusion_matrix.compute()
                fig , ax = self.test_confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "test_confusion_matrix_epoch_{}.png".format(self.current_epoch))
                plt.close(fig)
                
                # save txt 
                self.mlflow.log_text("\n".join(self.print_results) , "test_prediction_epoch_{}.txt".format(self.current_epoch))
        
        # reset 
        self.print_results = []
        self.test_confusion_matrix.reset()
class ContrastiveLearning(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr=0.0001 , drop_out = 0.1 , mlflow:mlflow = None , multi_graph_testing=False , weight=None) -> None: 
        super().__init__()
        self.lr = lr 
        self.mlflow = mlflow
        self.multi_graph_testing = multi_graph_testing
        self.num_classes = num_classes
        
        self.multi_graph_conv = MultiGraphConvolution(in_channels , hidden_channels , 32)
        
        self.classifier = torch.nn.Linear(32 , 1) # binary classification
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.acc = Accuracy(task='binary' , num_classes=2)
        self.multi_acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.binary_loss_positive = torch.nn.BCEWithLogitsLoss()
        self.binary_loss_negative = torch.nn.BCEWithLogitsLoss()
        self.pairwise_distance = torch.nn.CosineSimilarity()
        # self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        # self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        # self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        # self.sensivity = Recall(task="multiclass" , num_classes=num_classes , average="macro")
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        embedding = self.multi_graph_conv(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        output = self.classifier(embedding)
        return output , embedding
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters() , lr= self.lr , weight_decay=0.0001)
    
    def get_train_output(self, batch , i):
        batch1 , batch2 , batch3 = batch
        
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
        
        output , embedding = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        return output , embedding
        
    def training_step(self , batch):
        
        # Positive pair
        output_positive , embedding_positive = self.get_train_output(batch , 0)
        output_negative , embedding_negative = self.get_train_output(batch , 1)
        
        loss = self.binary_loss_positive(output_positive.squeeze(dim=-1) , torch.ones_like(output_positive.squeeze(dim=-1))) + self.binary_loss_negative(output_negative.squeeze(dim=-1) , torch.zeros_like(output_negative.squeeze(dim=-1))) + self.pairwise_distance(embedding_positive , embedding_negative).mean()
        acc = self.acc(torch.nn.functional.sigmoid(output_positive).squeeze(dim=-1) , torch.ones_like(output_positive).squeeze(dim=-1))

        self.log("train_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch[0][0].batch.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch[0][0].batch.shape[0])
        return loss
    
    def validation_step(self , batch):
        
        
        output , actual_class , batch_shape = self.get_output(batch)
        
        #loss = self.loss(output , actual_class)
        acc = self.multi_acc(torch.nn.functional.softmax(output) , actual_class)
        # f1 = self.f1(torch.nn.functional.sigmoid(output) , actual_class)
        # auroc = self.auc(torch.nn.functional.sigmoid(output) , actual_class)
        # specificity = self.specificity(torch.nn.functional.sigmoid(output) , actual_class)
        # sensivity = self.sensivity(torch.nn.functional.sigmoid(output) , actual_class)
        
        # self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_shape)
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        
    def get_output(self , batch):
        batch1 , batch2 , batch3 = batch # batch1 , batch2 , batch3 contains multiple topology of the same omic type
        
        #results = []
        results_1 = []
        for i in range(self.num_classes):
            x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
            x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
            x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
            
            output , embedding = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
            acutal_class = y1
            batch_shape = batch1_idx.shape[0]
            # loss = self.loss(output , y1)
            output_softmax = torch.nn.functional.sigmoid(output)
            
            with open("multigraph_testing_logs.txt" , "a") as log_file: 
                log_file.write(f"Epoch: {self.current_epoch}\t| Topology: {i}\t| Confidence score: {output_softmax}\t| Actual class: {y1}\n")
            #results.append({'topology': i , 'predicted_class': predicted_class , 'predicted_prob': predicted_prob , 'output': output })
            results_1.append(output_softmax[0].item())
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        final_prediction = torch.tensor([results_1], dtype=torch.float32 , device=device)
        return final_prediction , acutal_class , batch_shape
    
class TripletLearning(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr=0.0001 , drop_out = 0.1 , mlflow:mlflow = None , multi_graph_testing=False , weight=None , alpha=0.2 , binary = False , *args, **kwargs) -> None: 
        super().__init__()
        self.lr = lr 
        self.mlflow = mlflow
        self.multi_graph_testing = multi_graph_testing
        self.num_classes = num_classes
        self.binary = binary
        self.multi_graph_conv = MultiGraphConvolution(in_channels , hidden_channels , 32)
        
        self.classifier = torch.nn.Linear(32 , num_classes if not binary else 1) # class classification
        self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weight , dtype=torch.float32) if weight is not None else None
            ) if not binary else torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.acc = Accuracy(task='multiclass' , num_classes=num_classes) if not binary else Accuracy(task='binary')
        self.test_acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.triplet_loss = torch.nn.TripletMarginLoss()
        # self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        # self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        # self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        # self.sensivity = Recall(task="multiclass" , num_classes=num_classes , average="macro")
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        embedding = self.multi_graph_conv(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        output = self.classifier(embedding)
        return output , embedding
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters() , lr= self.lr , weight_decay=0.0001)
    
    def get_train_output(self, batch , i):
        batch1 , batch2 , batch3 = batch
        
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
        
        output , embedding = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        return output , embedding , y1
        
    def training_step(self , batch):
        
        # Positive pair
        output_anchor , embedding_anchor , actual_anchor = self.get_train_output(batch , 0)
        output_positive , embedding_positive , actual_positive = self.get_train_output(batch , 1)
        output_negative , embedding_negative , actual_negative = self.get_train_output(batch , 2)
        
        loss = self.alpha * self.triplet_loss(embedding_anchor , embedding_positive , embedding_negative) \
            + (1 - self.alpha) * self.loss(
                torch.nn.functional.softmax(output_anchor , dim=-1) if not self.binary else torch.nn.functional.sigmoid(output_anchor.squeeze()) , 
                actual_anchor if not self.binary else torch.ones_like(actual_positive.squeeze(dim=-1) , dtype=torch.float)
                )
            
        if self.binary: 
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            loss += (1 - self.alpha) * self.loss(torch.nn.functional.sigmoid(output_negative.squeeze()) , torch.zeros_like(actual_negative.squeeze(dim=-1) , dtype=torch.float , device=device))
            
        acc = self.acc(
            torch.nn.functional.softmax(output_positive , dim=-1) if not self.binary else torch.nn.functional.sigmoid(output_positive.squeeze()) , 
            actual_positive.squeeze(dim=-1) if not self.binary else torch.ones_like(actual_positive.squeeze(dim=-1) , dtype=torch.long)
        )

        self.log("train_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch[0][0].batch.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch[0][0].batch.shape[0])
        return loss
    
    def validation_step(self , batch):
        
        
        output , actual_class , batch_shape = self.get_output(batch)
        
        #loss = self.loss(output , actual_class)
        acc = self.test_acc(
            torch.nn.functional.softmax(output , dim=-1) , actual_class)
        self.test_confusion_matrix.update(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        # f1 = self.f1(torch.nn.functional.sigmoid(output) , actual_class)
        # auroc = self.auc(torch.nn.functional.sigmoid(output) , actual_class)
        # specificity = self.specificity(torch.nn.functional.sigmoid(output) , actual_class)
        # sensivity = self.sensivity(torch.nn.functional.sigmoid(output) , actual_class)
        
        # self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_shape)
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        
    def get_output(self , batch):
        batch1 , batch2 , batch3 = batch # batch1 , batch2 , batch3 contains multiple topology of the same omic type
        
        #results = []
        results_1 = []
        for i in range(self.num_classes):
            x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
            x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
            x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
            
            output , embedding = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
            acutal_class = y1
            batch_shape = batch1_idx.shape[0]
            
            class_confidence = torch.nn.functional.softmax(output , dim=-1)[0][i].item() if not self.binary else torch.nn.functional.sigmoid(output)[0].item()
            
            with open("multigraph_testing_logs.txt" , "a") as log_file: 
                log_file.write(f"Epoch: {self.current_epoch}\t| Topology: {i}\t| Confidence score: {class_confidence}\t| Actual class: {y1}\n")
            #results.append({'topology': i , 'predicted_class': predicted_class , 'predicted_prob': predicted_prob , 'output': output })
            results_1.append(class_confidence)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        final_prediction = torch.tensor([results_1], dtype=torch.float32 , device=device)
        return final_prediction , acutal_class , batch_shape
    
    def on_validation_epoch_end(self) -> None: 
        
        if self.current_epoch % 10 == 0:
            
            if self.mlflow is not None:
                logger.info("Logging confusion matrix for test epoch {}".format(self.current_epoch))
                # calculate confusion matrix

                self.test_confusion_matrix.compute()
                fig , ax = self.test_confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "test_confusion_matrix_epoch_{}.png".format(self.current_epoch))
                plt.close(fig)
                
        self.test_confusion_matrix.reset()
class MultiGraphClassification(pl.LightningModule):
    
    def __init__(self , in_channels , hidden_channels , num_classes , lr=0.0001 , drop_out = 0.1 , mlflow:mlflow = None , multi_graph_testing=False , weight=None , *args, **kwargs) -> None:
        super().__init__() 
        
        self.lr = lr 
        self.mlflow = mlflow
        self.multi_graph_testing = multi_graph_testing
        self.num_classes = num_classes
        self.drop_out = drop_out
        self.optim = kwargs.get('optimizer' , 'adam')
        self.weight_decay = kwargs.get('decay' , 0.0001)
        self.momentum = kwargs.get('momentum' , 0.9)
        
        self.graph1 = GraphPooling(in_channels , hidden_channels , **kwargs)
        self.graph2 = GraphPooling(in_channels , hidden_channels , **kwargs)
        self.graph3 = GraphPooling(in_channels , hidden_channels , **kwargs)
        
        
        self.class_1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2 , hidden_channels), 
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels , num_classes)
        )
        
        self.class_2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2 , hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels , num_classes)
        )
        
        self.class_3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2 , hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels , num_classes)
        )
        
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2 , hidden_channels), # 3 omics with 2 * hidden_channels
            torch.nn.Dropout1d(drop_out),
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.Linear(hidden_channels , num_classes),
        )
        
        self.acc = Accuracy(task='multiclass' , num_classes=num_classes)
        self.acc_1 = Accuracy(task='multiclass' , num_classes=num_classes)
        self.acc_2 = Accuracy(task='multiclass' , num_classes=num_classes)
        self.acc_3 = Accuracy(task='multiclass' , num_classes=num_classes)
        self.paper_acc = Accuracy(task='multiclass' , num_classes=num_classes)
        if weight is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight , dtype=torch.float32))
        else: 
            self.loss = torch.nn.CrossEntropyLoss()
            
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        self.train_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.specificity = Specificity(task="multiclass" , num_classes=num_classes)
        self.sensivity = Recall(task="multiclass" , num_classes=num_classes , average="macro")
        self.softmax_prediction = []
        self.predictions = []
        self.actuals = []
        self.predictions_1 = []
        self.actuals_1 = []
        self.predictions_2 = []
        self.actuals_2 = []
        self.predictions_3 = []
        self.actuals_3 = []
    
    def forward(self , x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx):
        output1 , perm1 , score1 , batch1 , attr_score11  = self.graph1(x1 , edge_index1 , edge_attr1 , batch1_idx , log=True)
        output2 , perm2 , score2 , batch2 , attr_score21  = self.graph2(x2 , edge_index2 , edge_attr2 , batch2_idx)
        output3 , perm3 , score3 , batch3 , attr_score31  = self.graph3(x3 , edge_index3 , edge_attr3 , batch3_idx)
        
        #output = torch.concat([output1 , output2 , output3] , dim=-1) # shape -> [ batch , hidden_dimension * 3 * 2 ]
        output = output1 * output2 * output3 # element wise multiplication
        output = self.mlp(output)
        
        output1 = self.class_1(output1)
        output2 = self.class_2(output2)
        output3 = self.class_3(output3)
        
        return output , output1 , output2 , output3
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        if self.optim == 'adam':
            self.optimizer =  optim.Adam(self.parameters() , lr= self.lr , weight_decay=self.weight_decay)
        elif self.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters() , lr=self.lr , weight_decay=self.weight_decay , momentum=self.momentum)
        elif self.optim == 'adamw':
            self.optimizer = optim.AdamW(self.parameters() , lr=self.lr , weight_decay=self.weight_decay)
        elif self.optim == 'rms':
            self.optimizer = optim.RMSprop(self.parameters() , lr=self.lr , weight_decay=self.weight_decay) 
            
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer , mode='min' , factor=0.1 , patience=10 , verbose=True , min_lr=1e-4)

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }
    def training_step(self , batch):
        batch1 , batch2 , batch3 = batch
        x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
        x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
        x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
        
        output , output1 , output2 , output3 = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
        loss = self.loss(output , y1)
        loss1 = self.loss(output1 , y1)
        loss2 = self.loss(output2 , y2)
        loss3 = self.loss(output3 , y3)
        total_loss = loss + loss1 + loss2 + loss3
        
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1) , y1)
        acc1 = self.acc_1(torch.nn.functional.softmax(output1 , dim=-1) , y1)
        acc2 = self.acc_2(torch.nn.functional.softmax(output2 , dim=-1) , y2)
        acc3 = self.acc_3(torch.nn.functional.softmax(output3 , dim=-1) , y3)
        latest_lr = self.optimizer.param_groups[0]['lr']
        self.train_confusion_matrix.update(output , y1)  
        
        self.log("train_loss" , total_loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("train_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("train_acc_omic1" , acc1 , on_epoch=True, on_step=False , prog_bar=False ,  batch_size=batch1_idx.shape[0])
        self.log("train_acc_omic2" , acc2 , on_epoch=True, on_step=False , prog_bar=False ,  batch_size=batch1_idx.shape[0])
        self.log("train_acc_omic3" , acc3 , on_epoch=True, on_step=False , prog_bar=False ,  batch_size=batch1_idx.shape[0])
        self.log("learning_rate" , latest_lr , on_epoch=True , on_step=False , prog_bar=False ,  batch_size=batch1_idx.shape[0])
        return total_loss
    
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
        output , actual_class , batch_shape , paper_output , output1 , output2 , output3 = self.get_output(batch)
        
        loss = self.loss(output , actual_class)
        acc = self.acc(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        if paper_output is not None:
            paper_acc = self.paper_acc(torch.nn.functional.softmax(paper_output , dim=-1) , actual_class)
            self.log('val_paper_acc' , paper_acc , on_step=False , prog_bar=True , batch_size=batch_shape)
        else:
            acc1 = self.acc_1(torch.nn.functional.softmax(output1 , dim=-1) , actual_class)
            acc2 = self.acc_2(torch.nn.functional.softmax(output2 , dim=-1) , actual_class)
            acc3 = self.acc_3(torch.nn.functional.softmax(output3 , dim=-1) , actual_class)
            
            loss1 = self.loss(output1 , actual_class)
            loss2 = self.loss(output2 , actual_class)
            loss3 = self.loss(output3 , actual_class)
            loss = loss + loss1 + loss2 + loss3
            
            self.log("val_acc_omic1" , acc1 , on_epoch=True, on_step=False , prog_bar=False ,  batch_size=batch_shape)
            self.log("val_acc_omic2" , acc2 , on_epoch=True, on_step=False , prog_bar=False ,  batch_size=batch_shape)
            self.log("val_acc_omic3" , acc3 , on_epoch=True, on_step=False , prog_bar=False ,  batch_size=batch_shape)
            
        f1 = self.f1(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        # auroc = self.auc(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        specificity = self.specificity(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        sensivity = self.sensivity(torch.nn.functional.softmax(output , dim=-1) , actual_class)
        self.test_confusion_matrix.update(output , actual_class)  
        self.softmax_prediction.extend(torch.softmax(output , dim=-1).cpu().numpy())
        self.predictions.extend(torch.argmax(torch.nn.functional.softmax(output , dim=-1) , dim=-1).cpu().numpy())
        self.actuals.extend(actual_class.cpu().numpy())
        self.predictions_1.extend(torch.argmax(torch.nn.functional.softmax(output1 , dim=-1) , dim=-1).cpu().numpy())
        self.actuals_1.extend(actual_class.cpu().numpy())
        self.predictions_2.extend(torch.argmax(torch.nn.functional.softmax(output2 , dim=-1) , dim=-1).cpu().numpy())
        self.actuals_2.extend(actual_class.cpu().numpy())
        self.predictions_3.extend(torch.argmax(torch.nn.functional.softmax(output3 , dim=-1) , dim=-1).cpu().numpy())
        self.actuals_3.extend(actual_class.cpu().numpy())
            
        self.log("val_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch_shape)
        self.log("val_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        self.log("val_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        # self.log("val_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        self.log("val_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
        self.log("val_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch_shape)
    
    def on_validation_epoch_end(self) -> None: 
        
        if (self.current_epoch+1) % 10 == 0:
            
            if self.mlflow is not None:
                # logger.info("Logging confusion matrix for test epoch {}".format(self.current_epoch+1))
                # calculate confusion matrix

                self.test_confusion_matrix.compute()
                fig , ax = self.test_confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "confusion_matrix_epoch_{:03d}_test.png".format(self.current_epoch+1))
                plt.close(fig)
                
                # logger.info(f"Logging classification report for test epoch {self.current_epoch+1}")
                report = f"--------- Overall Test Report ---------\n"
                report += classification_report(self.actuals , self.predictions , zero_division=0 , digits=4)
                
                # calculate auroc 
                auroc = roc_auc_score(self.actuals , self.softmax_prediction , multi_class="ovr" , average="macro")
                report += f"AUC [macro] : {auroc}\n"
                auroc = roc_auc_score(self.actuals , self.softmax_prediction , multi_class="ovr" , average="weighted")
                report += f"AUC [weighted] : {auroc}\n"
                auroc = roc_auc_score(self.actuals , self.softmax_prediction , multi_class="ovr" , average="micro")
                report += f"AUC [micro] : {auroc}\n\n"
                #self.mlflow.log_text(report , "test_classification_report_epoch_{}.txt".format(self.current_epoch+1))
                
                report += f"--------- Omic1 Test Report ---------\n"
                report += classification_report(self.actuals_1 , self.predictions_1 , zero_division=0, digits=4)
                #self.mlflow.log_text(report_1 , "test_classification_report_omic1_epoch_{}.txt".format(self.current_epoch+1))
                
                report += f"--------- Omic2 Test Report ---------\n"
                report += classification_report(self.actuals_2 , self.predictions_2 , zero_division=0, digits=4)
                #self.mlflow.log_text(report_2 , "test_classification_report_omic2_epoch_{}.txt".format(self.current_epoch+1))
                
                report += f"--------- Omic3 Test Report ---------\n"
                report += classification_report(self.actuals_3 , self.predictions_3 , zero_division=0, digits=4)
                #self.mlflow.log_text(report_3 , "test_classification_report_omic3_epoch_{}.txt".format(self.current_epoch+1))
                self.mlflow.log_text(report , "classification_report_epoch_{:03d}_test.txt".format(self.current_epoch+1))
                
        self.test_confusion_matrix.reset()
        self.softmax_prediction = []
        self.predictions = []
        self.actuals = []
        self.predictions_1 = []
        self.actuals_1 = []
        self.predictions_2 = []
        self.actuals_2 = []
        self.predictions_3 = []
        self.actuals_3 = []
    
    def on_train_epoch_end(self) -> None:
        
        if (self.current_epoch+1) % 10 == 0:
            if self.mlflow is not None:
                # logger.info("Logging confusion matrix for training epoch {}".format(self.current_epoch+1))
                # calculate confusion matrix
                self.train_confusion_matrix.compute()
                fig , ax = self.train_confusion_matrix.plot() 
                self.mlflow.log_figure(fig , "confusion_matrix_epoch_{:03d}_train.png".format(self.current_epoch+1))
                plt.close(fig)
        
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
        
        # omic1_pool1 , omic1_pool2 = self.get_rank_genes(self.rank['omic1'] , batch1.extra_label , batch1.num_graphs , 1000)
        # omic2_pool1 , omic2_pool2 = self.get_rank_genes(self.rank['omic2'] , batch2.extra_label , batch2.num_graphs , 1000)
        # omic3_pool1 , omic3_pool2 = self.get_rank_genes(self.rank['omic3'] , batch3.extra_label , batch3.num_graphs , 503)
        
        # self.genes['omic1_pool1'].extend(omic1_pool1)
        # self.genes['omic1_pool2'].extend(omic1_pool2)
        # self.genes['omic2_pool1'].extend(omic2_pool1)
        # self.genes['omic2_pool2'].extend(omic2_pool2)
        # self.genes['omic3_pool1'].extend(omic3_pool1)
        # self.genes['omic3_pool2'].extend(omic3_pool2)
        
        # self.allrank['omic1'].append(self.rank['omic1'])
        # self.allrank['omic2'].append(self.rank['omic2'])
        # self.allrank['omic3'].append(self.rank['omic3'])
        
        self.log("test_loss" , loss , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
        self.log("test_acc" , acc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_f1" , f1 , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_auroc" , auroc , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_spe" , specificity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        self.log("test_sen" , sensivity , on_epoch=True, on_step=False , prog_bar=True ,  batch_size=batch1_idx.shape[0])
        
    def multi_graph_test(self , batch):
        batch1 , batch2 , batch3 = batch # batch1 , batch2 , batch3 contains multiple topology of the same omic type
        
        results = []
        results_1 = []
        for i in range(self.num_classes):
            x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
            x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
            x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
            
            output = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
        
            # loss = self.loss(output , y1)
            output_softmax = torch.nn.functional.softmax(output , dim=-1)
            predicted_class = output_softmax.argmax(dim=-1)
            predicted_prob , _ = output_softmax.max(dim=-1)
            
            with open("multigraph_testing_logs.txt" , "a") as log_file: 
                log_file.write(f"Epoch: {self.current_epoch}\t| Topology: {i}\t| Predicted class: {predicted_class}\t| Predicted probability: {predicted_prob}\t | Output: {output_softmax}\t| Class confidence score: {output_softmax[0][i]}\t| Actual class: {y1}\n")
            results.append({'topology': i , 'predicted_class': predicted_class , 'predicted_prob': predicted_prob , 'output': output })
            results_1.append(output_softmax[0][i].item())
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        final_prediction = torch.tensor([results_1], dtype=torch.float32 , device=device)
        # get the top predicted probability
        # final_prediction = max(results, key=lambda x: x['predicted_prob'])
        with open("multigraph_testing_logs.txt" , "a") as log_file: 
            log_file.write(f"Epoch: {self.current_epoch}\t| Final prediction: {final_prediction.argmax(dim=-1)} | Actual class: {y1}\n")
            log_file.write("\n")
            
        acc = self.acc(final_prediction  , y1)
        self.log('test_acc_top_predicted_class' , acc , on_epoch=True , on_step=False , prog_bar=True , batch_size=batch1_idx.shape[0])
    
    def get_output(self , batch):
        if not self.multi_graph_testing:
            batch1 , batch2 , batch3 = batch 
            x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1.x , batch1.edge_index , batch1.edge_attr , batch1.batch ,  batch1.y
            x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2.x , batch2.edge_index , batch2.edge_attr , batch2.batch ,  batch2.y
            x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3.x , batch3.edge_index , batch3.edge_attr , batch3.batch ,  batch3.y
            
            output , output1 , output2 , output3 = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
            
            return output , y1 , batch1_idx.shape[0] , None , output1 , output2 , output3
        else: 
            batch1 , batch2 , batch3 = batch # batch1 , batch2 , batch3 contains multiple topology of the same omic type
        
            store_result = [] # number_of_classes * number_of_topologies
            results_1 = []
            paper_result = []
            for i in range(self.num_classes):
                x1 , edge_index1 , edge_attr1 , batch1_idx ,  y1 = batch1[i].x , batch1[i].edge_index , batch1[i].edge_attr , batch1[i].batch ,  batch1[i].y
                x2 , edge_index2 , edge_attr2 , batch2_idx ,  y2 = batch2[i].x , batch2[i].edge_index , batch2[i].edge_attr , batch2[i].batch ,  batch2[i].y
                x3 , edge_index3 , edge_attr3 , batch3_idx ,  y3 = batch3[i].x , batch3[i].edge_index , batch3[i].edge_attr , batch3[i].batch ,  batch3[i].y
                
                output , _ , _ , _ = self.forward(x1 , edge_index1 , edge_attr1 , x2 , edge_index2 , edge_attr2 , x3 , edge_index3 , edge_attr3 , batch1_idx , batch2_idx , batch3_idx)
                acutal_class = y1
                batch_shape = batch1_idx.shape[0]
                # loss = self.loss(output , y1)
                output_softmax = torch.nn.functional.softmax(output , dim=-1)
                
                # predicted_class = output_softmax.argmax(dim=-1)
                # predicted_prob , _ = output_softmax.max(dim=-1)
                # with open("multigraph_testing_logs.txt" , "a") as log_file: 
                #     log_file.write(f"Epoch: {self.current_epoch}\t| Topology: {i}\t| Predicted class: {predicted_class}\t| Predicted probability: {predicted_prob}\t | Output: {output_softmax}\t| Class confidence score: {output_softmax[0][i]}\t| Actual class: {y1}\n")
                # results.append({'topology': i , 'predicted_class': predicted_class , 'predicted_prob': predicted_prob , 'output': output })
                
                store_result.extend(output_softmax[0].tolist())
                store_result.extend([y1.item()])
                results_1.append(output_softmax[0][i].item())
                
                if i == y1.item():
                    paper_result = output 
            # log only the last epoch 
            if self.current_epoch % 10 == 0:
                with open(f"multigraph_testing_epochs_{self.current_epoch}_logs.txt" , "a") as log_file: 
                    log_file.write("\t".join([f"{x:.4f}" for x in store_result]))
                    log_file.write("\n")
            
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            final_prediction = torch.tensor([results_1], dtype=torch.float32 , device=device)
            return final_prediction , acutal_class , batch_shape , paper_result
        
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
    
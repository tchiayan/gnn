import lightning as pl 
import torch 
from torch import optim
import torch_geometric.nn as geom_nn
from math import ceil
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import dense_diff_pool
from torch_geometric.loader import DenseDataLoader , DataLoader
from utils import generate_graph , read_features_file 
from sklearn.model_selection import StratifiedKFold
from torchmetrics import AUROC , F1Score , Accuracy
import os 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import mlflow
# import early Stopping


base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")

class GCN(torch.nn.Module):
    
    def __init__(self , in_channels , hidden_channels , out_channels , lin=True):
        super().__init__()
        self.conv1 = geom_nn.DenseSAGEConv(in_channels , hidden_channels )
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = geom_nn.DenseSAGEConv(hidden_channels , hidden_channels  )
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = geom_nn.DenseSAGEConv(hidden_channels , out_channels  )
        self.batch_norm3 = torch.nn.BatchNorm1d(out_channels)
    
        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels , out_channels)
        else:
            self.lin = None

    def bn(self , i , x):
        batch_size , nodes , num_channels = x.size()
        
        x = x.view(-1 , num_channels)
        x = getattr(self , 'batch_norm{}'.format(i))(x)
        x = x.view(batch_size , nodes , num_channels)
        return x 
                
    def forward(self , x , edge_index , mask=None):
        x0 = x
        x1 = self.bn(1 , self.conv1(x0 , edge_index , mask).relu())
        x2 = self.bn(2 , self.conv2(x1 , edge_index , mask).relu())
        x3 = self.bn(3 , self.conv3(x2 , edge_index , mask).relu())
        x = torch.cat([x1 , x2 , x3] , dim=-1)
        
        if self.lin is not None:
            x = self.lin(x).relu()
            
        return x
    
class DiffPool(pl.LightningModule):
    
    def __init__(self  , num_features , num_classes , max_nodes=150 , hidden_embedding = 32):
        super().__init__()
        
        num_nodes = ceil(0.25 * max_nodes) # maximum of cluster nodes after first layer of pooling 
        self.gnn1_pool = GCN(num_features , hidden_embedding , num_nodes)
        self.gnn1_embed = GCN(num_features , hidden_embedding , hidden_embedding , lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GCN(3*hidden_embedding , hidden_embedding , num_nodes , lin=False)
        self.gnn2_embed = GCN(3*hidden_embedding , hidden_embedding , hidden_embedding , lin=False)
        
        self.gnn3_embed = GCN(3*hidden_embedding , hidden_embedding , hidden_embedding , lin=False)
        
        self.lin1 = torch.nn.Linear(3*hidden_embedding , hidden_embedding)
        self.lin2 = torch.nn.Linear(hidden_embedding ,  hidden_embedding)
        self.lin3 = torch.nn.Linear(hidden_embedding , num_classes)
        
        self.acc = Accuracy(task='multiclass' , num_classes = num_classes)
        self.auc = AUROC(task='multiclass' , num_classes=num_classes)
        self.f1 = F1Score(task='multiclass' , num_classes=num_classes , average='macro')
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self , x , edge_index  , mask=None):
        batch_size , node_size , _ = x.size()
        
        print("Shape of x" , x.size() , "| Shape of edge: ", edge_index.size() )
        adj = to_dense_adj(edge_index[0])
        print("Dense adj shape: " , adj.size())
        adj = adj.expand(batch_size , node_size , node_size )
        print("Batch dense adj shape: " , adj.size())
        
        s = self.gnn1_pool(x , adj , mask) # size => ( batch_size , num_nodes)
        x = self.gnn1_embed(x , adj , mask) # size => ( batch_size , num_nodes)
        print("1st layer pooling shape: " , s.size())
        print("1st layer embedding shape: " , x.size())
        
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        print("New adj shape: ", adj.size())
        print("New input shape: " , x.size())
        
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1) # size: sample_N x number_of_node x node_dimension (Number of classes)
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x).relu()
        x = self.softmax(x)
        return x  , l1+l2  , e1+e2
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters() , lr=1e-3)
        return optimizer 
    
    def training_step(self , batch , batch_idx):
        
        output , _l , _e = self.forward(batch.x , batch.edge_index , None)
        loss = torch.nn.functional.nll_loss(output , batch.y.unsqueeze(0).view(-1)) 
        
        acc = self.acc(output , batch.y)
        
        self.log('train_loss' , loss , prog_bar=True)
        self.log('train_acc' , acc , prog_bar=True)
        
        return loss
    
    def validation_step(self , batch , batch_idx):
        output , _ , _ = self.forward(batch.x , batch.edge_index , None)
        loss = torch.nn.functional.nll_loss(output , batch.y.unsqueeze(0).view(-1))
        
        acc = self.acc(output , batch.y)
        f1 = self.f1(output , batch.y)
        auc = self.auc(output , batch.y)
        
        self.log('val_loss' , loss , prog_bar=True)
        self.log('val_acc' , acc ,  prog_bar=True)
        self.log('val_auc' , auc , prog_bar=True)
        self.log('val_f1' , f1 , prog_bar=True)
    
def main():
    
    # read labels
    labels = os.path.join(base_path, "labels_tr.csv")
    df_labels = read_features_file(labels) 

    feature2 = os.path.join(base_path, "1_tr.csv")
    df2 = read_features_file(feature2)
    name2 = os.path.join(base_path, "1_featname.csv")
    df2_header = read_features_file(name2)
    gp2 = generate_graph(df2 , df2_header , df_labels[0].tolist(), threshold=400)
    
    kf = StratifiedKFold(n_splits=10 , shuffle=True)
    batch_size = 50 
    
    for i , (train_index , test_index) in enumerate(kf.split(df2.values , df_labels[0].values)):
        
        # split data into train and test set
        train_data = [gp2[idx] for idx in train_index]
        test_data = [gp2[idx] for idx in test_index]
        
        train_loader_2 = DataLoader(train_data , batch_size=batch_size , shuffle=True)
        for batch in train_loader_2:
            print(batch.size())
            break
        
        train_loader = DenseDataLoader(train_data , batch_size=batch_size , shuffle=True)
        test_loader = DenseDataLoader(test_data , batch_size=len(test_index))
        
        # Defined model
        model = DiffPool(1 , 5 , 100 , 32)
        
        # train model 
        mlflow.pytorch.autolog(
            log_every_n_epoch=1, 
            log_every_n_step=2
        )
        trainer = pl.Trainer(
            max_epochs=1 , 
            callbacks=[
                EarlyStopping(monitor='val_loss' , patience=4 , mode='min')        
            ], 
        )
        trainer.fit(
            model=model , 
            train_dataloaders=train_loader, 
            val_dataloaders=test_loader
        )
        
        break
    
if __name__ == "__main__":
    main()
        
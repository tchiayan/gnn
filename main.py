import os
import pandas as pd
from torch import nn
import torch 
import torch_geometric.nn as geom_nn
from torchmetrics import AUROC , F1Score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch import optim
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
from math import ceil 
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import dense_diff_pool
import torch.nn.functional as F
from torch_geometric.loader import DenseDataLoader

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BRCA")

class GNN(nn.Module):
    def __init__(self , in_channels , hidden_channels , out_channels):
        super(GNN , self).__init__()
        self.conv1 = geom_nn.GCNConv(in_channels , hidden_channels)
        self.conv2 = geom_nn.GCNConv(hidden_channels , hidden_channels)
        self.conv3 = geom_nn.GCNConv(hidden_channels , hidden_channels)
        self.conv4 = geom_nn.GCNConv(hidden_channels , out_channels)
        self.batch_norm1 = geom_nn.BatchNorm(hidden_channels)
        self.batch_norm2 = geom_nn.BatchNorm(hidden_channels)
        self.batch_norm3 = geom_nn.BatchNorm(hidden_channels)
    
    def forward(self , x , edge_index):
        x = self.conv1(x , edge_index).relu()
        x = self.batch_norm1(x)
        x = self.conv2(x , edge_index).relu()
        x = self.batch_norm2(x)
        x = self.conv3(x , edge_index).relu()
        x = self.batch_norm3(x)
        x = self.conv4(x , edge_index).relu()
        return x

class MOGONET(nn.Module):
    def __init__(self , num_features ,  num_classes , train_graph_label = False  ) -> None:
        super(MOGONET , self).__init__()
        
        in_channels = num_features
        hidden_channels = 64
        out_channels = 64
        if train_graph_label:
            out_channels = num_classes
        
        self.gnn1 = GNN(in_channels[0], hidden_channels , out_channels) 
        self.gnn2 = GNN(in_channels[1] , hidden_channels , out_channels)
        self.gnn3 = GNN(in_channels[2] , hidden_channels , out_channels)
        
        vdcn_hidden_channels = 64
        
        self.vcdn = nn.Sequential(
            nn.Linear(3 * out_channels , vdcn_hidden_channels), 
            nn.LeakyReLU(0.25) , 
            nn.Linear(vdcn_hidden_channels , num_classes)
        )
        
    def forward(self , x1 , x2 , x3 , edge_index1 , edge_index2 , edge_index3):
        x1 = self.gnn1(x1 , edge_index1)
        x2 = self.gnn2(x2 , edge_index2)
        x3 = self.gnn3(x3 , edge_index3)
        x = torch.cat([x1 , x2 , x3] , dim = 1)
        x = self.vcdn(x)
        return x , x1 , x2 , x3 
    
def read_features_file(path):
    df = pd.read_csv(path, header=None)
    return df

def generate_edge_index(df):
    cos_similarity = cosine_similarity(df)
    q4 = np.quantile(cos_similarity, 0.75)
    
    edge_index = [[] , []]
    for i in range(len(cos_similarity)):
        for j in range(len(cos_similarity)):
            if cos_similarity[i][j] > q4:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j) # undirected edge
                edge_index[1].append(i) # undirected edge

    print("Number of edges: ", len(edge_index[0]))
    return edge_index

def generate_graph(df , edge_index , labels , train_index , test_index):
    graph = Data(
        x = torch.tensor(df.values , dtype = torch.float32) , 
        edge_index = torch.tensor(edge_index , dtype = torch.long), 
        y = labels
    )
    
    
    graph.train_mask = torch.zeros(len(graph.x), dtype=torch.bool)
    graph.test_mask = torch.zeros(len(graph.x), dtype=torch.bool)
    graph.train_mask[train_index] = torch.tensor(True)
    graph.test_mask[test_index] = torch.tensor(True)
    
    assert len(graph.x.size()) == 2
    assert len(graph.edge_index.size()) == 2
    
    return graph

def train_mogonet():
    # Read features 1 file 
    feature1 = os.path.join(base_path, "1_tr.csv")
    df1 = read_features_file(feature1)
    ei1 = generate_edge_index(df1)
    
    feature2 = os.path.join(base_path, "2_tr.csv")
    df2 = read_features_file(feature2)
    ei2 = generate_edge_index(df2)
    #print(gph2)
    
    feature3 = os.path.join(base_path, "3_tr.csv")
    df3 = read_features_file(feature3)
    ei3 = generate_edge_index(df3)
    
    # read labels
    labels = os.path.join(base_path, "labels_tr.csv")
    df_labels = read_features_file(labels)
    labels = torch.tensor(df_labels.values, dtype=torch.long).view(-1)
    
    k_fold = 10 
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True)
    summary_df = pd.DataFrame(columns=["Fold", "Best Test Accuracy", "Test Epoch" , "F1 score" , "AUC"])
    
    for  i  , (train_index , test_index) in enumerate(skf.split(df1.values , labels)):
        gph1 = generate_graph(df1 , ei1 , labels ,  train_index , test_index)
        gph2 = generate_graph(df2 , ei2 , labels , train_index , test_index)
        gph3 = generate_graph(df3 , ei3 , labels , train_index , test_index)
        
        
        model = MOGONET(
            [gph1.x.size()[1] , gph2.x.size()[1] , gph3.x.size()[1]] ,
            5 , True)
        optimizer = optim.Adam(model.parameters() , lr = 0.01)
        criterion = nn.CrossEntropyLoss()
        
        def train(epoch):
            model.train()
            optimizer.zero_grad()
            output , output_gnn1 , output_gnn2 , output_gnn3  = model(
                gph1.x , gph2.x , gph3.x , 
                gph1.edge_index , gph2.edge_index , gph3.edge_index
            )
            loss  = criterion(output[train_index] , labels[train_index])
            loss_gnn1 = criterion(output_gnn1[train_index] , labels[train_index])
            loss_gnn2 = criterion(output_gnn2[train_index] , labels[train_index])
            loss_gnn3 = criterion(output_gnn3[train_index] , labels[train_index])
            #print(output.shape)
            #print(labels.shape)
            #print(len(train_index))
            total_loss = loss + loss_gnn1 + loss_gnn2 + loss_gnn3
            accuracy = (output[train_index].argmax(1) == labels[train_index]).sum().item() / len(labels[train_index]) * 100
            
            # recall and precision
            f1_score = F1Score(task='multiclass' , num_classes=5)
            f1 = f1_score(output[train_index] , labels[train_index])
            
            # Caculate AUROC
            auroc = AUROC(task="multiclass" , num_classes=5)
            auc = auroc(output[train_index] , labels[train_index])
            
            total_loss.backward()
            optimizer.step()
            #print("Epoch: ", epoch , " Loss: ", loss.item() , " Accuracy: ", accuracy)
            return total_loss , accuracy , f1.item() , auc.item()
        
        def test(epoch):
            model.eval()
            output , _ , _ , _ = model(
                gph1.x , gph2.x , gph3.x , 
                gph1.edge_index , gph2.edge_index , gph3.edge_index
            )
            loss = criterion(output[test_index] , labels[test_index])
            accuracy = (output[test_index].argmax(1) == labels[test_index]).sum().item() / len(labels[test_index]) * 100
            
            # recall and precision
            f1_score = F1Score(task='multiclass' , num_classes=5)
            f1 = f1_score(output[test_index] , labels[test_index])
            
            # Caculate AUROC
            auroc = AUROC(task="multiclass" , num_classes=5)
            auc = auroc(output[test_index] , labels[test_index])
                
            #print("Epoch: ", epoch , " Loss: ", loss.item() , " Accuracy: ", accuracy)
            return loss , accuracy , f1.item() , auc.item()
        
        best_accuracy = 0
        best_epoch = 0
        f1_score = 0
        auc_score = 0
        
        pbar = tqdm(range(100))
        for epoch in range(100):
            loss , accuracy , f1 , auc = train(epoch)
            test_loss , test_accuracy , test_f1 , test_auc = test(epoch)
            pbar.set_description("Epoch: {} | Train Loss: {:.4f} | Train Accuracy: {:.2f} | Test Loss: {:.4f} | Test Accuracy: {:.2f}".format(epoch , loss.item() , accuracy , test_loss.item() , test_accuracy))
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                f1_score = test_f1
                auc_score = test_auc
                
            pbar.update(1)
        pbar.close()
        
        summary_df = summary_df.append({
            "Fold": i + 1,
            "Best Test Accuracy": best_accuracy,
            "F1 score": f1_score,
            'AUC': auc_score,
            "Test Epoch": best_epoch
        }, ignore_index=True)
    print(summary_df)


### Proposed method 
def generate_graph(df , header_name , labels,  integration='PPI'):
    
    if integration == 'PPI':
        
        
        # Get PPI data
        CACHED_PPI = r"PPI/ppi.parquet.gzip"
        
        if os.path.exists(CACHED_PPI):
            PPI_merge = pd.read_parquet(CACHED_PPI)
        else:
            PPI_filepath = r"PPI/9606.protein.links.v12.0.txt"
            PPI_annotation = r"PPI/9606.protein.info.v12.0.txt"

            ## Read PPI Annotation file 
            PPI_annotation = pd.read_csv(PPI_annotation , sep="\t")
            # print(PPI_annotation.head())

            ## Read PPI file
            PPI = pd.read_csv(PPI_filepath , sep=" ")
            # print(PPI.head())

            # convert protein id to gene name 
            PPI_merge = pd.merge(PPI , PPI_annotation , left_on="protein1" , right_on="#string_protein_id", how="left")
            PPI_merge.drop(columns=["#string_protein_id" , "protein_size" , "annotation"] , inplace=True , axis=1)
            PPI_merge.rename(columns={"preferred_name":"protein1_name"} , inplace=True)

            PPI_merge = pd.merge(PPI_merge , PPI_annotation , left_on="protein2" , right_on="#string_protein_id", how="left")
            PPI_merge.drop(columns=["#string_protein_id" , "protein_size" , "annotation"] , inplace=True , axis=1)
            PPI_merge.rename(columns={"preferred_name":"protein2_name"} , inplace=True)
            
            PPI_merge.to_parquet(CACHED_PPI , compression="gzip")
        
        ## Filter PPI Score > threshold (default : 600)
        PPI_merge = PPI_merge[PPI_merge["combined_score"] > 700]
        
        GENE_NAME = [x.split("|")[0] for x in header_name[0].tolist()]
        edge_index = [[], []]
        edge_attr = []
        i = 0 
        
        pbar = tqdm(total=len(PPI_merge))
        pbar.set_description("Generate graph edge | Number of edges: {}".format(i))
        for idx  , row in PPI_merge.iterrows():
            if row["protein1_name"] in GENE_NAME and row["protein2_name"] in GENE_NAME:
                source_index = GENE_NAME.index(row["protein1_name"])
                target_index = GENE_NAME.index(row["protein2_name"])
                
                edge_index[0].append(source_index)
                edge_index[1].append(target_index)
                edge_index[0].append(target_index)
                edge_index[1].append(source_index)
                edge_attr.extend([row["combined_score"]] * 2)
        
                i += 2
                pbar.set_description("Number of edges: {}".format(i))
            pbar.update(1)
        pbar.close()
        
        ## Convert edge_index and edge_attr to tensor
        edge_index = torch.tensor(edge_index , dtype=torch.long)
        edge_attr = torch.tensor(edge_attr , dtype=torch.float).unsqueeze(1)
        
        assert edge_index.shape[1] == edge_attr.shape[0] , "Number of edges and edge attributes must be equal"
        
        graph_data  = []
        
        pbar = tqdm(total=len(df))
        pbar.set_description("Generate graph data")
        for idx , [ index , row ] in enumerate(df.iterrows()):
            x = torch.tensor(row.values , dtype=torch.float).unsqueeze(1)
            assert x.shape[1] == 1 , "Feature dimension must be 1"
            graph = Data(x=x , edge_index=edge_index , edge_attr=edge_attr , y=torch.tensor(labels[idx] , dtype=torch.long))
            graph_data.append(graph)
            pbar.update(1)
        pbar.close()
        
        return graph_data

class GCN(torch.nn.Module):
    def __init__(self , in_channels , hidden_channels , out_channels , lin = True):
        super(GCN , self).__init__()
        self.conv1 = geom_nn.DenseSAGEConv(in_channels , hidden_channels )
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = geom_nn.DenseSAGEConv(hidden_channels , hidden_channels  )
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = geom_nn.DenseSAGEConv(hidden_channels , out_channels  )
        self.batch_norm3 = torch.nn.BatchNorm1d(out_channels)
    
        if lin is True:
            self.lin = nn.Linear(2 * hidden_channels + out_channels , out_channels)
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
        #print(x0)
        #print(edge_index)
        x1 = self.bn(1 , self.conv1(x0 , edge_index , mask).relu())
        x2 = self.bn(2 , self.conv2(x1 , edge_index , mask).relu())
        x3 = self.bn(3 , self.conv3(x2 , edge_index , mask).relu())
        
        #print(x1.size())
        #print(x2.size())
        #print(x3.size())
        #x = torch.cat([x1 , x2 , x3] , dim=-1)
        x = torch.cat([x1 , x2 , x3] , dim=-1)
        
        if self.lin is not None:
            x = self.lin(x).relu()
            
        return x # output dimension 

class DiffPool(torch.nn.Module):
    def __init__(self , num_features , num_classes , max_nodes=150):
        super(DiffPool , self).__init__()
        hidden_channels = 16
        
        num_nodes = ceil(0.25 * max_nodes) # maximum of cluster nodes after first layer of pooling 
        self.gnn1_pool = GCN(num_features , hidden_channels , num_nodes)
        self.gnn1_embed = GCN(num_features , hidden_channels , hidden_channels , lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GCN(3*hidden_channels , hidden_channels , num_nodes , lin=False)
        self.gnn2_embed = GCN(3*hidden_channels , hidden_channels , hidden_channels , lin=False)
        
        self.gnn3_embed = GCN(3*hidden_channels , hidden_channels , hidden_channels , lin=False)
        
        self.lin1 = torch.nn.Linear(3*hidden_channels , hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels , num_classes)
        
    def forward(self , x , edge_index , mask=None):        
        batch_size , node_size , _ = x.size()
        #print("Batch_size: {} | Node_size: {} | Feature_Dimension: {}".format(batch_size , node_size , _) )
        adj = to_dense_adj(edge_index[0]).expand(batch_size , node_size , node_size )
        #print(adj.size())
        
        s = self.gnn1_pool(x , adj , mask) # size => ( batch_size , num_nodes)
        x = self.gnn1_embed(x , adj , mask) # size => ( batch_size , num_nodes)
        
        # convert edge_index to dense adjacency matrix
        
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

def train_graph_diffpool():
    
    # read labels
    labels = os.path.join(base_path, "labels_tr.csv")
    df_labels = read_features_file(labels)
    
    # Read feature 1 file 
    # feature1 = os.path.join(base_path, "1_tr.csv")
    # df1 = read_features_file(feature1)
    # name1 = os.path.join(base_path, "1_featname.csv")
    # df1_header = read_features_file(name1)
    # gp1 = generate_graph(df1 , df1_header , df_labels[0].tolist())
    
    # Read feature 2 file 
    feature2 = os.path.join(base_path, "2_tr.csv")
    df2 = read_features_file(feature2)
    name2 = os.path.join(base_path, "2_featname.csv")
    df2_header = read_features_file(name2)
    gp2 = generate_graph(df2 , df2_header , df_labels[0].tolist())
    batch_size = 30 
    
    summary_df = pd.DataFrame(columns=['fold' , 'train_loss' , 'train_acc' , 'best_epoch' , 'best_test_accuracy' , 'auc' , 'f1'])
    kf = StratifiedKFold(n_splits=10 , shuffle=True)
    
    for i , (train_index , test_index) in enumerate(kf.split(df2.values , df_labels[0].values)):
        
        # split data into train and test set
        train_data = [gp2[idx] for idx in train_index]
        test_data = [gp2[idx] for idx in test_index]
        
        train_loader = DenseDataLoader(train_data , batch_size=batch_size , shuffle=True)
        test_loader = DenseDataLoader(test_data , batch_size=len(test_index))
    
        model = DiffPool(1 , 5 , 50)
        optimizer = optim.Adam(model.parameters() , lr=0.01)
        
        model.train()
        
        def train(epoch):
            model.train()
            loss_all = 0
            correct_pred = 0
            
            for data in train_loader:
                optimizer.zero_grad()
                output  , _  , _ = model(data.x , data.edge_index , None)
                loss = F.nll_loss(output, data.y.unsqueeze(0).view(-1))
                loss.backward()
                loss_all += data.y.size(0) * float(loss)
                
                correct_pred += sum((data.y.unsqueeze(0).view(-1) == output.argmax(dim=1)).type(torch.int)).item()
                optimizer.step()
        
            return loss_all / len(train_index) , correct_pred / len(train_index) * 100
        
        def test(epoch):
            model.eval()
            loss_all = 0
            correct_pred = 0
            
            for data in test_loader:
                output  , _  , _ = model(data.x , data.edge_index , None)
                loss = F.nll_loss(output, data.y.unsqueeze(0).view(-1))
                loss_all += data.y.size(0) * float(loss)
                
                
                correct_pred += sum((data.y.unsqueeze(0).view(-1) == output.argmax(dim=1)).type(torch.int)).item()

                # recall and precision
                f1_score = F1Score(task='multiclass' , num_classes=5 , average='macro')
                f1 = f1_score(output , data.y)
                # Caculate AUROC
                auroc = AUROC(task="multiclass" , num_classes=5)
                auc = auroc(output , data.y)
            return loss_all / len(test_index) , correct_pred / len(test_index) * 100 , f1.item() , auc.item()
                
        best_test_accuracy = 0
        best_epoch = 0
        best_auc = 0
        best_f1 = 0

        max_epoch = 200
        pbar = tqdm(total=max_epoch)
        pbar.set_description("Epoch: {} | Loss: {:.4f} | Acc : {:.2f} | Test Loss : {:.4f} | Test Acc: {:.2f}".format(0 , 0 , 0 , 0 , 0))
        for epoch in range(0 , max_epoch):
            loss , acc = train(epoch)
            loss_test , acc_test , f1 , auc = test(epoch)
            pbar.update(1)
            pbar.set_description("Epoch: {} | Loss: {:.4f} | Acc : {:.2f} | Test Loss : {:.4f} | Test Acc: {:.2f}".format(epoch , loss , acc , loss_test , acc_test))

            if acc_test > best_test_accuracy:
                best_test_accuracy = acc_test
                best_epoch = epoch
                best_auc = auc
                best_f1 = f1
            #     torch.save(model.state_dict() , "checkpoints/model_fold_{}.pt".format(i))
        pbar.close()
        print({
            'fold':i , 
            'train_loss':loss , 
            'train_acc':acc , 
            'best_epoch':best_epoch , 
            'best_test_accuracy':best_test_accuracy , 
            'auc': best_auc, 
            'f1': best_f1
        })
        # summary_df = summary_df.append({
        #     'fold':i , 
        #     'train_loss':loss , 
        #     'train_acc':acc , 
        #     'best_epoch':best_epoch , 
        #     'best_test_accuracy':best_test_accuracy , 
        #     'auc': best_auc, 
        #     'f1': best_f1
        # } , ignore_index=True)
        
    print(summary_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mogonet', help='Model name')
    
    args = parser.parse_args()
    
    if args.model == 'mogonet':
        train_mogonet()
    else: 
        train_graph_diffpool()
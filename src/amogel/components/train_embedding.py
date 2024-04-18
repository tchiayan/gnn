from torch_geometric.nn import GAE
import torch
from amogel import logger
from amogel.model.graph_autoencoder import GCNEncoder
from torch_geometric.utils import train_test_split_edges , to_undirected
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from amogel.entity.config_entity import EmbeddingTrainerConfig
import os
import seaborn as sns
import matplotlib.pyplot as plt

class EmbeddingTrainer():
    
    def __init__(self , config:EmbeddingTrainerConfig ,  out_channels , epochs , lr , omic_type:int , dataset:str):
        
        self.epochs = epochs
        self.config = config
        self.dataset = dataset 
        self.omic_type = omic_type 
        
        # load sample data 
        data_filepath = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"{omic_type}_tr.csv")
        df_data = pd.read_csv(data_filepath , header=None)
        self.num_features = df_data.shape[1]
        self.num_sample = df_data.shape[0]
        
        # create model & optimizer
        self.model = GAE(GCNEncoder(self.num_sample , out_channels))
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr = lr)
        
        # load ac filepath 
        ac_file_path = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"ac_rule_{omic_type}.tsv")
        df_ac = pd.read_csv(ac_file_path , sep="\t" , header=None)
        df_ac.columns = ['class' , 'support', 'confidence' , 'antecedents', 'interestingness']
        
        # select top 1000 rules for each class with highest interestingness 
        df_ac_filtered = df_ac.groupby(["class"]).apply(lambda x: x.nlargest(1000 , 'interestingness')).reset_index(drop=True) 
        
        # build adjacency matrics
        logger.info(f"Build adjacency matrix for omic type: {omic_type}")
        adjacancy_matrix = torch.zeros((self.num_features , self.num_features))
        with tqdm(total=df_ac_filtered.shape[0]) as pbar:
            for index , row in df_ac_filtered.iterrows():
                node_idx = [int(x.split(":")[0]) for x  in row['antecedents'].split(',')]
                
                vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)]) # generate possible edges pair given the list of node index
                adjacancy_matrix[vector_idx[:,0] , vector_idx[:,1]] += 1
                adjacancy_matrix[vector_idx[:,1] , vector_idx[:,0]] += 1 # it is undirected graph
                
                pbar.update(1)
        
        # normalize adjacency matrix 
        normalized_adjacancy_matrix = adjacancy_matrix / adjacancy_matrix.max() 
        
        # plot adjacency matrix and save to root_dir
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        plt.figure(figsize=(10,10))
        sns.heatmap(normalized_adjacancy_matrix)
        plt.savefig(os.path.join(self.config.root_dir , self.dataset , f"{omic_type}_adjacency_matrix.png"))
        
        # build pytorch geometric data
        edge_coo = self.symmetric_matrix_to_coo(normalized_adjacancy_matrix , 0.5)
        geom_data = self.coo_to_pyg_data(edge_coo , torch.tensor(df_data.T.values , dtype=torch.float32))
        
        # split data
        geom_data.train_mask = geom_data.val_mask = geom_data.test_mask = None
        self.data = train_test_split_edges(geom_data)
        
        # move to GPU (if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.data.to(device)
        self.model = self.model.to(device)
        
    
    def train(self):
        
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.data.x , self.data.train_pos_edge_index)
        loss = self.model.recon_loss(z , self.data.train_pos_edge_index)
        loss.backward()
        
        self.optimizer.step()
        return float(loss)
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(self.data.x , self.data.train_pos_edge_index)
        return self.model.test(z , self.data.test_pos_edge_index , self.data.test_neg_edge_index)
    
    def run(self):
        logger.info(f"Learn embedding for {self.omic_type} omic type")
        for epoch in range(self.epochs):
            loss = self.train()
            auc , ap = self.test()
            if epoch % 10 == 0:
                logger.info(f"Epoch: {epoch+1} | Loss: {loss} | AUC: {auc} | AP: {ap}")
                
        self.loss = loss 
        self.auc = auc 
        self.ap = ap
    
    @staticmethod
    def coo_to_pyg_data(coo_matrix , node_features):
        values = torch.FloatTensor(coo_matrix.data).unsqueeze(1)
        indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
        size = torch.Size(coo_matrix.shape)

        indices , values = to_undirected(indices , values)
        
        return Data(x=node_features, edge_index=indices, edge_attr=values, num_nodes=size[0])
    
    @staticmethod 
    def symmetric_matrix_to_coo(matrix , threshold):
        # Find the nonzero entries in the upper triangle (including the main diagonal)
        rows, cols = np.triu_indices_from(matrix, k=0)

        # Extract the corresponding values from the upper triangle
        data = matrix[rows, cols]

        # Filter entries based on the threshold
        mask = data >= threshold
        rows = rows[mask]
        cols = cols[mask]
        data = data[mask]
        
        # Create a COO matrix
        coo = coo_matrix((data, (rows, cols)), shape=matrix.shape)

        return coo
    
    def save_embedding(self):
        # create directory if not exist
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        
        embedding_filepath = os.path.join(self.config.root_dir , self.dataset , f"{self.omic_type}_embedding.pt")
        logger.info(f"Save embedding: {embedding_filepath}")
        torch.save(self.model.encode(self.data.x , self.data.train_pos_edge_index) , embedding_filepath)
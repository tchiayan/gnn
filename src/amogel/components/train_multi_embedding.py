from torch_geometric.nn import GAE , VGAE
import torch
from amogel import logger
from amogel.model.graph_autoencoder import GCNEncoder , VariationalGCNEncoder , Pooling
from torch_geometric.utils import train_test_split_edges , to_undirected
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from amogel.entity.config_entity import EncoderTrainingConfig
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report
from torchmetrics import Accuracy
from torch_geometric.loader import DataLoader 

# suppress warning
import warnings
warnings.filterwarnings("ignore")

MODEL = {
    'GAE': GAE,
    'VGAE': VGAE
}

ENCODER = {
    'GAE': GCNEncoder,
    'VGAE': VariationalGCNEncoder
}

class MultiEmbeddingTrainer():
    
    def __init__(self , config:EncoderTrainingConfig ,  out_channels , epochs , lr , omic_type:int , dataset:str):
        
        self.epochs = epochs
        self.config = config
        self.dataset = dataset 
        self.omic_type = omic_type 
        
        # load sample data 
        data_filepath = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"{omic_type}_tr.csv")
        df_data = pd.read_csv(data_filepath , header=None)
        self.num_features = df_data.shape[1]
        self.num_sample = df_data.shape[0]
        
        # discretize the data
        kbin = KBinsDiscretizer(n_bins=2 , encode='ordinal' , strategy='quantile')
        df_data = pd.DataFrame(kbin.fit_transform(df_data))
        
        # load sample label 
        label_filepath = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"labels_tr.csv")
        df_label = pd.read_csv(label_filepath , header=None , names=['label'])
        
        # create model & optimizer
        self.model = MODEL[config.model](ENCODER[config.model](1 , out_channels)) # GAE(GCNEncoder(1 , out_channels))
        self.class_model = Pooling(out_channels , df_label['label'].nunique())
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr = lr)
        
        # load ac filepath 
        ac_file_path = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"ac_rule_{omic_type}.tsv")
        df_ac = pd.read_csv(ac_file_path , sep="\t" , header=None)
        df_ac.columns = ['class' , 'support', 'confidence' , 'antecedents', 'interestingness']
        
        # select top 1000 rules for each class with highest interestingness 
        # df_ac_filtered = df_ac.groupby(["class"]).apply(lambda x: x.nlargest(1000 , 'interestingness')).reset_index(drop=True) 
        
        # build adjacency matrics
        # logger.info(f"Build adjacency matrix for omic type: {omic_type}")
        # adjacancy_matrix = torch.zeros((self.num_features , self.num_features))
        # with tqdm(total=df_ac_filtered.shape[0]) as pbar:
        #     for index , row in df_ac_filtered.iterrows():
        #         node_idx = [int(x.split(":")[0]) for x  in row['antecedents'].split(',')]
                
        #         vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)]) # generate possible edges pair given the list of node index
        #         adjacancy_matrix[vector_idx[:,0] , vector_idx[:,1]] += 1
        #         adjacancy_matrix[vector_idx[:,1] , vector_idx[:,0]] += 1 # it is undirected graph
                
        #         pbar.update(1)
        
        synthetic_adjacancy_dict = self._generate_synthetic_graph(normalize_method='binary')
        # normalize adjacency matrix 
        # normalized_adjacancy_matrix = adjacancy_matrix / adjacancy_matrix.max() 
        
        # Duplicated from training_embedding
        # plot adjacency matrix and save to root_dir
        # os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        # plt.figure(figsize=(10,10))
        # sns.heatmap(normalized_adjacancy_matrix)
        # plt.savefig(os.path.join(self.config.root_dir , self.dataset , f"{omic_type}_adjacency_matrix.png"))
        
        # build pytorch geometric data
        # edge_coo = self.symmetric_matrix_to_coo(normalized_adjacancy_matrix , 0.5)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # df_data.T => # number of node x number of feature (sample)
        self.graphs = []
        
        logger.info(f"Generate synthetic graph for omic type: {omic_type}")
        
        with tqdm(total=len(df_data)) as pbar:
            for idx , row in df_data.iterrows():
                
                target_label = df_label.loc[idx].values[0]
                #synthetic_adjacancy_matrix = synthetic_adjacancy_dict[target_label]
                synthetic_adjacancy_matrix = self._generate_synthetic_network(row.values)
                
                edge_coo = self.symmetric_matrix_to_coo(synthetic_adjacancy_matrix , 0.5)
                geom_data = self.coo_to_pyg_data(edge_coo , torch.tensor(row.values , dtype=torch.float32 , device=device).unsqueeze(1) , y=torch.tensor(target_label , dtype=torch.long , device=device))
                
                # split data 
                geom_data.train_mask = geom_data.val_mask = geom_data.test_mask = None
                geom_data = train_test_split_edges(geom_data)
                
                # move to GPU (if available)
                geom_data = geom_data.to(device)
                self.graphs.append(geom_data)
                pbar.update(1)
            
        self.loader = DataLoader(self.graphs , batch_size=20 , shuffle=True)
            
        self.model = self.model.to(device)
        self.class_model = self.class_model.to(device)
    
    def _generate_synthetic_network(self , expression_values  ,topk=50 ):
        
        ac_file_path = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"ac_rule_{self.omic_type}.tsv")
        df_ac = pd.read_csv(ac_file_path , sep="\t" , header=None)
        df_ac.columns = ['class' , 'support', 'confidence' , 'antecedents', 'interestingness']
        
        df_ac_filtered = df_ac.groupby(["class"]).apply(lambda x: x.nlargest(topk , 'interestingness')).reset_index(drop=True)
        
        edge_matrix = torch.zeros(self.num_features , self.num_features)
        expression_set = set([f"{idx}:{value}" for idx , value in enumerate(expression_values)])
        for idx , row in df_ac_filtered.iterrows():
            rule_set = set([x for x in row['antecedents'].split(',')])
            
            if rule_set.issubset(expression_set):
                node_idx = [int(x.split(":")[0]) for x in row['antecedents'].split(',')]
                vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)])
                edge_matrix[vector_idx[:,0] , vector_idx[:,1]] += 1
                edge_matrix[vector_idx[:,1] , vector_idx[:,0]] += 1
        
        return edge_matrix
            
        
    def _generate_synthetic_graph(self , topk=50 , normalize=True , normalize_method='max'):
        # load ac filepath 
        ac_file_path = os.path.join(self.config.data_preprocessing_dir , self.dataset , f"ac_rule_{self.omic_type}.tsv")
        df_ac = pd.read_csv(ac_file_path , sep="\t" , header=None)
        df_ac.columns = ['class' , 'support', 'confidence' , 'antecedents', 'interestingness']
        
        # select top 1000 rules for each class with highest interestingness 
        df_ac_filtered = df_ac.groupby(["class"]).apply(lambda x: x.nlargest(topk , 'interestingness')).reset_index(drop=True) 
        
        # generate synthetic graph for each class 
        unique_class = df_ac_filtered['class'].unique()
        synthetic_tensor = {}
        
        logger.info(f"Generate synthetic graph for omic type: {self.omic_type}")
        with tqdm(total=len(unique_class)) as pbar:
            for class_label in unique_class:
                
                knowledge_tensor = torch.zeros(self.num_features, self.num_features)
                
                for idx , row in df_ac_filtered[df_ac_filtered['class'] == class_label].iterrows():
                    node_idx = [int(x.split(":")[0]) for x in row['antecedents'].split(',')]
                    vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)])
                    knowledge_tensor[vector_idx[:,0] , vector_idx[:,1]] += 1
                    knowledge_tensor[vector_idx[:,1] , vector_idx[:,0]] += 1
                
                if normalize:
                    if normalize_method == 'max':
                        # normalize the numpy [0, 1]
                        knowledge_tensor = knowledge_tensor / knowledge_tensor.max()
                    elif normalize_method == 'binary':
                        knowledge_tensor = (knowledge_tensor > 0).float()
                
                # change nan to 0
                knowledge_tensor[torch.isnan(knowledge_tensor)] = 0
                
                synthetic_tensor[int(class_label)] = knowledge_tensor
                pbar.update(1)
                
        ## sort the tensor based on the key
        synthetic_tensor = dict(sorted(synthetic_tensor.items()))
        
        return synthetic_tensor
                
    def train(self):
        
        self.model.train()
        self.class_model.train()
        self.optimizer.zero_grad()
        
        criteron = torch.nn.CrossEntropyLoss()
        accuracy = Accuracy(task='multiclass' , num_classes=5)
        
        prediction = []
        actual = []
        # with tqdm(total=len(self.graphs)) as pbar:
        #     pbar.set_description("Training")
        #     for data in self.graphs:
        #         z = self.model.encode(data.x , data.train_pos_edge_index)
        #         loss = self.model.recon_loss(z , data.train_pos_edge_index)
        #         loss.backward()
        #         pbar.update(1)
        with tqdm(total=len(self.loader)) as pbar:
            for batch in self.loader:
                z = self.model.encode(batch.x , batch.train_pos_edge_index)
                loss = self.model.recon_loss(z , batch.train_pos_edge_index)
                
                pbar.update(1)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # prediction  = torch.stack(prediction).to(device)
        # actual = torch.stack(actual).to(device)
        # prediction = torch.argmax(prediction , dim=1)
        # report = classification_report(actual.cpu().numpy() , prediction.cpu().numpy())
        # print(report)
        # acc = prediction.eq(actual).sum().item() / len(actual)
        #print(prediction)
        #print(actual)
        #acc = accuracy(prediction , actual).to(device).item()
            
        self.optimizer.step()
        return loss , None
    
    def test(self):
        self.model.eval()
        self.class_model.eval()
        with torch.no_grad():
            total_auc = 0
            total_ap = 0
            # prediction = []
            # actual = []
            
            # acc = Accuracy(task='multiclass' , num_classes=5)
            with tqdm(total=len(self.loader)) as pbar:
                for batch in self.loader:
                    z = self.model.encode(batch.x , batch.train_pos_edge_index)
                    auc , ap = self.model.test(z , batch.test_pos_edge_index , batch.test_neg_edge_index)
                    total_auc += auc
                    total_ap += ap
                    
                    pbar.update(1)
                    
                # for data in self.graphs:
                #     z = self.model.encode(batch.x , data.train_pos_edge_index)
                #     auc , ap = self.model.test(z , data.test_pos_edge_index , data.test_neg_edge_index)
                #     total_auc += auc
                #     total_ap += ap
            
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # prediction = torch.stack(prediction).to(device)
        # actual = torch.stack(actual).to(device)
        # prediction = torch.argmax(prediction , dim=1)
        # acc = prediction.eq(actual).sum().item() / len(actual)
        
        return total_auc / len(self.graphs) , total_ap / len(self.graphs) , None
    
    def run(self):
        logger.info(f"Learn multi embedding encoder for {self.omic_type} omic type")
        for epoch in range(self.epochs+1):
            loss , train_acc = self.train()
            auc , ap , test_acc = self.test()
            if epoch % self.config.print_interval == 0:
                logger.info(f"Epoch: {epoch}\t| Loss: {loss}\t| AUC: {auc}\t| AP: {ap}")
                
        self.loss = loss 
        self.auc = auc 
        self.ap = ap
    
    @staticmethod
    def coo_to_pyg_data(coo_matrix , node_features , y):
        values = torch.FloatTensor(coo_matrix.data).unsqueeze(1)
        indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))
        size = torch.Size(coo_matrix.shape)

        indices , values = to_undirected(indices , values)
        
        return Data(x=node_features, edge_index=indices, edge_attr=values, y=y , num_nodes=size[0])
    
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
    
    def save_model(self):
        # create directory if not exist
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        
        embedding_filepath = os.path.join(self.config.root_dir , self.dataset , f"encoder_omic_{self.omic_type}_model.pth")
        logger.info(f"Save model: {embedding_filepath}")
        
        torch.save(self.model.state_dict() , embedding_filepath)
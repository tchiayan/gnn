### Generate prior knowledge graph 
from amogel import logger 
from amogel.entity.config_entity import KnowledgeGraphConfig
import os
import torch 
import pandas as pd
from tqdm import tqdm
from amogel.utils.common import symmetric_matrix_to_coo , coo_to_pyg_data

class KnowledgeGraph():
    
    def __init__(self , config: KnowledgeGraphConfig , omic_type: int , dataset: str):
        self.config = config 
        self.omic_type = omic_type 
        self.dataset = dataset
        self.embedding = self.__load_embedding()
        self.feature_names = self.__load_feature_name(extract_gene_name=(omic_type == 1))
        self.train_data = self.__load_train_data()
        self.test_data = self.__load_test_data()
        self.train_label = self.__load_train_label()
        self.test_label = self.__load_test_label()
        self.related_protein = self.__load_ppi()
    
    def __load_train_data(self):
        
        train_datapath = os.path.join(self.config.data_dir , self.dataset , f"{self.omic_type}_tr.csv")
        
        if not os.path.exists(train_datapath):
            raise FileNotFoundError(f"Data file not found at {train_datapath}")
        
        logger.info(f"Loading data : {train_datapath}")
        df_train = pd.read_csv(train_datapath)
        
        return df_train
    
    def __load_test_data(self):
        
        test_datapath = os.path.join(self.config.data_dir , self.dataset , f"{self.omic_type}_te.csv")

        if not os.path.exists(test_datapath):
            raise FileNotFoundError(f"Data file not found at {test_datapath}")
        
        logger.info(f"Loading data : {test_datapath}")
        df_test = pd.read_csv(test_datapath)

        return df_test
    
    def __load_train_label(self):
        
        train_label_path = os.path.join(self.config.data_dir , self.dataset , f"labels_tr.csv")
        
        if not os.path.exists(train_label_path):
            raise FileNotFoundError(f"Data file not found at {train_label_path}")
        
        logger.info(f"Loading label data : {train_label_path}")
        df_label = pd.read_csv(train_label_path)
        
        return df_label
    
    def __load_test_label(self):
        
        test_label_path = os.path.join(self.config.data_dir , self.dataset , f"labels_te.csv")
        
        if not os.path.exists(test_label_path):
            raise FileNotFoundError(f"Data file not found at {test_label_path}")
        
        logger.info(f"Loading label data : {test_label_path}")
        df_label = pd.read_csv(test_label_path)
        
        return df_label
    
    def __load_feature_name(self , extract_gene_name = False):
        
        data_filepath = os.path.join(self.config.data_dir , self.dataset , f"{self.omic_type}_featname.csv")
        
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Data file not found at {data_filepath}")
        
        logger.info(f"Loading feature name data : {data_filepath}")
        df_omic = pd.read_csv(data_filepath , header=None)
        
        if extract_gene_name:
            df_omic['gene_name'] = df_omic[0].apply(lambda x: x.split("|")[0])
        else: 
            df_omic.rename({0:"gene_name"}, axis=1, inplace=True)
        
        df_omic = df_omic.index.to_frame(name="gene_idx").join(df_omic)

        return df_omic 
                    
    def __load_embedding(self):
        
        
        emb_path = os.path.join(self.config.embedding_dir, self.dataset , f"{self.omic_type}_embedding.pt")
        
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding file not found at {emb_path}")
        
        
        logger.info(f"Loading embedding : {emb_path}")
        emb = torch.load(emb_path)
        
        return emb
    
    def __load_ppi(self):
        
        logger.info("Loading PPI data")
        
        ppi_info_path = os.path.join(self.config.ppi_dir, f"protein_info.txt")
        ppi_link_path = os.path.join(self.config.ppi_dir, f"protein_links.txt")
        
        if not os.path.exists(ppi_info_path):
            raise FileNotFoundError(f"PPI info file not found at {ppi_info_path}")
        
        if not os.path.exists(ppi_link_path):
            raise FileNotFoundError(f"PPI link file not found at {ppi_link_path}")
        
        logger.info(f"Loading PPI info : {ppi_info_path} , PPI link : {ppi_link_path}")
        df_protein = pd.read_csv(ppi_info_path, sep='\t')
        df_protein_link = pd.read_csv(ppi_link_path, sep='\s' , engine="python")
        
        df_protein_merged = pd.merge(df_protein_link, df_protein[['#string_protein_id','preferred_name']], left_on="protein1", right_on="#string_protein_id")
        df_protein_merged.rename(columns={"preferred_name":"protein1_name"}, inplace=True)

        df_protein_merged = pd.merge(df_protein_merged, df_protein[['#string_protein_id','preferred_name']], left_on="protein2", right_on="#string_protein_id")
        df_protein_merged.rename(columns={"preferred_name":"protein2_name"}, inplace=True)

        # drop columns
        df_protein_merged.drop(columns=["#string_protein_id_x", "#string_protein_id_y", "protein1" , "protein2"], inplace=True)
        df_protein_merged.head()
        

        df_protein_merged = df_protein_merged.merge(self.feature_names[['gene_idx' , 'gene_name']] , left_on="protein1_name", right_on="gene_name" , how="left")
        df_protein_merged.rename(columns={"gene_idx":"gene1_idx"}, inplace=True)

        df_protein_merged = df_protein_merged.merge(self.feature_names[['gene_idx' , 'gene_name']] , left_on="protein2_name", right_on="gene_name" , how="left")
        df_protein_merged.rename(columns={"gene_idx":"gene2_idx"}, inplace=True)

        df_protein_merged.drop(columns=["gene_name_x", "gene_name_y"], inplace=True)
        
        
        # filter rows with only gene1_idx and gene2_idx
        df_filter_protein = df_protein_merged[df_protein_merged['gene1_idx'].notnull()][df_protein_merged['gene2_idx'].notnull()]

        return df_filter_protein

    def generate_knowledge_graph(self):
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0] 
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        logger.info("Generating Knowledge Tensor")
        with tqdm(total=self.related_protein.shape[0]) as pbar: 
            for idx, row in self.related_protein.iterrows():
                knowledge_tensor[int(row['gene1_idx']) , int(row['gene2_idx'])] += 1
                #knowledge_tensor[int(row['gene2_idx']) , int(row['gene1_idx'])] += 1
                pbar.update(1)
            
        coo_matrix = symmetric_matrix_to_coo(knowledge_tensor.numpy() , 1)
        graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=self.embedding)
        
        logger.info("Generating Assembled Knowledge Graph [Training Graph]")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32).unsqueeze(-1)
                node_embedding = torch.concat([torch_sample , self.embedding] , dim=-1)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=node_embedding , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) )
                training_graphs.append(graph)
                pbar.update(1)
                
        
        logger.info("Generating Assembled Knowledge Graph [Testing Graph]")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar: 
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32).unsqueeze(-1)
                node_embedding = torch.concat([torch_sample , self.embedding] , dim=-1)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=node_embedding , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) )
                testing_graphs.append(graph)
                pbar.update(1)
                
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_graphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_graphs_omic_{self.omic_type}.pt"))
        
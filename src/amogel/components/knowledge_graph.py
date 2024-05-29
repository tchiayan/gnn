### Generate prior knowledge graph 
from amogel import logger 
from amogel.entity.config_entity import KnowledgeGraphConfig
import os
from sklearn.preprocessing import KBinsDiscretizer
import torch 
from torch.nn.functional import one_hot
import pandas as pd
from tqdm import tqdm
from amogel.utils.common import symmetric_matrix_to_coo , coo_to_pyg_data , symmetric_matrix_to_pyg
import itertools
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
import numpy as np
from typing import List
import random
import pickle

class KnowledgeGraph():
    
    def __init__(self , config: KnowledgeGraphConfig , omic_type: int , dataset: str):
        self.config = config 
        os.makedirs(os.path.join(self.config.root_dir , dataset) , exist_ok=True)
        self.omic_type = omic_type 
        self.dataset = dataset
        # self.embedding = self.__load_embedding()
        self.feature_names = self.__load_feature_name(extract_gene_name=(omic_type == 1))
        self.train_data = self.__load_train_data()
        self.test_data = self.__load_test_data()
        self.train_label = self.__load_train_label()
        self.test_label = self.__load_test_label()
        #self.related_protein = self.__load_ppi()
        
        # self.ppi_graph_tensor = self.__generate_ppi_graph()
        # self.kegg_go_graph_tensor = self.__generate_kegg_go_graph()
        # self.synthetic_graph = self.__generate_synthetic_graph()
        # self.synthetic_graph_test = self.__generate_synthetic_test_graph()
        # self.kbin_model = self.__load_kbin_model()
        
        self.graph_topology_summary = []
        self.dataset_summary = []
        
    
    def summary(self):
        """Summary of the generated dataset"""
        
        avg_no_of_edges_ppi = self.__measure_graph(self.ppi_graph_tensor)
        avg_no_of_edges_kegg_go = self.__measure_graph(self.kegg_go_graph_tensor)
        avg_no_of_edges_synthetic = {k: self.__measure_graph(v) for k,v in self.synthetic_graph.items()}
        
        self.graph_topology_summary.extend([
            {"name":"PPI Graph" , "avg_no_of_edges":avg_no_of_edges_ppi},
            {"name":"KEGG_GO Graph" , "avg_no_of_edges":avg_no_of_edges_kegg_go},
        ])
        
        for k,v in avg_no_of_edges_synthetic.items():
            self.graph_topology_summary.append({"name":f"Synthetic Graph Class {k}" , "avg_no_of_edges":v})

        df = pd.DataFrame(self.graph_topology_summary)
        
        output_filepath = os.path.join(self.config.root_dir , self.dataset , f"graph_summary_omic_{self.omic_type}.csv")
        logger.info(f"Saving graph summary to {output_filepath}")
        df.to_csv(output_filepath , index=False)
        
        df = pd.DataFrame(self.dataset_summary)
        output_filepath = os.path.join(self.config.root_dir , self.dataset , f"dataset_summary_omic_{self.omic_type}.csv")
        logger.info(f"Saving dataset summary to {output_filepath}")
        df.to_csv(output_filepath , index=False)
        
    def __load_kbin_model(self):
        kbin_model_filepath = os.path.join(self.config.data_dir , self.dataset , f"kbins_{self.omic_type}.joblib")
        
        if not os.path.exists(kbin_model_filepath):
            raise FileNotFoundError(f"KBins model not found at {kbin_model_filepath}")
        
        logger.info(f"Loading KBins model : {kbin_model_filepath}")
        with open(kbin_model_filepath, 'rb') as file:
            kbin_model = pickle.load(file)
        
        return kbin_model
        
        # est = KBinsDiscretizer(n_bins=2 , encode='ordinal' , strategy='quantile')
        # est.fit(self.train_data)
        
        # transformed_result = est.transform(transform)
        # df = pd.DataFrame(transformed_result)
        # return df
    
    def __load_train_data(self):
        
        train_datapath = os.path.join(self.config.data_dir , self.dataset , f"{self.omic_type}_tr.csv")
        
        if not os.path.exists(train_datapath):
            raise FileNotFoundError(f"Data file not found at {train_datapath}")
        
        logger.info(f"Loading data : {train_datapath}")
        df_train = pd.read_csv(train_datapath , header=None)
        
        return df_train
    
    def __load_test_data(self):
        
        test_datapath = os.path.join(self.config.data_dir , self.dataset , f"{self.omic_type}_te.csv")

        if not os.path.exists(test_datapath):
            raise FileNotFoundError(f"Data file not found at {test_datapath}")
        
        logger.info(f"Loading data : {test_datapath}")
        df_test = pd.read_csv(test_datapath , header=None)

        return df_test
    
    def __load_train_label(self):
        
        train_label_path = os.path.join(self.config.data_dir , self.dataset , f"labels_tr.csv")
        
        if not os.path.exists(train_label_path):
            raise FileNotFoundError(f"Data file not found at {train_label_path}")
        
        logger.info(f"Loading label data : {train_label_path}")
        df_label = pd.read_csv(train_label_path , header=None)
        
        return df_label
    
    def __load_test_label(self):
        
        test_label_path = os.path.join(self.config.data_dir , self.dataset , f"labels_te.csv")
        
        if not os.path.exists(test_label_path):
            raise FileNotFoundError(f"Data file not found at {test_label_path}")
        
        logger.info(f"Loading label data : {test_label_path}")
        df_label = pd.read_csv(test_label_path , header=None)
        
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
        
        ppi_info_path = os.path.join(self.config.ppi_dir, f"protein_info.parquet.gzip")
        ppi_link_path = os.path.join(self.config.ppi_dir, f"protein_links.parquet.gzip")
        
        if not os.path.exists(ppi_info_path):
            raise FileNotFoundError(f"PPI info file not found at {ppi_info_path}")
        
        if not os.path.exists(ppi_link_path):
            raise FileNotFoundError(f"PPI link file not found at {ppi_link_path}")
        
        logger.info(f"Loading PPI info : {ppi_info_path} , PPI link : {ppi_link_path}")
        df_protein = pd.read_parquet(ppi_info_path)
        df_protein_link = pd.read_parquet(ppi_link_path)
        
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
        
        if self.config.combined_score > 0:
            df_filter_protein = df_filter_protein[df_filter_protein['combined_score'] >= self.config.combined_score]
            
        return df_filter_protein

    def __generate_ppi_graph(self , normalize_method='binary'):
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0] 
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        logger.info("Generating PPI Knowledge Tensor")
        self.related_protein = self.__load_ppi()
        with tqdm(total=self.related_protein.shape[0]) as pbar: 
            for idx, row in self.related_protein.iterrows():
                knowledge_tensor[int(row['gene1_idx']) , int(row['gene2_idx'])] += row['combined_score']
                knowledge_tensor[int(row['gene2_idx']) , int(row['gene1_idx'])] += row["combined_score"]
                pbar.update(1)
                
        if normalize_method == 'binary':
            knowledge_tensor = (knowledge_tensor > 0).float()
        elif normalize_method == 'max':
            knowledge_tensor = knowledge_tensor / knowledge_tensor.max()
            
        # save the knowledge tensor
        logger.info("Saving PPI Knowledge Tensor")
        torch.save(knowledge_tensor , os.path.join(self.config.root_dir , self.dataset , f"knowledge_ppi_{self.omic_type}.pt"))
        
        return knowledge_tensor 
    
    def __generate_kegg_go_graph(self , normalize_method='binary'):
        
        annotation_filepath = os.path.join("artifacts/data_ingestion/unzip" , f"{self.dataset}_kegg_go" , "consol_anno_chart.tsv")
        if not os.path.exists(annotation_filepath):
            raise FileNotFoundError(f"Annotation file not found at {annotation_filepath}")
        
        logger.info(f"Loading annotation data (KEGG pathway and GO) : {annotation_filepath}")
        annotation_df = pd.read_csv(annotation_filepath , sep="\t")[['Genes' , 'PValue']]
        annotation_df['Genes'] = annotation_df['Genes'].apply(lambda x: [float(n) for n in x.split(",")])
        
        feature_conversion_filepath = os.path.join("artifacts/data_ingestion/unzip" , f"{self.dataset}_kegg_go" , f"{self.omic_type}_featname_conversion.csv")
        if not os.path.exists(feature_conversion_filepath):
            raise FileNotFoundError(f"Feature conversion file not found at {feature_conversion_filepath}")
        
        logger.info(f"Loading feature conversion data : {feature_conversion_filepath}")
        feature_df = pd.read_csv(feature_conversion_filepath)
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0] 
        knowledge_tensor = torch.zeros(no_of_genes , no_of_genes)
        logger.info("Generating KEGG Pathway and GO Knowledge Tensor")
        with tqdm(total=annotation_df.shape[0]) as pbar:
            for idx , row in annotation_df.iterrows():
                gene_ids = row['Genes']
                #print(feature_1['gene id'])
                #print(gene_ids)
                #print(feature_1['gene id'].isin(gene_ids))
                gene_idx = feature_df[feature_df['gene id'].isin(gene_ids) ].index.to_list()
                #print(gene_idx)
                gene_numpy = np.array(list(itertools.product(gene_idx , gene_idx))) # number of edges , 2 (source and target)
                #print(gene_numpy)
                if gene_numpy.shape[0] > 0:
                    knowledge_tensor[gene_numpy[:,0] , gene_numpy[:,1]] += 1
                    
                pbar.update(1)
                
        if normalize_method == 'binary':
            knowledge_tensor = (knowledge_tensor > 0).float()
        elif normalize_method == 'max':
            knowledge_tensor = knowledge_tensor / knowledge_tensor.max()
        
        # save the knowledge tensor
        logger.info("Saving KEGG Pathway and GO Knowledge Tensor")
        torch.save(knowledge_tensor , os.path.join(self.config.root_dir , self.dataset , f"knowledge_kegg_go_{self.omic_type}.pt"))
        
        return knowledge_tensor 
    
    def __generate_synthetic_graph(self , topk = 50 , normalize=True , normalize_method='max' , distinct_set=False):
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        
        
        synthetic_rules_filepath = os.path.join(self.config.data_dir , self.dataset , f"ac_rule_{self.omic_type}.tsv" )
        
        if not os.path.exists(synthetic_rules_filepath):
            raise FileNotFoundError(f"Synthetic rules file not found at {synthetic_rules_filepath}")
        
        synthetic_df = pd.read_csv(synthetic_rules_filepath , sep='\t' , header=None)        
        synthetic_df.columns = ['class' , 'confidence', 'support' , 'antecedents', 'interestingness']
        
        synthetic_df_filtered = synthetic_df.groupby(["class"]).apply(lambda x : x.nlargest(topk , self.config.metric)).reset_index(drop=True)
        
        # generate synthetic graph for each class 
        unique_class = synthetic_df['class'].unique()
        synthetic_tensor = {}
        
        logger.info(f"Generating Synthetic Knowledge Tensor for {len(unique_class)} classes")
        
        class_summary = {}
        for label in unique_class: 
            filtered = synthetic_df_filtered[synthetic_df_filtered['class'] == label]
            class_summary[label] = {}
            for idx , row in filtered.iterrows():
                antecedents = row['antecedents'].split(",")
                for antecedent in antecedents:
                    if antecedent not in class_summary[label].keys():
                        class_summary[label][antecedent] = 1 
                    else:
                        class_summary[label][antecedent] += 1
        
        class_summary_distinct_set = {}
        for key in class_summary.keys():
            class_summary_distinct_set[key] = set([ k for k , v in class_summary[key].items() if v == self.config.topk])
            
        # find the distinct set compare with other class 
        testing_model = {}
        for key in class_summary_distinct_set.keys():
            class_distinct_set = class_summary_distinct_set[key]
            # for compared_key in class_summary_distinct_set.keys():
            #     if key != compared_key:
            #         distinct_set = distinct_set.difference(class_summary_distinct_set[compared_key])
                    
            testing_model[key] = [",".join(list(class_distinct_set))]
                
        # find the distinct set compare with other class
            
        with tqdm(total=len(unique_class)) as pbar:
            for label in unique_class:
                
                knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
                
                if distinct_set:
                    _iterrows = enumerate(testing_model[label])
                    for idx , row in _iterrows:
                        try:
                            node_idx = [int(x.split(":")[0]) for x in row.split(',')]
                            vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)])
                            knowledge_tensor[vector_idx[:,0] , vector_idx[:,1]] += 1
                            knowledge_tensor[vector_idx[:,1] , vector_idx[:,0]] += 1
                        except:
                            print(row , idx)
                            raise Exception(f"Error at parsing row [{idx}]")
                        
                else:
                    _iterrows = synthetic_df_filtered[synthetic_df_filtered['class'] == label].iterrows()

                    try:
                        for idx , row in _iterrows:
                            node_idx = [int(x.split(":")[0]) for x in row['antecedents'].split(',')]
                            vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)])
                            knowledge_tensor[vector_idx[:,0] , vector_idx[:,1]] += 1
                            knowledge_tensor[vector_idx[:,1] , vector_idx[:,0]] += 1
                    except:
                        print(row['antecedents'])
                        print(node_idx)
                        print(vector_idx)
                        raise Exception(f"Error at parsing row [{idx}]")
                
                if normalize:
                    if normalize_method == 'max':
                        # normalize the numpy [0, 1]
                        knowledge_tensor = knowledge_tensor / knowledge_tensor.max()
                    elif normalize_method == 'binary':
                        knowledge_tensor = (knowledge_tensor > 0).float()
                
                # change nan to 0
                knowledge_tensor[torch.isnan(knowledge_tensor)] = 0
                
                synthetic_tensor[int(label)] = knowledge_tensor
                pbar.update(1)
        
        ## sort the tensor based on the key
        synthetic_tensor = dict(sorted(synthetic_tensor.items()))
            
        logger.info("Saving Synthetic Knowledge Tensor")
        
        torch.save(synthetic_tensor , os.path.join(self.config.root_dir , self.dataset , f"knowledge_synthetic_{self.omic_type}.pt"))
        
        return synthetic_tensor
    
    def __generate_synthetic_test_graph(self , topk = 50, normalize=True , normalize_method='max'):
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        
        
        synthetic_rules_filepath = os.path.join(self.config.data_dir , self.dataset , f"ac_rule_{self.omic_type}.tsv" )
        
        if not os.path.exists(synthetic_rules_filepath):
            raise FileNotFoundError(f"Synthetic rules file not found at {synthetic_rules_filepath}")
        
        synthetic_df = pd.read_csv(synthetic_rules_filepath , sep='\t' , header=None)        
        synthetic_df.columns = ['class' , 'support', 'confidence' , 'antecedents', 'interestingness']
        
        synthetic_df_filtered = synthetic_df.groupby(["class"]).apply(lambda x : x.nlargest(topk , 'interestingness')).reset_index(drop=True)
        
        # generate synthetic graph for each class 
        unique_class = synthetic_df['class'].unique()
        synthetic_tensor = {}
        
        logger.info(f"Generating Synthetic Knowledge Tensor for {len(unique_class)} classes")
        with tqdm(total=len(unique_class)) as pbar:
            for label in unique_class:
                
                knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
                
                for idx , row in synthetic_df_filtered[synthetic_df_filtered['class'] == label].iterrows():
                    node_idx = [int(x.split(":")[0]) for x in row['antecedents'].split(',')]
                    vector_idx = np.array([x for x in itertools.combinations(node_idx , 2)])
                    knowledge_tensor[vector_idx[:,0] , vector_idx[:,1]] += 1
                    knowledge_tensor[vector_idx[:,1] , vector_idx[:,0]] += 1
                
                if normalize:
                    if normalize_method == 'max':
                        # normalize the numpy [0, 1]
                        knowledge_tensor = knowledge_tensor / knowledge_tensor.sum(dim=1 , keepdim=True)
                    elif normalize_method == 'binary':
                        knowledge_tensor = (knowledge_tensor > 0).float()
                
                # change nan to 0
                knowledge_tensor[torch.isnan(knowledge_tensor)] = 0
                
                synthetic_tensor[int(label)] = knowledge_tensor
                pbar.update(1)
            
        ## sort the tensor based on the key
        synthetic_tensor = dict(sorted(synthetic_tensor.items()))
        
        logger.info("Saving Synthetic Knowledge Tensor")
        torch.save(synthetic_tensor , os.path.join(self.config.root_dir , self.dataset , f"knowledge_synthetic_test_{self.omic_type}.pt"))
        
        return synthetic_tensor
    
    def __measure_graph(self , graph_tensor: torch.Tensor):
        """measure graph matrix 
        
        Calculate average no of edges in the graph

        Args:
            graph_tensor (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        
        assert graph_tensor.dim() == 2 , "Graph tensor should be 2D"
        
        avg_no_of_edges = graph_tensor.count_nonzero(dim=1).float().mean().item()
        
        return avg_no_of_edges
    
    def __measure_graph_from_data(self , data_list:List[Data] , name:str):
        """_summary_

        Args:
            data (Data): _description_

        Returns:
            _type_: _description_
        """
        num_nodes = []
        num_edges = []
        undirected_graph_count = 0
        directed_graph_count = 0 
        
        for data in data_list:
            num_nodes.append(data.num_nodes)
            num_edges.append(data.edge_index.shape[1] / 2)
            if is_undirected(data.edge_index):
                undirected_graph_count += 1
            else:
                directed_graph_count += 1
        
        self.dataset_summary.append({
            "name":name,
            "avg_no_of_nodes":np.mean(num_nodes),
            "avg_no_of_edges":np.mean(num_edges), 
            "no_of_undirected_graph":undirected_graph_count,
            "no_of_directed_graph":directed_graph_count
        })
    
    def generate_contrastive_multigraph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.synthetic_graph
        
        logger.info("Generate training contrastive multigraph")
        training_graphs = []
        labels = synthetic_tensor_dict.keys()
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                label = int(self.train_label.iloc[idx].values.item())
                non_label = [x for x in labels if x != label]
                positive_topology = synthetic_tensor_dict[label] + knowledge_tensor
                negative_topology = synthetic_tensor_dict[np.random.choice(non_label)] + knowledge_tensor
            
                # positive graph
                positive_coo_matrix = symmetric_matrix_to_coo(positive_topology.numpy() , self.config.edge_threshold)
                positive_graph = coo_to_pyg_data(coo_matrix=positive_coo_matrix , node_features=torch_sample , y = torch.tensor([1] , dtype=torch.long) , extra_label=True )
                
                # negative graph
                negative_coo_matrix = symmetric_matrix_to_coo(negative_topology.numpy() , self.config.edge_threshold)
                negative_graph = coo_to_pyg_data(coo_matrix=negative_coo_matrix , node_features=torch_sample , y = torch.tensor([0] , dtype=torch.long) , extra_label=True )
                
                training_graphs.append([positive_graph , negative_graph])
                pbar.update(1)
        
        logger.info("Generate testing contrastive multigraph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                topology = knowledge_tensor
                graphs = []
                for synthetic_graph in synthetic_tensor_dict.values():
                    coo_matrix = symmetric_matrix_to_coo((topology + synthetic_graph).numpy() , self.config.edge_threshold)
                    graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                    graphs.append(graph)
                testing_graphs.append(graphs)
                pbar.update(1)
        
        # Measure 
        # self.__measure_graph_from_data(training_graphs , f"Training Unified Graph")
        # self.__measure_graph_from_data(testing_graphs , f"Testing Unified Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_contrastive_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_constrastive_multigraphs_omic_{self.omic_type}.pt"))
        
    def generate_triplet_multigraph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.__generate_ppi_graph()
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.__generate_kegg_go_graph()
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.__generate_synthetic_graph(normalize_method='binary')
        
        logger.info("Generate training triplet multigraph")
        training_graphs = []
        labels = synthetic_tensor_dict.keys()
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                label = int(self.train_label.iloc[idx].values.item())
                same_label_sample_idx = [x for x in self.train_label[self.train_label[0] == label].index.to_list() if x != idx ]
                same_label = int(self.train_label.iloc[random.choice(same_label_sample_idx)].values.item())
                
                non_label = [x for x in labels if x != label]
                if label in synthetic_tensor_dict.keys():
                    positive_topology = (synthetic_tensor_dict[label] + knowledge_tensor > 0).float()
                else:
                    positive_topology = (knowledge_tensor > 0).float()
                random_negative_label = np.random.choice(non_label)
                if random_negative_label in synthetic_tensor_dict.keys():
                    negative_topology = (synthetic_tensor_dict[random_negative_label] + knowledge_tensor > 0).float()
                else:
                    negative_topology = (knowledge_tensor > 0).float()

                # anchor graph 
                anchor_coo_matrix = symmetric_matrix_to_coo(positive_topology.numpy() , self.config.edge_threshold)
                anchor_graph = coo_to_pyg_data(coo_matrix=anchor_coo_matrix , node_features=torch_sample , y = torch.tensor([label] , dtype=torch.long) , extra_label=True )
                
                # positive graph (From another sample with same label)
                positive_torch_sample = torch.tensor(self.train_data.iloc[random.choice(same_label_sample_idx)].values, dtype=torch.float32 , device=device).unsqueeze(-1)
                positive_graph = coo_to_pyg_data(coo_matrix=anchor_coo_matrix , node_features=positive_torch_sample , y = torch.tensor([same_label] , dtype=torch.long) , extra_label=True )
                
                # negative graph (Same sample with different topology)
                negative_coo_matrix = symmetric_matrix_to_coo(negative_topology.numpy() , self.config.edge_threshold)
                negative_graph = coo_to_pyg_data(coo_matrix=negative_coo_matrix , node_features=torch_sample , y = torch.tensor([random_negative_label] , dtype=torch.long) , extra_label=True )
                
                training_graphs.append([anchor_graph ,  positive_graph , negative_graph])
                pbar.update(1)
        
        logger.info("Generate testing triplet multigraph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                # topology = knowledge_tensor
                graphs = []
                for i in range(0 , 5):
                    
                    if i in synthetic_tensor_dict.keys():
                        synthetic_graph = synthetic_tensor_dict[i]
                        topology = (synthetic_graph + knowledge_tensor > 0).float()
                    else:
                        topology = (knowledge_tensor > 0).float()
                    
                    coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                    graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                    graphs.append(graph)
                testing_graphs.append(graphs)
                pbar.update(1)
        
        # Measure 
        # self.__measure_graph_from_data(training_graphs , f"Training Unified Graph")
        # self.__measure_graph_from_data(testing_graphs , f"Testing Unified Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_triplet_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_triplet_multigraphs_omic_{self.omic_type}.pt"))
        
    def generate_binaryclassifier_multigraph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.__generate_ppi_graph()
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.__generate_kegg_go_graph()
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.__generate_synthetic_graph(normalize_method='binary')
        
        logger.info("Generate training binary classifier multigraph")
        training_graphs = []
        labels = synthetic_tensor_dict.keys()
        with tqdm(total=self.train_data.shape[0]) as pbar:
            
            if self.config.discretized: 
                self.train_data = self.__load_kbin_model(self.train_data)
                
            for idx , sample in self.train_data.iterrows():
                
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                label = int(self.train_label.iloc[idx].values.item())
                non_label = np.random.choice([x for x in labels if x != label])
                random_label = random.choice([label , non_label])
                if random_label in synthetic_tensor_dict.keys():
                    topology = synthetic_tensor_dict[random_label] + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                # convert to 0 and 1 
                topology = (topology > 0).float()
                    
                # random graph
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor([random_label == label], dtype=torch.long) , extra_label=True )
                
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info("Generate testing  binary classifier multigraph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            
            if self.config.discretized:
                self.test_data = self.__load_kbin_model(self.test_data)
                
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                topology = knowledge_tensor
                graphs = []
                for i in range(0,5):
                    
                    if i in synthetic_tensor_dict.keys():
                        synthetic_graph = synthetic_tensor_dict[i]
                        topology = synthetic_graph + knowledge_tensor
                    else:
                        topology = knowledge_tensor
                    
                    # convert to 0 and 1 
                    topology = (topology > 0).float()
                    
                    coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                    graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                    graphs.append(graph)
                    
                testing_graphs.append(graphs)
                pbar.update(1)
        
        # Measure 
        # self.__measure_graph_from_data(training_graphs , f"Training Unified Graph")
        # self.__measure_graph_from_data(testing_graphs , f"Testing Unified Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_binaryclassifier_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_binaryclassifier_multigraphs_omic_{self.omic_type}.pt"))
        
    def generate_unified_multigraph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.synthetic_graph
        
        logger.info("Generate training unified multigraph")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    topology = synthetic_tensor_dict[label] + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info("Generate testing unified multigraph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                topology = knowledge_tensor
                graphs = []
                for synthetic_graph in synthetic_tensor_dict.values():
                    coo_matrix = symmetric_matrix_to_coo((topology + synthetic_graph).numpy() , self.config.edge_threshold)
                    graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                    graphs.append(graph)
                testing_graphs.append(graphs)
                pbar.update(1)
        
        # Measure 
        # self.__measure_graph_from_data(training_graphs , f"Training Unified Graph")
        # self.__measure_graph_from_data(testing_graphs , f"Testing Unified Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_unified_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_unified_multigraphs_omic_{self.omic_type}.pt"))
        
    def generate_unified_test_multigraph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.__generate_ppi_graph() # self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.__generate_kegg_go_graph() # self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.__generate_synthetic_graph(normalize_method='binary')  # self.synthetic_graph
            synthetic_tensor_dict_test = self.__generate_synthetic_test_graph(normalize_method='binary') # self.synthetic_graph_test
        
        logger.info("Generate training unified multigraph (synthetic train)")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    if label in synthetic_tensor_dict.keys():
                        topology = synthetic_tensor_dict[label] + knowledge_tensor
                    else:
                        topology = knowledge_tensor
                else: 
                    topology = knowledge_tensor
                
                # convert to binary 
                topology = (topology > 0).float()
                    
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info(f"Generate testing unified multigraph (synthetic test) : {synthetic_tensor_dict_test.keys()}")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                
                graphs = []
                for i in range(0 , 5):
                #for synthetic_graph in synthetic_tensor_dict_test.values():
                    if i in synthetic_tensor_dict_test.keys():
                        synthetic_graph = synthetic_tensor_dict_test[i]
                        topology = knowledge_tensor + synthetic_graph
                    else:
                        topology = knowledge_tensor 
                    topology = (topology > 0).float()
                    coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                    graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                    graphs.append(graph)
                testing_graphs.append(graphs)
                pbar.update(1)
        
        # Measure 
        # self.__measure_graph_from_data(training_graphs , f"Training Unified Graph")
        # self.__measure_graph_from_data(testing_graphs , f"Testing Unified Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_unified_test_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_unified_test_multigraphs_omic_{self.omic_type}.pt"))
        
    def generate_unified_graph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.synthetic_graph
        
        logger.info("Generate training unified graph")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    topology = synthetic_tensor_dict[label] + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info("Generate testing unified graph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                topology = knowledge_tensor
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                testing_graphs.append(graph)
                pbar.update(1)
        
        # Measure 
        self.__measure_graph_from_data(training_graphs , f"Training Unified Graph")
        self.__measure_graph_from_data(testing_graphs , f"Testing Unified Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_unified_graphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_unified_graphs_omic_{self.omic_type}.pt"))
    
    def generate_multiedges_graph(self , ppi=True , kegg_go=False , synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor = torch.stack(list(self.synthetic_graph.values()) , dim=-1) # shape => no_of_genes , no_of_genes , no_of_synthetic_graph
        
        logger.info("Generate training multiedges graph")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    topology = synthetic_tensor # + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                # coo_matrix = symmetric_matrix_to_coo(topology.numpy())
                graph = symmetric_matrix_to_pyg(
                    matrix=topology.numpy() , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , edge_threshold=self.config.edge_threshold
                )
                training_graphs.append(graph)
                pbar.update(1)
                
        logger.info("Generate testing multiedges graph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic:
                    topology = synthetic_tensor # + synthetic_tensor
                else:
                    topology = knowledge_tensor
                #coo_matrix = symmetric_matrix_to_coo(topology.numpy())
                graph = symmetric_matrix_to_pyg(
                    matrix=topology.numpy() , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , edge_threshold=self.config.edge_threshold
                )
                testing_graphs.append(graph)
                pbar.update(1)
                
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_multiedges_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_multiedges_multigraphs_omic_{self.omic_type}.pt"))
    
    def generate_discretized_multiedges_graph(self , ppi=True , kegg_go=False , synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        topology_tensor_stack = []
        
        if ppi:
            ppi_tensor = self.__generate_ppi_graph(normalize_method='binary')
            topology_tensor_stack.append(ppi_tensor)
        
        if kegg_go:
            kegg_pathway_tensor = self.__generate_kegg_go_graph(normalize_method='binary')
            topology_tensor_stack.append(kegg_pathway_tensor)
        
        if synthetic:
            synthetic_tensor_dict = self.__generate_synthetic_graph(topk=self.config.topk , normalize_method='binary')
            #synthetic_tensor = torch.stack(list(synthetic_tensor_dict.values()) , dim=-1) # shape => no_of_genes , no_of_genes , no_of_synthetic_graph
            topology_tensor_stack.extend(list(synthetic_tensor_dict.values()))
            #synthetic_tensor = torch.stack(list(self.synthetic_graph.values()) , dim=-1) # shape => no_of_genes , no_of_genes , no_of_synthetic_graph
        
        logger.info("Generate training discretized multiedges graph")
        training_graphs = []
        num_edges = []
        
        # discretize the data
        if self.config.discretized: 
            kbin = self.__load_kbin_model()
            self.train_data = pd.DataFrame(kbin.fit_transform(self.train_data))
            
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                #sample_value = self.kbin_model.transform(sample.values.reshape(1, -1))[0]
                if self.config.discretized:
                    torch_sample = torch.tensor(sample.values, dtype=torch.long , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                    torch_sample = one_hot(torch_sample.long(), num_classes=self.config.n_bins).squeeze(1).to(device)
                else:
                    torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                topology = torch.stack(topology_tensor_stack , dim=-1)
                    
                # coo_matrix = symmetric_matrix_to_coo(topology.numpy())
                graph = symmetric_matrix_to_pyg(
                    matrix=topology.numpy() , node_features=torch_sample , 
                    y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) ,
                    edge_threshold=self.config.edge_threshold
                )
                num_edges.append(graph.edge_index.shape[1])
                training_graphs.append(graph)
                pbar.update(1)
        # calculate average no of edges and std deviation
        avg_no_of_edges = np.mean(num_edges)
        std_no_of_edges = np.std(num_edges)
        logger.info(f"Average no of edges in testing graph: {avg_no_of_edges} +- {std_no_of_edges}")
                
        logger.info("Generate testing discretized multiedges graph")
        testing_graphs = []
        num_edges = []
        if self.config.discretized:
            self.test_data = pd.DataFrame(kbin.transform(self.test_data))
            
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                #sample_value = self.kbin_model.transform(sample.values.reshape(1, -1))[0]
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                topology = torch.stack(topology_tensor_stack , dim=-1)
                #coo_matrix = symmetric_matrix_to_coo(topology.numpy())
                graph = symmetric_matrix_to_pyg(
                    matrix=topology.numpy() , node_features=torch_sample , 
                    y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , 
                    edge_threshold=self.config.edge_threshold
                )
                num_edges.append(graph.edge_index.shape[1])
                testing_graphs.append(graph)
                pbar.update(1)
        # calculate average no of edges and std deviation
        avg_no_of_edges = np.mean(num_edges)
        std_no_of_edges = np.std(num_edges)
        logger.info(f"Average no of edges in testing graph: {avg_no_of_edges} +- {std_no_of_edges}")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_discretized_multiedges_multigraphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_discretized_multiedges_multigraphs_omic_{self.omic_type}.pt"))
                
                            
    def generate_common_graph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor = torch.zeros_like(knowledge_tensor)
            
            for label in self.synthetic_graph.keys():
                synthetic_tensor += self.synthetic_graph[label]
        
        logger.info("Generate training common graph")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    topology = synthetic_tensor + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info("Generate testing common graph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic:
                    topology = knowledge_tensor + synthetic_tensor
                else:
                    topology = knowledge_tensor
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                testing_graphs.append(graph)
                pbar.update(1)
        
        # Measure 
        self.__measure_graph_from_data(training_graphs , f"Training Common Graph")
        self.__measure_graph_from_data(testing_graphs , f"Testing Common Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_common_graphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_common_graphs_omic_{self.omic_type}.pt"))
    
    def generate_discretized_common_graph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor = torch.zeros_like(knowledge_tensor)
            
            for label in self.synthetic_graph.keys():
                synthetic_tensor += self.synthetic_graph[label]
        
        logger.info("Generate training common graph")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                sample_value = self.kbin_model.transform([sample.values])
                torch_sample = torch.tensor(sample_value[0], dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    topology = synthetic_tensor + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                graph = symmetric_matrix_to_pyg(
                    matrix=topology.numpy() , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , edge_threshold=self.config.edge_threshold
                )
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info("Generate testing common graph")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                sample_value = self.kbin_model.transform([sample.values])
                torch_sample = torch.tensor(sample_value[0], dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic:
                    topology = knowledge_tensor + synthetic_tensor
                else:
                    topology = knowledge_tensor
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True)
                testing_graphs.append(graph)
                pbar.update(1)
        
        # Measure 
        self.__measure_graph_from_data(training_graphs , f"Training Discretized Common Graph")
        self.__measure_graph_from_data(testing_graphs , f"Testing Discretized Common Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_discretized_common_graphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_discretized_common_graphs_omic_{self.omic_type}.pt"))
        
    
    def generate_correlation_graph(self , ppi=True , kegg_go=False, synthetic=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0]
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if synthetic:
            synthetic_tensor_dict = self.synthetic_graph
        
        logger.info("Generate training correlation graph")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1) # shape => number_of_node , 1 (gene expression)
                
                if synthetic: 
                    label = int(self.train_label.iloc[idx].values.item())
                    # print(synthetic_tensor_dict.keys())
                    # print(label)
                    # print(type(label))
                    topology = synthetic_tensor_dict[label] + knowledge_tensor
                else: 
                    topology = knowledge_tensor
                    
                coo_matrix = symmetric_matrix_to_coo(topology.numpy() , self.config.edge_threshold)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                training_graphs.append(graph)
                pbar.update(1)
        
        logger.info("Generate testing correlation graph")
        testing_graphs = []
        # Get correlation graph for test dataset 
        corr_matrix = self.test_data.corr().to_numpy()
        coo_matrix = symmetric_matrix_to_coo(corr_matrix , self.config.edge_threshold)
        
        with tqdm(total=self.test_data.shape[0]) as pbar:
            for idx , sample in self.test_data.iterrows():
                # TO-DO 
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=torch_sample , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                testing_graphs.append(graph)
                pbar.update(1)
        
        
        # Measure 
        self.__measure_graph_from_data(training_graphs , f"Training Correlation Graph")
        self.__measure_graph_from_data(testing_graphs , f"Testing Correlation Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_corr_graphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_corr_graphs_omic_{self.omic_type}.pt"))
        
    
    def generate_knowledge_graph(self, ppi=True , kegg_go=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # feature dimension (no of genes)
        no_of_genes = self.feature_names.shape[0] 
        knowledge_tensor = torch.zeros(no_of_genes, no_of_genes)
        
        if ppi:
            partial_knowledge_tensor = self.ppi_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        if kegg_go:
            partial_knowledge_tensor = self.kegg_go_graph_tensor
            knowledge_tensor += partial_knowledge_tensor
        
        # normalize the knowledge tensor
        knowledge_tensor = knowledge_tensor / knowledge_tensor.sum(dim=1 , keepdim=True)
        coo_matrix = symmetric_matrix_to_coo(knowledge_tensor.numpy() , 0.1)
        
        logger.info("Generating Assembled Knowledge Graph [Training Graph]")
        training_graphs = []
        with tqdm(total=self.train_data.shape[0]) as pbar:
            for idx , sample in self.train_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1)
                node_embedding = torch.concat([torch_sample , self.embedding] , dim=-1)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=node_embedding , y = torch.tensor(self.train_label.iloc[idx].values , dtype=torch.long) , extra_label=True )
                training_graphs.append(graph)
                pbar.update(1)
                
        
        logger.info("Generating Assembled Knowledge Graph [Testing Graph]")
        testing_graphs = []
        with tqdm(total=self.test_data.shape[0]) as pbar: 
            for idx , sample in self.test_data.iterrows():
                torch_sample = torch.tensor(sample.values, dtype=torch.float32 , device=device).unsqueeze(-1)
                node_embedding = torch.concat([torch_sample , self.embedding] , dim=-1)
                graph = coo_to_pyg_data(coo_matrix=coo_matrix , node_features=node_embedding , y = torch.tensor(self.test_label.iloc[idx].values , dtype=torch.long) )
                testing_graphs.append(graph)
                pbar.update(1)
        
        # Measure 
        self.__measure_graph_from_data(training_graphs , f"Training Embedding Graph")
        self.__measure_graph_from_data(testing_graphs , f"Testing Embedding Graph")
        
        # Save the graphs 
        logger.info("Saving Training Graphs")
        os.makedirs(os.path.join(self.config.root_dir , self.dataset) , exist_ok=True)
        torch.save(training_graphs , os.path.join(self.config.root_dir , self.dataset , f"training_embedding_graphs_omic_{self.omic_type}.pt"))
        logger.info("Saving Testing Graphs")
        torch.save(testing_graphs , os.path.join(self.config.root_dir , self.dataset , f"testing_embedding_graphs_omic_{self.omic_type}.pt"))
        
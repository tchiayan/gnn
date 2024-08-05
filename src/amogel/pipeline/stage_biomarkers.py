from amogel.config.configuration import ConfigurationManager
from amogel.utils.gene import biomarkers_selection
from amogel.utils.common import load_omic_features_name 
from amogel import logger 
import torch
import os 
import pandas as pd
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import seaborn as sns 
import math
import warnings 

warnings.filterwarnings("ignore")

class BiomarkersPipeline:
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.dataset = self.config.get_dataset()
        self.topk = self.config.get_topk()
        self.top_gene = self.config.get_biomarkers()

    def run(self):
        batch_file = f"./artifacts/amogel/batches_{self.topk}.pt"
        edge_attn_file = [
            f"./artifacts/amogel/edge_attn_l1_{self.topk}.pt", 
            f"./artifacts/amogel/edge_attn_l2_{self.topk}.pt"
        ]
        biomarkers , summarized_edges = biomarkers_selection(
            batch_file=batch_file , 
            edge_attn_files=edge_attn_file , 
            topk=self.top_gene
        )
        
        # saved the summarize_edge 
        os.makedirs("./artifacts/biomarkers/" , exist_ok=True)
        torch.save(summarized_edges , f"./artifacts/biomarkers/graph_edges.pt")
        print("----- Summarize edges information -----")
        print(f"-- Edges matrix shape : {summarized_edges.shape}")
        assert summarized_edges.shape[0] == summarized_edges.shape[1] , "Number of genes must be equal"
        #print(biomarkers)
        
        # load genes name 
        omic_types = [ 1 , 2 , 3 ]
        feature_names = load_omic_features_name(
            "artifacts/data_preprocessing/" , self.dataset , [1,2,3]
        )
        
        # ac selected genes 
        ac_genes = torch.load("artifacts/ac_genes/gene.pt" , map_location=torch.device("cpu"))
        # selection mapping 
        map_ac_genes = {k:v for k,v in enumerate(ac_genes)}
        #print(feature_names[feature_names['gene_idx'].isin(list(ac_genes))])
        top_n = 10
        top_N_gene_idx = [map_ac_genes[selected_genes] for selected_genes in biomarkers.indices.numpy()[:top_n]]
        top_N_gene_names = [ feature_names[feature_names['gene_idx'].isin(list(ac_genes))].iloc[gx , 1] for gx in biomarkers.indices.numpy()[:top_n]]
        #print(top_N_gene_name)
        self.plot_t_test(top_N_gene_idx , top_N_gene_names)
        
        selected_features = feature_names[feature_names['gene_idx'].isin(list(ac_genes))].iloc[biomarkers.indices.numpy(),:]
        
        print("==== TopN from all omic type ====")
        for i in top_N_gene_names: 
            print(i)
        print("==================================")
        
        for i in omic_types:
            print(f"Top genes from omic type {i}") 
            for idx , row in selected_features[selected_features['omic_type'] == i].iterrows():
                print(row['gene_name'])
                
            print("------ end of list ------")
    
    def plot_t_test(self, top_n_idx , top_n_names):
        # load gene expression for all the omics data 
        test_data_omic_1 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"1_te.csv"), header=None)
        test_data_omic_2 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"2_te.csv"), header=None)
        test_data_omic_3 = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset , f"3_te.csv"), header=None)
        
        # join 
        test_data = pd.concat([test_data_omic_1 , test_data_omic_2 , test_data_omic_3 ], axis=1)
        # rename 
        test_data.columns = [x for x in range(test_data.shape[1])]
        
        # load label 
        test_label = pd.read_csv(os.path.join("./artifacts/data_preprocessing" , self.dataset, "labels_te.csv") , header=None , names=['label'])
        
        f_statistic , p_value = f_regression(test_data , test_label)
        
        df = test_data.join(test_label)
        num_of_rows = math.ceil(len(top_n_idx) / 4)
        fig , axis = plt.subplots(num_of_rows, 4, figsize=(20, 20))
        #fig.tight_layout()
        plt.subplots_adjust(hspace=0.5 , wspace=0.3)
        
        for row in range(num_of_rows):
            for col in range(4):
                image_idx = row * 4 + col 
                if image_idx > len(top_n_idx)-1:
                    break
                print(image_idx , top_n_idx)
                sns.boxplot(x='label' , y=top_n_idx[image_idx] , data=df , color="gray" , ax=axis[row , col])
                
                axis[row , col].set_title(f'Gene {top_n_names[image_idx]} (p-value={p_value[top_n_idx[image_idx]]:.1E})')
                axis[row , col].set_xlabel('Class')
                axis[row , col].set_ylabel('Gene Expression')
                
        fig.savefig('./artifacts/biomarkers/top_n_t_test.png')
        
if __name__ == "__main__":
    pipeline = BiomarkersPipeline()
    pipeline.run()